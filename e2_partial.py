"""E2 staged learning rate searches followed by three seed evaluation"""
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from run_outputs import result_directory
from e1_head_only import (
    LABELS, best_trial, configure_output, decode_document, evaluate_standardized_records,
    learning_rate_factor, load_e1_split, make_collator, release_device_memory,
    save_json, select_device, summarize_seeds, synchronize_device, training_class_weights,
)


# Train only the final four encoder layers and the head
def configure_e2(model, encoder_learning_rate=2e-5, head_learning_rate=1e-4):
    """Fix E2 to four trainable layers and create disjoint optimizer groups."""
    layers = model.base_model.encoder.layer
    if len(layers) != 12:
        raise ValueError('E2 requires the 12-layer BioBERT backbone')
    for parameter in model.parameters():
        parameter.requires_grad = False
    encoder_parameters = []
    for layer in layers[8:12]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
            encoder_parameters.append(parameter)
    head_parameters = list(model.classifier.parameters())
    for parameter in head_parameters:
        parameter.requires_grad = True
    expected = {id(p) for p in encoder_parameters + head_parameters}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if actual != expected or len(expected) != len(encoder_parameters) + len(head_parameters):
        raise ValueError('E2 parameter groups must cover the trainable parameters exactly once')
    return [
        {'params': encoder_parameters, 'lr': encoder_learning_rate},
        {'params': head_parameters, 'lr': head_learning_rate},
    ]


# Keep the frozen prefix deterministic during training
def set_e2_training_mode(model):
    """Enable dropout in the trainable layers/head, keep the frozen prefix in eval."""
    model.train()
    model.base_model.embeddings.eval()
    for layer in model.base_model.encoder.layer[:8]:
        layer.eval()


# Edit these settings before direct execution
@dataclass(frozen=True)
class E2Config:
    root: Path = Path(__file__).resolve().parent
    output_dir: Path = field(default_factory=lambda: Path('results/completed/5. fine-tune/5.2 E2') /
                             datetime.now(ZoneInfo('Australia/Sydney')).strftime('run_%Y%m%d_%H%M%S_%f'))
    e1_config_path: Path | None = None
    model: str = 'dmis-lab/biobert-base-cased-v1.1'
    epochs: int = 30
    batch_size: int = 8
    weight_decay: float = 0.01
    scheduler: str = 'constant'
    patience: int = 3
    loss: str = 'cross_entropy'
    seed: int = 9444
    dropout: float = 0.1
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 7.5e-4
    max_length: int = 512
    stride: int = 64
    micro_batch_size: int = 8
    cuda_fast_mode: bool = True
    cuda_evaluation_batch_size: int = 32
    limit_docs: int | None = None
    evaluate_test: bool = True


# Inherit hyperparameters from a completed formal E1 run
def inherit_e1_config(config):
    source = config.e1_config_path
    if source is None:
        folder = config.root / 'results/completed/5. fine-tune/5.1 E1'
        candidates = sorted(folder.glob('run_*/selected_config.json'), reverse=True)
        source = next((path for path in candidates
                       if (path.parent / '5.1.5 Seeds and final results/summary.json').exists()
                       and json.loads(path.read_text()).get('limit_docs') is None), None)
        if source is None:
            raise ValueError('No completed formal E1 run found set e1_config_path to its selected_config.json')
    source = config.root / source
    values = json.loads(source.read_text(encoding='utf-8'))
    if values.get('limit_docs') is not None:
        raise ValueError('E2 cannot inherit smoke test hyperparameters')
    keys = ('model', 'epochs', 'batch_size', 'weight_decay', 'scheduler', 'patience',
            'loss', 'dropout', 'max_length', 'stride')
    inherited = {key: values[key] for key in keys}
    inherited.update(head_learning_rate=values['learning_rate'], e1_config_path=source.resolve())
    return replace(config, **inherited)


# Enable fast CUDA kernels while keeping a portable fallback
def configure_runtime(config, device):
    import torch

    fast = config.cuda_fast_mode and device.type == 'cuda'
    torch.use_deterministic_algorithms(not fast, warn_only=True)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = fast
        torch.backends.cudnn.deterministic = not fast
        torch.set_float32_matmul_precision('high' if fast else 'highest')
    return torch.bfloat16 if fast and torch.cuda.is_bf16_supported() else None


# Use mixed precision only on the CUDA fast path
def precision_context(device, amp_dtype):
    import torch

    return torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()


# Pad CUDA sequences to a Tensor Core friendly multiple
def make_e2_collator(tokenizer, device, align=False):
    import torch.nn.functional as functional

    base = make_collator(tokenizer, device)

    def collate(windows):
        batch, inputs = base(windows)
        padding = (-inputs['input_ids'].shape[1]) % 8 if align else 0
        if padding:
            for key, value in inputs.items():
                fill = -100 if key == 'labels' else tokenizer.pad_token_id if key == 'input_ids' else 0
                inputs[key] = functional.pad(value, (0, padding), value=fill)
        return batch, inputs

    return collate


# Keep the MPS memory fallback separate from CUDA throughput settings
def execution_batch_sizes(config, device):
    if device.type == 'cuda':
        return config.micro_batch_size, config.cuda_evaluation_batch_size
    return min(config.micro_batch_size, 2), min(config.micro_batch_size, 2)


# Use the same document level metric as E1 with a trainable encoder
def evaluate_e2(model, records, windows, batch_size, collate, amp_dtype=None):
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    sums = [torch.zeros((len(record['input_ids']), len(LABELS))) for record in records]
    counts = [torch.zeros(len(record['input_ids'])) for record in records]
    votes = [{} for _ in records]
    with torch.no_grad():
        for batch, inputs in DataLoader(windows, batch_size=batch_size, collate_fn=collate):
            inputs.pop('labels')
            with precision_context(inputs['input_ids'].device, amp_dtype):
                logits = model(**inputs).logits.float().cpu()
            for row, window in enumerate(batch):
                doc, start, end = window['doc'], window['start'], window['end']
                values = logits[row, 1:1 + end - start]
                sums[doc][start:end] += values
                counts[doc][start:end] += 1
                for position, label in enumerate(values.argmax(-1).tolist(), start):
                    votes[doc].setdefault(position, set()).add(label)
            del logits, inputs
    predictions, illegal, conflicts = [], 0, 0
    for record, total, count, doc_votes in zip(records, sums, counts, votes):
        if not (count > 0).all():
            raise ValueError('Uncovered document tokens')
        prediction, bad = decode_document(record, (total / count[:, None]).argmax(-1))
        predictions.append(prediction)
        illegal += bad
        conflicts += sum(len(labels) > 1 for labels in doc_votes.values())
    metrics = evaluate_standardized_records(records, predictions)
    metrics.update(illegal_bio_transitions=illegal, window_overlap_token_conflicts=conflicts)
    return metrics, predictions


# Normalize micro batch gradients over the full logical batch
def backward_e2_batch(model, inputs, weights, micro_batch_size, amp_dtype=None, return_tensors=False):
    import torch
    from torch.nn.functional import cross_entropy

    labels = inputs['labels']
    valid = labels != -100
    denominator = (weights[labels.clamp_min(0)] * valid).sum() if weights is not None else valid.sum()
    numerator = torch.zeros((), device=labels.device, dtype=torch.float32)
    for start in range(0, labels.shape[0], micro_batch_size):
        targets = labels[start:start + micro_batch_size]
        chunk = {key: value[start:start + micro_batch_size] for key, value in inputs.items() if key != 'labels'}
        with precision_context(labels.device, amp_dtype):
            logits = model(**chunk).logits
            loss_sum = cross_entropy(logits.reshape(-1, len(LABELS)), targets.reshape(-1),
                                     weight=weights, ignore_index=-100, reduction='sum')
            loss = loss_sum / denominator
        loss.backward()
        numerator += loss_sum.detach()
        del logits, loss_sum, loss, chunk
    if return_tensors:
        return numerator, denominator.detach()
    return numerator.item(), denominator.item()


# Save only updated parameters and reconstruct frozen weights from BioBERT
def save_trainable_checkpoint(model, path):
    import torch

    torch.save(snapshot_trainable_parameters(model), path)


# Keep the best trainable weights in RAM until the trial finishes
def snapshot_trainable_parameters(model):
    return {name: parameter.detach().to('cpu', copy=True) for name, parameter in model.named_parameters()
            if parameter.requires_grad}


# Require exact coverage of the four trainable layers and the head
def restore_trainable_checkpoint(model, path):
    import torch

    state = torch.load(path, map_location='cpu', weights_only=True)
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError('E2 checkpoint does not match the trainable parameter scope')
    model.load_state_dict(state, strict=False)


# Initialize every trial from original BioBERT with a seeded new head
def run_trial(config, output_dir, data):
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, BertConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f'\nTrial: {output_dir} | seed={config.seed} | encoder LR={config.encoder_learning_rate:g} | '
          f'head LR={config.head_learning_rate:g}', flush=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    tokenizer, train_records, train_windows, valid_records, valid_windows = data
    device = torch.device('cuda') if torch.cuda.is_available() else select_device(torch)
    amp_dtype = configure_runtime(config, device)
    micro_batch_size, evaluation_batch_size = execution_batch_sizes(config, device)
    model_config = BertConfig.from_pretrained(
        config.model, num_labels=len(LABELS), classifier_dropout=config.dropout,
        id2label=dict(enumerate(LABELS)), label2id={label: index for index, label in enumerate(LABELS)})
    model = AutoModelForTokenClassification.from_pretrained(
        config.model, config=model_config, **({'attn_implementation': 'sdpa'} if device.type == 'cuda' and config.cuda_fast_mode else {}))
    groups = configure_e2(model, config.encoder_learning_rate, config.head_learning_rate)
    if config.max_length > model.config.max_position_embeddings:
        raise ValueError('Window length exceeds model position embeddings')
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay,
                                  **({'fused': True} if device.type == 'cuda' and config.cuda_fast_mode else {}))
    total_steps = math.ceil(len(train_windows) / config.batch_size) * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, total_steps, config.scheduler))
    weights = training_class_weights(train_windows, torch).to(device) if config.loss == 'weighted_cross_entropy' else None
    collate = make_e2_collator(tokenizer, device, align=device.type == 'cuda' and config.cuda_fast_mode)
    loader = DataLoader(train_windows, batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed), collate_fn=collate)
    metadata = asdict(config)
    metadata.update(device=str(device),
                    gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    amp_dtype=str(amp_dtype), actual_micro_batch_size=micro_batch_size,
                    evaluation_batch_size=evaluation_batch_size,
                    attention_implementation=model.config._attn_implementation,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    cuda_version=torch.version.cuda, trainable_parameters=sum(p.numel() for p in trainable),
                    train_documents=len(train_records), validation_documents=len(valid_records),
                    trainable_layers=[8, 9, 10, 11], labels=LABELS,
                    class_weights=weights.cpu().tolist() if weights is not None else None,
                    torch_version=torch.__version__,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   (Path(__file__), Path(__file__).with_name('e1_head_only.py'))})
    save_json(output_dir / 'config.json', metadata)
    print(f'Device: {device} | Trainable parameters: {metadata["trainable_parameters"]:,} | '
          f'batch={config.batch_size} | micro batch={micro_batch_size} | '
          f'precision={amp_dtype or torch.float32} | attention={model.config._attn_implementation}', flush=True)
    synchronize_device(torch, device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best_f1, stale, history = -1.0, 0, []
    training_seconds, validation_seconds = 0.0, 0.0
    best_state = None
    try:
        for epoch in range(1, config.epochs + 1):
            epoch_started = time.perf_counter()
            set_e2_training_mode(model)
            numerator = torch.zeros((), device=device)
            denominator = torch.zeros((), device=device)
            rates = [group['lr'] for group in optimizer.param_groups]
            for _, inputs in loader:
                optimizer.zero_grad(set_to_none=True)
                loss_sum, count = backward_e2_batch(model, inputs, weights, micro_batch_size, amp_dtype, return_tensors=True)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                numerator += loss_sum
                denominator += count
                del inputs
            optimizer.zero_grad(set_to_none=True)
            synchronize_device(torch, device)
            training_loss = (numerator / denominator).item()
            train_time = time.perf_counter() - epoch_started
            validation_started = time.perf_counter()
            metrics, predictions = evaluate_e2(model, valid_records, valid_windows, evaluation_batch_size, collate, amp_dtype)
            synchronize_device(torch, device)
            valid_time = time.perf_counter() - validation_started
            training_seconds += train_time
            validation_seconds += valid_time
            f1 = metrics['typed_exact']['f1']
            history.append({'epoch': epoch, 'training_loss': training_loss, 'validation_f1': f1,
                            'encoder_learning_rate': rates[0], 'head_learning_rate': rates[1],
                            'training_seconds': train_time, 'validation_seconds': valid_time})
            if f1 > best_f1:
                best_f1, best_epoch, stale = f1, epoch, 0
                best_metrics = metrics
                best_state = snapshot_trainable_parameters(model)
                save_json(output_dir / 'validation_metrics.json', metrics)
                save_json(output_dir / 'validation_predictions.json', predictions)
            else:
                stale += 1
            save_json(output_dir / 'history.json', history)
            print(f'Epoch {epoch:02d}/{config.epochs} | loss={training_loss:.4f} | '
                  f'Val F1={f1:.2%} | best={best_f1:.2%} at epoch {best_epoch} | '
                  f'encoder LR={rates[0]:.3g} | head LR={rates[1]:.3g} | '
                  f'train={train_time:.1f}s | val={valid_time:.1f}s | patience={stale}/{config.patience}', flush=True)
            if device.type == 'mps':
                release_device_memory(torch, device)
            if stale >= config.patience:
                print(f'Early stopping at epoch {epoch}', flush=True)
                break
        torch.save(best_state, output_dir / 'best_trainable.pt')
        result = {'config': asdict(config), 'path': str(output_dir.resolve()),
                  'validation_f1': best_f1, 'validation': best_metrics, 'best_epoch': best_epoch,
                  'epochs_run': len(history), 'training_seconds': training_seconds,
                  'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  'validation_seconds': validation_seconds,
                  'training_validation_wall_seconds': time.perf_counter() - started}
        save_json(output_dir / 'summary.json', result)
        return result
    except BaseException as error:
        save_json(output_dir / 'failure.json', {'error': repr(error), 'elapsed_seconds': time.perf_counter() - started})
        raise
    finally:
        del model, optimizer, scheduler, groups, trainable, loader, weights
        release_device_memory(torch, device)


# Evaluate each locked seed checkpoint on the test split
def evaluate_final_trial(config, trial, tokenizer, test_data):
    import torch
    from transformers import AutoModelForTokenClassification, BertConfig

    device = torch.device('cuda') if torch.cuda.is_available() else select_device(torch)
    amp_dtype = configure_runtime(config, device)
    _, evaluation_batch_size = execution_batch_sizes(config, device)
    model_config = BertConfig.from_pretrained(
        config.model, num_labels=len(LABELS), classifier_dropout=config.dropout,
        id2label=dict(enumerate(LABELS)), label2id={label: index for index, label in enumerate(LABELS)})
    model = AutoModelForTokenClassification.from_pretrained(
        config.model, config=model_config, **({'attn_implementation': 'sdpa'} if device.type == 'cuda' and config.cuda_fast_mode else {}))
    configure_e2(model, config.encoder_learning_rate, config.head_learning_rate)
    restore_trainable_checkpoint(model, Path(trial['path']) / 'best_trainable.pt')
    model.to(device)
    synchronize_device(torch, device)
    started = time.perf_counter()
    try:
        metrics, predictions = evaluate_e2(model, *test_data, evaluation_batch_size,
            make_e2_collator(tokenizer, device, align=device.type == 'cuda' and config.cuda_fast_mode), amp_dtype)
        synchronize_device(torch, device)
        seconds = time.perf_counter() - started
        save_json(Path(trial['path']) / 'test_metrics.json', metrics)
        save_json(Path(trial['path']) / 'test_predictions.json', predictions)
        print(f'Test seed={config.seed} | micro F1={metrics["typed_exact"]["f1"]:.2%} | time={seconds:.1f}s', flush=True)
        return {'seed': config.seed, 'metrics': metrics, 'test_seconds': seconds}
    finally:
        del model
        release_device_memory(torch, device)

# Save comparison tables and learning curves for each search stage
def render_results(output_dir, stages, trials, tests, summary):
    import matplotlib.pyplot as plt
    import pandas as pd

    smoke = trials[0]['config']['limit_docs'] is not None

    def table(frame, folder, name, title):
        frame.to_csv(folder / f'{name}.csv', index=False)
        fig, ax = plt.subplots(figsize=(12, max(2.4, len(frame) * 0.4 + 1.2)))
        ax.axis('off')
        ax.set_title(('SMOKE CHECK | ' if smoke else '') + title, pad=16)
        cells = ax.table(cellText=frame.values, colLabels=frame.columns, loc='center', cellLoc='center')
        cells.auto_set_font_size(False)
        cells.set_fontsize(9)
        cells.scale(1, 1.5)
        fig.tight_layout()
        fig.savefig(folder / f'{name}.png', dpi=160, bbox_inches='tight')
        plt.close(fig)

    for stage, candidates in stages.items():
        folder = output_dir / stage
        rows = [{'Encoder LR': f"{t['config']['encoder_learning_rate']:g}",
                 'Head LR': f"{t['config']['head_learning_rate']:g}",
                 'Best epoch': t['best_epoch'], 'Val F1 %': round(t['validation_f1'] * 100, 2),
                 'Train s': round(t['training_seconds'], 1), 'Val s': round(t['validation_seconds'], 1)} for t in candidates]
        table(pd.DataFrame(rows), folder, 'comparison', stage)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for index, trial in enumerate(candidates, 1):
            history = json.loads((Path(trial['path']) / 'history.json').read_text())
            epochs = [row['epoch'] for row in history]
            axes[0].plot(epochs, [row['validation_f1'] * 100 for row in history], label=f'Trial {index}')
            axes[1].plot(epochs, [row['training_loss'] for row in history], label=f'Trial {index}')
        axes[0].set(xlabel='Epoch', ylabel='Validation micro F1 (%)')
        axes[1].set(xlabel='Epoch', ylabel='Training loss')
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(folder / 'learning_curves.png', dpi=160)
        plt.close(fig)
    folder = output_dir / '5.2.3 Seeds and final results'
    test_by_seed = {test['seed']: test['metrics'] for test in tests}
    rows, per_type = [], []
    for trial in trials:
        seed = trial['config']['seed']
        splits = [('Validation', trial['validation'])]
        if seed in test_by_seed:
            splits.append(('Test', test_by_seed[seed]))
        for split, metrics in splits:
            rows.append({'Split': split, 'Seed': seed,
                         **{name: round(metrics['typed_exact'][key] * 100, 2) for name, key in
                            [('Precision %', 'precision'), ('Recall %', 'recall'), ('Micro F1 %', 'f1')]},
                         'Macro F1 %': round(metrics['typed_macro']['f1'] * 100, 2),
                         'Train s': round(trial['training_seconds'], 1)})
            for kind, values in metrics['per_type'].items():
                per_type.append({'Split': split, 'Seed': seed, 'Type': kind, **values})
    for split in ('Validation', 'Test'):
        stats = summary[split.lower()]
        if stats:
            rows.append({'Split': split, 'Seed': 'Mean ± SD',
                         **{name: f"{stats[key]['mean'] * 100:.2f} ± {stats[key]['std'] * 100:.2f}" for name, key in
                            [('Precision %', 'precision'), ('Recall %', 'recall'), ('Micro F1 %', 'f1'), ('Macro F1 %', 'macro_f1')]},
                         'Train s': f"{summary['training_seconds']['mean']:.1f} ± {summary['training_seconds']['std']:.1f}"})
    table(pd.DataFrame(rows), folder, 'final_results', 'E2 final results')
    pd.DataFrame(per_type).to_csv(folder / 'per_type_results.csv', index=False)
    settings = {key: trials[0]['config'][key] for key in
                ('encoder_learning_rate', 'head_learning_rate', 'batch_size', 'dropout', 'weight_decay', 'loss', 'scheduler', 'epochs', 'patience')}
    table(pd.DataFrame(settings.items(), columns=['Setting', 'Selected value']), folder,
          'selected_configuration', 'E2 selected configuration')


# Search on validation before locking settings for independent seeds
def run_experiment(config=None, trial_runner=None, prepare_data=None, test_runner=None, render=True):
    configure_output()
    config = inherit_e1_config(config or E2Config())
    if min(config.epochs, config.batch_size, config.micro_batch_size, config.cuda_evaluation_batch_size, config.patience) < 1:
        raise ValueError('Epochs batch sizes and patience must be positive')
    if config.limit_docs is not None and (config.limit_docs < 1 or config.evaluate_test):
        raise ValueError('Smoke runs require a positive document limit and no test evaluation')
    if config.seed != 9444:
        raise ValueError('Learning rate searches must use seed 9444')
    if config.loss not in {'cross_entropy', 'weighted_cross_entropy'}:
        raise ValueError('Unsupported inherited loss')
    if config.scheduler not in {'constant', 'linear', 'linear_warmup'}:
        raise ValueError('Unsupported inherited scheduler')
    output_dir = config.root / config.output_dir
    keep = config.limit_docs is None and config.evaluate_test
    with result_directory(output_dir, keep=keep):
        result = _run_experiment(config, output_dir, trial_runner or run_trial, prepare_data,
                                 test_runner or evaluate_final_trial, render)
    print(f'E2 complete | Results: {output_dir.resolve()}' if keep else 'E2 smoke check complete | Temporary results discarded', flush=True)
    return result


# Keep test loading after all searches and seed checkpoint selection
def _run_experiment(config, output_dir, trial_runner, prepare_data, test_runner, render):
    print(f'E2 output: {output_dir.resolve()}\nE1 settings: {config.e1_config_path}', flush=True)
    print(f'Fixed batch={config.batch_size} | dropout={config.dropout:g} | WD={config.weight_decay:g} | '
          f'loss={config.loss} | scheduler={config.scheduler} | patience={config.patience}', flush=True)
    save_json(output_dir / 'protocol.json', {
        'baseline': asdict(config), 'e1_config_sha256': hashlib.sha256(config.e1_config_path.read_bytes()).hexdigest(),
        'encoder_learning_rates': [1e-5, 2e-5, 3e-5], 'head_learning_rates': [5e-5, 1e-4, 2e-4],
        'search_seed': 9444, 'final_seeds': [42, 9444, 2026], 'smoke_only': config.limit_docs is not None,
        'selection': 'maximum validation typed exact span micro F1 with first candidate for ties',
        'test_policy': 'evaluate all three seed checkpoints only after validation selection',
        'optimizer_steps': 'one update per logical batch with token normalized micro batch gradients'})
    if prepare_data is None:
        from transformers import AutoTokenizer
        print('Loading and validating training and validation data', flush=True)
        tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased', use_fast=True)
        train_records, train_windows = load_e1_split(config, tokenizer, 'training')
        valid_records, valid_windows = load_e1_split(config, tokenizer, 'validation')
        data = tokenizer, train_records, train_windows, valid_records, valid_windows
    else:
        data = prepare_data(config)
    stages = {}

    def search(name, candidates):
        print(f'\nStage: {name} | {len(candidates)} trials', flush=True)
        trials = [trial_runner(candidate, output_dir / name / f'trial_{index:02d}', data)
                  for index, candidate in enumerate(candidates, 1)]
        stages[name] = trials
        selected = best_trial(trials)
        save_json(output_dir / name / 'selection.json', {'selected': selected, 'candidates': trials})
        print(f'Stage complete | best Val F1={selected["validation_f1"]:.2%}', flush=True)
        return replace(config, encoder_learning_rate=selected['config']['encoder_learning_rate'],
                       head_learning_rate=selected['config']['head_learning_rate'])

    selected = search('5.2.1 Encoder learning rate', [replace(config, encoder_learning_rate=rate) for rate in [1e-5, 2e-5, 3e-5]])
    selected = search('5.2.2 Head learning rate', [replace(selected, head_learning_rate=rate) for rate in [5e-5, 1e-4, 2e-4]])
    save_json(output_dir / 'selected_config.json', asdict(selected))
    folder = output_dir / '5.2.3 Seeds and final results'
    print('\nStage: 5.2.3 Seeds and final results | seeds=42, 9444, 2026', flush=True)
    trials = [trial_runner(replace(selected, seed=seed), folder / f'seed_{seed}', data) for seed in [42, 9444, 2026]]
    save_json(folder / 'validation_selection.json', trials)
    tests = []
    if config.evaluate_test:
        test_data = load_e1_split(config, data[0], 'testing')
        for trial in trials:
            tests.append(test_runner(replace(selected, seed=trial['config']['seed']), trial, data[0], test_data))
    summary = summarize_seeds(trials, tests)
    save_json(folder / 'summary.json', summary)
    if render:
        print('Generating comparison tables and figures', flush=True)
        render_results(output_dir, stages, trials, tests, summary)
    return {'selected_config': asdict(selected), 'stages': stages, 'trials': trials, 'tests': tests, 'summary': summary}


# Execute directly without command line arguments
if __name__ == '__main__':
    run_experiment(E2Config())
