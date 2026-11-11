"""E2: train BioBERT layers 8-11 and the typed-BIO head; freeze embeddings/layers 0-7.

Run with the cs9517 Python environment. With no arguments, compare encoder
learning rates 1e-5, 2e-5 and 3e-5 on Validation, then evaluate the selected
checkpoint on Test. Each run gets a fresh directory. --encoder-learning-rate
runs a single configuration instead (Test remains opt-in for that mode). Common data/window/decoding/evaluation helpers
are imported from e1_head_only.py; E1 itself is not executed or changed.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

from e1_head_only import (
    LABELS, decode_document, evaluate_standardized_records,
    index_records, load_jsonl_records, make_windows, select_device,
)


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


def set_e2_training_mode(model):
    """Enable dropout in the trainable layers/head, keep the frozen prefix in eval."""
    model.train()
    model.base_model.embeddings.eval()
    for layer in model.base_model.encoder.layer[:8]:
        layer.eval()


def run_trial(args):
    """Train one independent configuration, or evaluate one selected checkpoint."""
    import random
    import subprocess
    import time
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer, BertConfig

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('output-dir must be empty; choose a new directory for each run')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(torch)
    print(f'Device: {device}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased', use_fast=True)
    # The original BioBERT config predates the model_type field required by Transformers 5.
    model_config = BertConfig.from_pretrained(
        args.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)})
    model = AutoModelForTokenClassification.from_pretrained(args.evaluate_checkpoint or args.model, **({} if args.evaluate_checkpoint else {'config': model_config}))
    if args.evaluate_checkpoint and model.config.id2label != dict(enumerate(LABELS)):
        raise ValueError('Checkpoint label mapping differs from E2')
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError('max-length exceeds model position embeddings')
    model.to(device)
    parameter_groups = configure_e2(model, args.encoder_learning_rate, args.head_learning_rate)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}; "
          f"encoder LR={args.encoder_learning_rate:g}, head LR={args.head_learning_rate:g}", flush=True)

    def load_split(name):
        records = load_jsonl_records(args.data_dir / f'NCBI_{name}.jsonl')
        if args.limit_docs:
            records = records[:args.limit_docs]
        for record in records:
            if record.get('model') != args.model:
                raise ValueError('Tokenized data model differs from --model')
            # Validate that cached token IDs still match the exact full text.
            encoded = tokenizer(record['text'], add_special_tokens=False, return_offsets_mapping=True)
            if encoded['input_ids'] != record['input_ids'] or [list(x) for x in encoded['offset_mapping']] != record['offset_mapping']:
                raise ValueError(f"Tokenizer mismatch: {record['record_id']}")
        if name != 'training':
            # Validation/Test caches deliberately hide gold; use the evaluation source.
            gold = index_records(load_jsonl_records(args.gold_dir / f'NCBI_{name}.jsonl'), name)
            for record in records:
                reference = gold[record['record_id']]
                if reference['pmid'] != record['pmid'] or reference['text'] != record['text']:
                    raise ValueError('Gold and tokenized documents differ')
                record['entities'] = reference['entities']
                record['bio_tags'] = ['O'] * len(record['input_ids'])  # Unused during inference.
        if not records:
            raise ValueError(f'Empty {name} split')
        return records, make_windows(records, tokenizer, args.max_length, args.stride)

    def collate(windows):
        length = max(len(w['input_ids']) for w in windows)
        inputs = torch.tensor([w['input_ids'] + [tokenizer.pad_token_id] * (length - len(w['input_ids'])) for w in windows], device=device)
        mask = torch.tensor([[1] * len(w['input_ids']) + [0] * (length - len(w['input_ids'])) for w in windows], device=device)
        labels = torch.tensor([w['labels'] + [-100] * (length - len(w['labels'])) for w in windows], device=device)
        return windows, {'input_ids': inputs, 'attention_mask': mask, 'labels': labels}

    def evaluate(records, windows):
        model.eval()
        sums = [torch.zeros((len(r['input_ids']), len(LABELS))) for r in records]
        counts = [torch.zeros(len(r['input_ids'])) for r in records]
        votes = [{} for _ in records]
        conflicts = 0
        with torch.no_grad():
            for batch, inputs in DataLoader(windows, batch_size=args.batch_size, collate_fn=collate):
                inputs.pop('labels')
                logits = model(**inputs).logits.cpu()
                for row, window in enumerate(batch):
                    doc, start, end = window['doc'], window['start'], window['end']
                    values = logits[row, 1:1 + end - start]
                    for pos, label in enumerate(values.argmax(-1).tolist(), start):
                        votes[doc].setdefault(pos, set()).add(label)
                    sums[doc][start:end] += values
                    counts[doc][start:end] += 1
        predictions, illegal = [], 0
        for record, total, count, doc_votes in zip(records, sums, counts, votes):
            assert (count > 0).all(), 'Uncovered tokens'
            prediction, bad = decode_document(record, (total / count[:, None]).argmax(-1))
            predictions.append(prediction)
            illegal += bad
            conflicts += sum(len(v) > 1 for v in doc_votes.values())
        metrics = evaluate_standardized_records(records, predictions)
        metrics.update(illegal_bio_transitions=illegal, window_overlap_token_conflicts=conflicts)
        return metrics, predictions

    def save_json(name, value):
        (args.output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    if args.evaluate_checkpoint:
        records, windows = load_split('testing')
        metrics, predictions = evaluate(records, windows)
        save_json('test_metrics.json', metrics)
        save_json('test_predictions.json', predictions)
        save_json('evaluation_config.json', {
            'checkpoint': str(args.evaluate_checkpoint.resolve()),
            'device': str(device), 'max_length': args.max_length,
            'stride': args.stride, 'limit_docs': args.limit_docs,
        })
        print(f"Test typed exact F1: {metrics['typed_exact']['f1']:.4f}", flush=True)
        return

    train_records, train_windows = load_split('training')
    valid_records, valid_windows = load_split('validation')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(experiment='E2', device=str(device), labels=LABELS,
                  trainable_parameters=sum(p.numel() for p in trainable),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  torch_version=torch.__version__, frozen_scope='embeddings and Transformer layers 0-7',
                  trainable_layers=[8, 9, 10, 11],
                  encoder_trainable_parameters=sum(p.numel() for p in parameter_groups[0]['params']),
                  head_trainable_parameters=sum(p.numel() for p in parameter_groups[1]['params']),
                  transformers_version=importlib.metadata.version('transformers'),
                  selection='maximum validation typed exact-span micro F1')
    config['git_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    config['git_dirty'] = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip())
    config['source_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), Path(__file__).with_name('e1_head_only.py'))}
    save_json('config.json', config)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best_f1, stale, history, best_epoch = -1, 0, [], None
    try:
        for epoch in range(1, args.epochs + 1):
            set_e2_training_mode(model)
            loss_sum, token_count = 0.0, 0
            for _, inputs in DataLoader(train_windows, batch_size=args.batch_size, shuffle=True, collate_fn=collate):
                optimizer.zero_grad(set_to_none=True)
                loss = model(**inputs).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                count = int((inputs['labels'] != -100).sum().item())
                loss_sum += loss.item() * count
                token_count += count
            metrics, predictions = evaluate(valid_records, valid_windows)
            f1 = metrics['typed_exact']['f1']
            history.append({'epoch': epoch, 'training_loss': loss_sum / token_count, 'validation': metrics})
            save_json('history.json', history)
            print(f'Epoch {epoch}: loss={loss_sum / token_count:.4f}, validation F1={f1:.4f}', flush=True)
            if f1 > best_f1:
                best_f1, stale, best_epoch = f1, 0, epoch
                model.save_pretrained(args.output_dir / 'best_checkpoint', safe_serialization=True)
                tokenizer.save_pretrained(args.output_dir / 'best_checkpoint')
                save_json('validation_metrics.json', metrics)
                save_json('validation_predictions.json', predictions)
            else:
                stale += 1
            if stale >= args.patience:
                break
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        summary = {'best_epoch': best_epoch, 'best_validation_f1': best_f1,
                   'training_seconds': time.perf_counter() - started,
                   'trainable_parameters': config['trainable_parameters'],
                   'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                   'memory_note': 'CUDA peak allocation; MPS/CPU peak GPU memory unavailable'}
        if args.test:
            model = AutoModelForTokenClassification.from_pretrained(args.output_dir / 'best_checkpoint').to(device)
            records, windows = load_split('testing')
            metrics, predictions = evaluate(records, windows)
            save_json('test_metrics.json', metrics)
            save_json('test_predictions.json', predictions)
        save_json('summary.json', summary)
    except BaseException as error:
        save_json('failure.json', {'error': repr(error), 'elapsed_seconds': time.perf_counter() - started})
        raise



def release_device_cache():
    import gc
    import torch
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run_sweep(args):
    """Select only on Validation, freeze that choice, then evaluate Test once."""
    import csv
    from types import SimpleNamespace

    root = args.output_dir
    if root.exists() and any(root.iterdir()):
        raise ValueError('Search output directory must be empty; existing runs are preserved')
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'sweep_config.json', {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    })
    candidates, curves = [], []
    try:
        for learning_rate in args.encoder_learning_rates:
            label = format(learning_rate, '.8g')
            trial = SimpleNamespace(**vars(args))
            trial.output_dir = root / label
            trial.encoder_learning_rate = learning_rate
            trial.evaluate_checkpoint = None
            trial.test = False
            print(f'\n=== E2 encoder LR {learning_rate:g}; head LR {args.head_learning_rate:g} ===', flush=True)
            run_trial(trial)
            summary = json.loads((trial.output_dir / 'summary.json').read_text())
            history = json.loads((trial.output_dir / 'history.json').read_text())
            candidates.append({'encoder_learning_rate': learning_rate,
                               'head_learning_rate': args.head_learning_rate,
                               'run_dir': str(trial.output_dir.resolve()),
                               'completed_epochs': len(history), **summary})
            for epoch in history:
                metrics = epoch['validation']
                curves.append({'encoder_learning_rate': learning_rate, 'epoch': epoch['epoch'],
                               'training_loss': epoch['training_loss'],
                               'validation_precision': metrics['typed_exact']['precision'],
                               'validation_recall': metrics['typed_exact']['recall'],
                               'validation_micro_f1': metrics['typed_exact']['f1'],
                               'validation_macro_f1': metrics['typed_macro']['f1']})
            release_device_cache()
        # Ties keep the first candidate in the configured order.
        selected = max(candidates, key=lambda run: run['best_validation_f1'])
        checkpoint = Path(selected['run_dir']) / 'best_checkpoint'
        selection = {'criterion': 'maximum validation typed exact-span micro F1; Test excluded',
                     'tie_break': 'first candidate in configured order',
                     'seed': args.seed, 'limit_docs': args.limit_docs,
                     'selected': selected, 'candidates': candidates,
                     'checkpoint': str(checkpoint),
                     'checkpoint_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in checkpoint.glob('*.safetensors')}}
        write_json(root / 'selection.json', selection)  # Persist before any Test access.
        validation = json.loads((Path(selected['run_dir']) / 'validation_metrics.json').read_text())
        results = {'experiment': 'E2', 'selection': selection, 'validation': validation,
                   'test': None, 'smoke_test': args.limit_docs is not None}
        with (root / 'learning_curves.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curves[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(curves)
        write_json(root / 'results.json', results)
        print(f"\nSelected encoder LR={selected['encoder_learning_rate']:g}, "
              f"epoch={selected['best_epoch']}, validation F1={selected['best_validation_f1']:.4f}", flush=True)
        if not args.validation_only:
            evaluation = SimpleNamespace(**vars(args))
            evaluation.encoder_learning_rate = selected['encoder_learning_rate']
            evaluation.evaluate_checkpoint = checkpoint
            evaluation.output_dir = root / 'selected_test'
            evaluation.test = True
            run_trial(evaluation)
            results['test'] = json.loads((evaluation.output_dir / 'test_metrics.json').read_text())
            write_json(root / 'results.json', results)
            release_device_cache()
        lines = ['# E2 partial fine-tuning', '',
                 'SMOKE TEST ONLY: not formal results.' if args.limit_docs is not None else f'Single-seed formal run (seed={args.seed}).', '',
                 '| Encoder LR | Epochs | Best epoch | Validation Micro F1 | Training + validation seconds |',
                 '|---|---:|---:|---:|---:|']
        lines += [f"| {r['encoder_learning_rate']:g} | {r['completed_epochs']} | {r['best_epoch']} | {r['best_validation_f1']:.2%} | {r['training_seconds']:.2f} |" for r in candidates]
        lines += ['', f"Selected encoder LR: {selected['encoder_learning_rate']:g}; head LR: {args.head_learning_rate:g}; best epoch: {selected['best_epoch']}.", '',
                  '| Split | Precision | Recall | Micro F1 | Macro F1 |', '|---|---:|---:|---:|---:|']
        for name, metrics in [('Validation', validation), ('Test', results['test'])]:
            if metrics is not None:
                exact = metrics['typed_exact']
                lines.append(f"| {name} | {exact['precision']:.2%} | {exact['recall']:.2%} | {exact['f1']:.2%} | {metrics['typed_macro']['f1']:.2%} |")
        lines += ['', 'Selection uses Validation only. Single-seed results do not estimate uncertainty.',
                  'MPS peak GPU memory is unavailable (null). Full per-type metrics and confusion matrices are in results.json.',
                  'Training time includes per-epoch Validation and checkpoint saves; excludes other LR candidates and final Test.']
        (root / 'summary.md').write_text('\n'.join(lines) + '\n')
        print('SMOKE TEST ONLY' if args.limit_docs is not None else 'E2 complete', flush=True)
        print(f'Results: {root.resolve()}', flush=True)
        return results
    except BaseException as error:
        write_json(root / 'sweep_failure.json', {'error': repr(error), 'completed_candidates': candidates})
        raise


def main():
    import argparse
    import math
    from datetime import datetime, timezone
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'Dataset/TokenizedData/BioBERT')
    parser.add_argument('--gold-dir', type=Path, default=Path(__file__).resolve().parent / 'Dataset/ProcessedData')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--model', default='dmis-lab/biobert-base-cased-v1.1')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--evaluate-checkpoint', type=Path, help='Evaluate a frozen selected checkpoint without training; requires --test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--encoder-learning-rate', type=float, default=None, help='Run only this encoder LR instead of the default three-LR search')
    parser.add_argument('--head-learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-docs', type=int, help='Smoke test only; not a formal experiment')
    parser.add_argument('--test', action='store_true', help='Evaluate Test after selecting the checkpoint on Validation')
    parser.add_argument('--encoder-learning-rates', type=float, nargs='+', default=[1e-5, 2e-5, 3e-5], help='Encoder LR candidates for the default search')
    parser.add_argument('--validation-only', action='store_true', help='Search without final Test evaluation')
    args = parser.parse_args()
    if args.evaluate_checkpoint and not args.test:
        parser.error('--evaluate-checkpoint requires --test')
    if args.validation_only and (args.test or args.evaluate_checkpoint):
        parser.error('--validation-only cannot be combined with Test evaluation')
    if min(args.epochs, args.batch_size, args.patience) < 1 or (args.limit_docs is not None and args.limit_docs < 1):
        parser.error('epochs, batch-size, patience and limit-docs must be positive')
    rates = [args.head_learning_rate, *args.encoder_learning_rates]
    if args.encoder_learning_rate is not None:
        rates.append(args.encoder_learning_rate)
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        parser.error('learning rates must be finite and positive')
    if len(set(args.encoder_learning_rates)) != len(args.encoder_learning_rates):
        parser.error('encoder-learning-rates must not contain duplicates')
    if not 0 <= args.stride < args.max_length - 2 or args.max_length > 512:
        parser.error('Use max-length <= 512 and 0 <= stride < max-length - 2')
    if args.output_dir is None:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        args.output_dir = Path(__file__).resolve().parent / 'results/E2_partial_sweep' / run_id
    print(f'Output directory: {args.output_dir.resolve()}', flush=True)
    if args.encoder_learning_rate is not None or args.evaluate_checkpoint:
        if args.encoder_learning_rate is None:
            args.encoder_learning_rate = 2e-5  # No optimization occurs in evaluation-only mode.
        run_trial(args)
    else:
        run_sweep(args)


if __name__ == '__main__':
    main()
