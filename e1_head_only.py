"""E1: frozen BioBERT with a trained typed-BIO classification head."""
from collections import Counter
import json
from pathlib import Path


EVALUATION_ENTITY_TYPES = (
    "SpecificDisease",
    "DiseaseClass",
    "CompositeMention",
    "Modifier",
)


# Read nonempty JSONL records
def load_jsonl_records(file_path):
    records = []
    with Path(file_path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {file_path}"
                ) from error
    return records


# Require unique document identifiers
def index_records(records, source_name):
    indexed = {}
    for record in records:
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"Missing record_id in {source_name}")
        if record_id in indexed:
            raise ValueError(
                f"Duplicate record_id {record_id} in {source_name}"
            )
        indexed[record_id] = record
    return indexed


# Check entity types and character offsets
def validate_entity(entity, record_id, text_length):
    if not isinstance(entity, dict):
        raise TypeError(f"Entity in {record_id} must be a dictionary")
    if set(entity) != {"start", "end", "type"}:
        raise ValueError(
            f"Entity in {record_id} must contain only start, end, and type"
        )

    start, end = entity["start"], entity["end"]
    entity_type = entity["type"]
    valid_offsets = (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
    )
    if not valid_offsets or not 0 <= start < end <= text_length:
        raise ValueError(
            f"Invalid span [{start}, {end}) in {record_id}"
        )
    if entity_type not in EVALUATION_ENTITY_TYPES:
        raise ValueError(
            f"Invalid entity type {entity_type!r} in {record_id}"
        )
    return start, end, entity_type


# Validate predictions against gold document metadata
def validate_prediction(prediction, gold_record):
    required_fields = {"record_id", "pmid", "entities"}
    if set(prediction) != required_fields:
        raise ValueError(
            f"Prediction {prediction.get('record_id')} must contain only "
            "record_id, pmid, and entities"
        )
    record_id = prediction["record_id"]
    if prediction["pmid"] != gold_record["pmid"]:
        raise ValueError(f"PMID mismatch for {record_id}")
    if not isinstance(prediction["entities"], list):
        raise ValueError(f"entities must be a list in {record_id}")

    text_length = len(gold_record["text"])
    entities = [
        validate_entity(entity, record_id, text_length)
        for entity in prediction["entities"]
    ]
    if entities != sorted(entities):
        raise ValueError(f"Entities are not sorted in {record_id}")
    if len(entities) != len(set(entities)):
        raise ValueError(f"Duplicate entities in {record_id}")

    spans = [(start, end) for start, end, _ in entities]
    if len(spans) != len(set(spans)):
        raise ValueError(
            f"A span has multiple predicted types in {record_id}"
        )
    return entities


# Compute precision recall and F1 with zero division handling
def calculate_prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# Count exact matches and unmatched values
def set_counts(gold_values, predicted_values):
    return (
        len(gold_values & predicted_values),
        len(predicted_values - gold_values),
        len(gold_values - predicted_values),
    )


# Preserve repeated boundary counts
def boundary_counts(gold_values, predicted_values):
    gold_counter = Counter(gold_values)
    predicted_counter = Counter(predicted_values)
    return (
        sum((gold_counter & predicted_counter).values()),
        sum((predicted_counter - gold_counter).values()),
        sum((gold_counter - predicted_counter).values()),
    )


# Expand spans into character positions
def spans_to_characters(spans):
    characters = set()
    for start, end in spans:
        characters.update(range(start, end))
    return characters


# Accumulate evaluation counts
def add_counts(total, current):
    for index, value in enumerate(current):
        total[index] += value


# Evaluate typed spans and boundary diagnostics
def evaluate_standardized_records(gold_records, prediction_records):
    gold_by_id = index_records(gold_records, "gold records")
    pred_by_id = index_records(prediction_records, "predictions")
    missing = sorted(set(gold_by_id) - set(pred_by_id))
    extra = sorted(set(pred_by_id) - set(gold_by_id))
    if missing or extra:
        raise ValueError(
            f"Record mismatch: missing={missing}, extra={extra}"
        )

    typed_total = [0, 0, 0]
    span_only_total = [0, 0, 0]
    start_total = [0, 0, 0]
    end_total = [0, 0, 0]
    character_total = [0, 0, 0]
    per_type_totals = {
        entity_type: [0, 0, 0]
        for entity_type in EVALUATION_ENTITY_TYPES
    }
    type_confusion = {
        gold_type: {pred_type: 0 for pred_type in EVALUATION_ENTITY_TYPES}
        for gold_type in EVALUATION_ENTITY_TYPES
    }
    matched_span_count = 0
    correct_type_count = 0
    gold_entity_count = 0
    predicted_entity_count = 0

    for record_id, gold_record in gold_by_id.items():
        text_length = len(gold_record["text"])
        gold_entities = [
            validate_entity(entity, record_id, text_length)
            for entity in gold_record["entities"]
        ]
        predicted_entities = validate_prediction(
            pred_by_id[record_id], gold_record
        )

        gold_spans = [(start, end) for start, end, _ in gold_entities]
        predicted_spans = [
            (start, end) for start, end, _ in predicted_entities
        ]
        if len(gold_spans) != len(set(gold_spans)):
            raise ValueError(
                f"A gold span has multiple types in {record_id}"
            )

        gold_entity_count += len(gold_entities)
        predicted_entity_count += len(predicted_entities)
        gold_entity_set = set(gold_entities)
        predicted_entity_set = set(predicted_entities)
        gold_span_set = set(gold_spans)
        predicted_span_set = set(predicted_spans)

        add_counts(
            typed_total,
            set_counts(gold_entity_set, predicted_entity_set),
        )
        add_counts(
            span_only_total,
            set_counts(gold_span_set, predicted_span_set),
        )

        for entity_type in EVALUATION_ENTITY_TYPES:
            gold_for_type = {
                entity for entity in gold_entity_set
                if entity[2] == entity_type
            }
            predicted_for_type = {
                entity for entity in predicted_entity_set
                if entity[2] == entity_type
            }
            add_counts(
                per_type_totals[entity_type],
                set_counts(gold_for_type, predicted_for_type),
            )

        gold_type_by_span = {
            (start, end): entity_type
            for start, end, entity_type in gold_entities
        }
        predicted_type_by_span = {
            (start, end): entity_type
            for start, end, entity_type in predicted_entities
        }
        for span in gold_span_set & predicted_span_set:
            gold_type = gold_type_by_span[span]
            predicted_type = predicted_type_by_span[span]
            type_confusion[gold_type][predicted_type] += 1
            matched_span_count += 1
            correct_type_count += int(gold_type == predicted_type)

        add_counts(
            start_total,
            boundary_counts(
                [start for start, _ in gold_spans],
                [start for start, _ in predicted_spans],
            ),
        )
        add_counts(
            end_total,
            boundary_counts(
                [end for _, end in gold_spans],
                [end for _, end in predicted_spans],
            ),
        )
        add_counts(
            character_total,
            set_counts(
                spans_to_characters(gold_spans),
                spans_to_characters(predicted_spans),
            ),
        )

    per_type = {}
    for entity_type, counts in per_type_totals.items():
        values = calculate_prf(*counts)
        values["support"] = counts[0] + counts[2]
        values["predicted_count"] = counts[0] + counts[1]
        per_type[entity_type] = values

    typed_macro = {
        metric: sum(values[metric] for values in per_type.values())
        / len(EVALUATION_ENTITY_TYPES)
        for metric in ("precision", "recall", "f1")
    }
    wrong_type_count = matched_span_count - correct_type_count

    return {
        "averaging": "micro",
        "document_count": len(gold_by_id),
        "gold_entity_count": gold_entity_count,
        "predicted_entity_count": predicted_entity_count,
        "typed_exact": calculate_prf(*typed_total),
        "typed_macro": typed_macro,
        "per_type": per_type,
        "span_only_exact": calculate_prf(*span_only_total),
        "exact_span_type": {
            "matched_span_count": matched_span_count,
            "correct_type_count": correct_type_count,
            "wrong_type_count": wrong_type_count,
            "accuracy": (
                correct_type_count / matched_span_count
                if matched_span_count
                else 0.0
            ),
        },
        "type_confusion_matrix": type_confusion,
        "start_boundary": calculate_prf(*start_total),
        "end_boundary": calculate_prf(*end_total),
        "character_overlap": calculate_prf(*character_total),
    }



LABELS = ['O'] + [f'{prefix}-{kind}' for kind in EVALUATION_ENTITY_TYPES for prefix in ('B', 'I')]


# Prefer the local accelerator when available
def select_device(torch):
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Supervise each full document token once
def make_windows(records, tokenizer, max_length, stride):
    """Keep full-document token indices; supervise overlap tokens only once."""
    capacity = max_length - 2
    if not 0 <= stride < capacity:
        raise ValueError('stride must be between 0 and max_length - 3')
    windows = []
    for doc, record in enumerate(records):
        ids, tags = record['input_ids'], record['bio_tags']
        if not len(ids) == len(tags) == len(record['offset_mapping']):
            raise ValueError(f"Unaligned tokens: {record['record_id']}")
        covered = 0
        for start in range(0, len(ids), capacity - stride):
            end = min(start + capacity, len(ids))
            labels = [LABELS.index(tag) if i >= covered else -100
                      for i, tag in enumerate(tags[start:end], start)]
            windows.append({'doc': doc, 'start': start, 'end': end,
                            'input_ids': [tokenizer.cls_token_id] + ids[start:end] + [tokenizer.sep_token_id],
                            'labels': [-100] + labels + [-100]})
            covered = end
            if end == len(ids):
                break
    return windows


# Repair illegal BIO transitions during document decoding
def decode_document(record, tag_ids):
    """Repair illegal I transitions as B; report their original count."""
    entities, active, illegal = [], None, 0
    for (start, end), label in zip(record['offset_mapping'], tag_ids):
        tag = LABELS[int(label)]
        prefix, kind = ('O', None) if tag == 'O' else tag.split('-', 1)
        if prefix == 'I' and (active is None or active['type'] != kind):
            illegal += 1
            prefix = 'B'
        if prefix != 'I' and active is not None:
            entities.append(active)
            active = None
        if prefix == 'B':
            active = {'start': start, 'end': end, 'type': kind}
        elif prefix == 'I':
            active['end'] = end
    if active is not None:
        entities.append(active)
    return {'record_id': record['record_id'], 'pmid': record['pmid'], 'entities': entities}, illegal


# Configure direct execution without command line arguments
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import time


@dataclass(frozen=True)
class E1Config:
    root: Path = Path(__file__).resolve().parent
    # Keep each direct run in a separate Sydney timestamp folder
    output_dir: Path = field(default_factory=lambda: Path('results/5. fine-tune/5.1 E1') /
                             datetime.now(ZoneInfo('Australia/Sydney')).strftime('run_%Y%m%d_%H%M%S_%f'))
    model: str = 'dmis-lab/biobert-base-cased-v1.1'
    epochs: int = 30
    batch_size: int = 8
    weight_decay: float = 0.01
    scheduler: str = 'constant'
    patience: int = 3
    loss: str = 'cross_entropy'
    seed: int = 9444
    learning_rate: float = 5e-4
    dropout: float = 0.1
    max_length: int = 512
    stride: int = 64
    run_stage4: bool = True
    batch_improvement_threshold: float = 0.0
    limit_docs: int | None = None
    evaluate_test: bool = True


# Store machine readable results without console output
def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


# Synchronize asynchronous device work before timing
def synchronize_device(torch, device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


# Count each supervised training token once for inverse frequency weights
def training_class_weights(windows, torch):
    counts = torch.zeros(len(LABELS), dtype=torch.float64)
    for window in windows:
        labels = [label for label in window['labels'] if label != -100]
        counts += torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=len(LABELS))
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights.float()


# Define decay over the full epoch budget with early stopping applied separately
def learning_rate_factor(step, total_steps, scheduler):
    if scheduler == 'constant':
        return 1.0
    if scheduler not in {'linear', 'linear_warmup'}:
        raise ValueError(f'Unknown scheduler {scheduler}')
    warmup = math.ceil(total_steps * 0.1) if scheduler == 'linear_warmup' else 0
    if step < warmup:
        return step / max(1, warmup)
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup))


# Freeze the encoder and vary only the classification head dropout
def configure_e1(model, dropout):
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    model.dropout.p = dropout
    model.config.classifier_dropout = dropout
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if names != {'classifier.weight', 'classifier.bias'}:
        raise ValueError('E1 must train only the classification head')


# Validate cached full text tokenization without long sequence warnings
def load_e1_split(config, tokenizer, name):
    records = load_jsonl_records(config.root / f'Dataset/TokenizedData/BioBERT/NCBI_{name}.jsonl')
    if config.limit_docs is not None:
        records = records[:config.limit_docs]
    gold = None
    if name != 'training':
        gold = index_records(load_jsonl_records(config.root / f'Dataset/ProcessedData/NCBI_{name}.jsonl'), name)
    for record in records:
        if record.get('model') != config.model:
            raise ValueError('Tokenized data model differs from the configured model')
        encoded = tokenizer(record['text'], add_special_tokens=False,
                            return_offsets_mapping=True, verbose=False)
        if encoded['input_ids'] != record['input_ids'] or [list(x) for x in encoded['offset_mapping']] != record['offset_mapping']:
            raise ValueError(f"Tokenizer mismatch for {record['record_id']}")
        if gold is not None:
            reference = gold[record['record_id']]
            if reference['pmid'] != record['pmid'] or reference['text'] != record['text']:
                raise ValueError('Gold and tokenized documents differ')
            record['entities'] = reference['entities']
            record['bio_tags'] = ['O'] * len(record['input_ids'])
    if not records:
        raise ValueError(f'Empty {name} split')
    return records, make_windows(records, tokenizer, config.max_length, config.stride)


# Pad windows and preserve document metadata for merged predictions
def make_collator(tokenizer, device):
    import torch

    def collate(windows):
        length = max(len(window['input_ids']) for window in windows)
        inputs = torch.tensor([w['input_ids'] + [tokenizer.pad_token_id] * (length - len(w['input_ids'])) for w in windows], device=device)
        mask = torch.tensor([[1] * len(w['input_ids']) + [0] * (length - len(w['input_ids'])) for w in windows], device=device)
        labels = torch.tensor([w['labels'] + [-100] * (length - len(w['labels'])) for w in windows], device=device)
        return windows, {'input_ids': inputs, 'attention_mask': mask, 'labels': labels}

    return collate


# Average overlapping window logits before full document decoding
def evaluate_e1(model, records, windows, batch_size, collate):
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    sums = [torch.zeros((len(record['input_ids']), len(LABELS))) for record in records]
    counts = [torch.zeros(len(record['input_ids'])) for record in records]
    votes = [{} for _ in records]
    with torch.no_grad():
        for batch, inputs in DataLoader(windows, batch_size=batch_size, collate_fn=collate):
            inputs.pop('labels')
            logits = model(**inputs).logits.cpu()
            for row, window in enumerate(batch):
                doc, start, end = window['doc'], window['start'], window['end']
                values = logits[row, 1:1 + end - start]
                for pos, label in enumerate(values.argmax(-1).tolist(), start):
                    votes[doc].setdefault(pos, set()).add(label)
                sums[doc][start:end] += values
                counts[doc][start:end] += 1
    predictions, illegal, conflicts = [], 0, 0
    for record, total, count, doc_votes in zip(records, sums, counts, votes):
        if not (count > 0).all():
            raise ValueError('Uncovered document tokens')
        prediction, bad = decode_document(record, (total / count[:, None]).argmax(-1))
        predictions.append(prediction)
        illegal += bad
        conflicts += sum(len(values) > 1 for values in doc_votes.values())
    metrics = evaluate_standardized_records(records, predictions)
    metrics.update(illegal_bio_transitions=illegal, window_overlap_token_conflicts=conflicts)
    return metrics, predictions


# Train each trial from the same pretrained encoder and a seeded new head
def run_trial(config, output_dir, data):
    import random
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, BertConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f'\nTrial: {output_dir} | seed={config.seed} | LR={config.learning_rate:g} | '
          f'dropout={config.dropout:g} | loss={config.loss} | scheduler={config.scheduler} | '
          f'batch={config.batch_size} | WD={config.weight_decay:g}', flush=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    tokenizer, train_records, train_windows, valid_records, valid_windows = data
    device = select_device(torch)
    print(f'Device: {device} | Loading model', flush=True)
    # Supply the legacy BioBERT architecture explicitly
    model_config = BertConfig.from_pretrained(
        config.model, num_labels=len(LABELS), classifier_dropout=config.dropout,
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)})
    model = AutoModelForTokenClassification.from_pretrained(config.model, config=model_config)
    if not 2 < config.max_length <= model.config.max_position_embeddings:
        raise ValueError('Invalid model window length')
    configure_e1(model, config.dropout)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = math.ceil(len(train_windows) / config.batch_size) * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, total_steps, config.scheduler))
    weights = training_class_weights(train_windows, torch) if config.loss == 'weighted_cross_entropy' else None
    criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    collate = make_collator(tokenizer, device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(train_windows, batch_size=config.batch_size, shuffle=True,
                        generator=generator, collate_fn=collate)
    metadata = asdict(config)
    metadata.update(device=str(device), labels=LABELS, torch_version=torch.__version__,
                    trainable_parameters=sum(p.numel() for p in trainable),
                    class_weights=weights.tolist() if weights is not None else None,
                    selection='maximum validation typed exact span micro F1',
                    scheduler_total_steps=total_steps, train_documents=len(train_records),
                    validation_documents=len(valid_records))
    save_json(output_dir / 'config.json', metadata)
    synchronize_device(torch, device)
    started = time.perf_counter()
    history, best_f1, stale = [], -1.0, 0
    training_seconds, validation_seconds = 0.0, 0.0
    best_state = None
    try:
        for epoch in range(1, config.epochs + 1):
            synchronize_device(torch, device)
            epoch_started = time.perf_counter()
            model.train()
            model.base_model.eval()
            loss_sum, normalizer_sum = 0.0, 0.0
            initial_lr = optimizer.param_groups[0]['lr']
            for _, inputs in loader:
                optimizer.zero_grad(set_to_none=True)
                labels = inputs.pop('labels')
                logits = model(**inputs).logits
                loss = criterion(logits.reshape(-1, len(LABELS)), labels.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                # Match the weighted mean denominator used by CrossEntropy
                supervised = labels[labels != -100]
                denominator = float(weights[supervised.cpu()].sum()) if weights is not None else supervised.numel()
                loss_sum += loss.item() * denominator
                normalizer_sum += denominator
            synchronize_device(torch, device)
            epoch_train_seconds = time.perf_counter() - epoch_started
            validation_started = time.perf_counter()
            metrics, predictions = evaluate_e1(model, valid_records, valid_windows, config.batch_size, collate)
            synchronize_device(torch, device)
            epoch_valid_seconds = time.perf_counter() - validation_started
            training_seconds += epoch_train_seconds
            validation_seconds += epoch_valid_seconds
            f1 = metrics['typed_exact']['f1']
            history.append({'epoch': epoch, 'training_loss': loss_sum / normalizer_sum,
                            'validation_f1': f1, 'learning_rate': initial_lr,
                            'training_seconds': epoch_train_seconds, 'validation_seconds': epoch_valid_seconds})
            if f1 > best_f1:
                best_f1, best_epoch, stale = f1, epoch, 0
                best_metrics, best_predictions = metrics, predictions
                # Keep only the tiny trainable head while searching
                best_state = {name: value.detach().cpu().clone() for name, value in model.classifier.state_dict().items()}
            else:
                stale += 1
            save_json(output_dir / 'history.json', history)
            print(f'Epoch {epoch:02d}/{config.epochs} | loss={loss_sum / normalizer_sum:.4f} | '
                  f'Val F1={f1:.2%} | best={best_f1:.2%} at epoch {best_epoch} | '
                  f'LR={initial_lr:.3g} | train={epoch_train_seconds:.1f}s | '
                  f'val={epoch_valid_seconds:.1f}s | patience={stale}/{config.patience}', flush=True)
            if stale >= config.patience:
                print(f'Early stopping at epoch {epoch}', flush=True)
                break
        synchronize_device(torch, device)
        wall_seconds = time.perf_counter() - started
        torch.save(best_state, output_dir / 'best_head.pt')
        save_json(output_dir / 'validation_metrics.json', best_metrics)
        save_json(output_dir / 'validation_predictions.json', best_predictions)
        result = {'config': asdict(config), 'path': str(output_dir.resolve()),
                  'best_epoch': best_epoch, 'epochs_run': len(history),
                  'validation_f1': best_f1, 'validation': best_metrics,
                  'training_seconds': training_seconds, 'validation_seconds': validation_seconds,
                  'training_validation_wall_seconds': wall_seconds}
        save_json(output_dir / 'summary.json', result)
        print(f'Trial complete | best Val F1={best_f1:.2%} | '
              f'total train={training_seconds:.1f}s | total val={validation_seconds:.1f}s', flush=True)
        return result
    except BaseException as error:
        save_json(output_dir / 'failure.json', {'error': repr(error), 'elapsed_seconds': time.perf_counter() - started})
        raise
    finally:
        del model, optimizer
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()


# Select strictly by validation F1 and retain candidate order for ties
def best_trial(trials):
    return max(trials, key=lambda trial: trial['validation_f1'])


# Restore the best head without storing repeated frozen encoder copies
def evaluate_final_trial(config, trial, tokenizer, test_data):
    import torch
    from transformers import AutoModelForTokenClassification, BertConfig

    model_config = BertConfig.from_pretrained(
        config.model, num_labels=len(LABELS), classifier_dropout=config.dropout,
        id2label=dict(enumerate(LABELS)), label2id={label: i for i, label in enumerate(LABELS)})
    model = AutoModelForTokenClassification.from_pretrained(config.model, config=model_config)
    model.classifier.load_state_dict(torch.load(Path(trial['path']) / 'best_head.pt', map_location='cpu', weights_only=True))
    device = select_device(torch)
    model.to(device)
    synchronize_device(torch, device)
    started = time.perf_counter()
    metrics, predictions = evaluate_e1(model, *test_data, config.batch_size, make_collator(tokenizer, device))
    synchronize_device(torch, device)
    save_json(Path(trial['path']) / 'test_metrics.json', metrics)
    save_json(Path(trial['path']) / 'test_predictions.json', predictions)
    return {'seed': config.seed, 'metrics': metrics, 'test_seconds': time.perf_counter() - started}


# Summarize independent seeds with sample standard deviation
def summarize_seeds(trials, tests):
    import statistics

    def describe(values):
        return {'mean': statistics.mean(values), 'std': statistics.stdev(values)}

    summary = {'seeds': [trial['config']['seed'] for trial in trials],
               'standard_deviation': 'sample ddof 1', 'validation': {}, 'test': {}}
    for metric in ('precision', 'recall', 'f1'):
        summary['validation'][metric] = describe([trial['validation']['typed_exact'][metric] for trial in trials])
        if tests:
            summary['test'][metric] = describe([test['metrics']['typed_exact'][metric] for test in tests])
    summary['validation']['macro_f1'] = describe([trial['validation']['typed_macro']['f1'] for trial in trials])
    if tests:
        summary['test']['macro_f1'] = describe([test['metrics']['typed_macro']['f1'] for test in tests])
    summary['training_seconds'] = describe([trial['training_seconds'] for trial in trials])
    return summary


# Save figures and tables as the only displayed experiment results
def render_results(output_dir, stages, final_trials, tests, summary, show=False):
    import matplotlib.pyplot as plt
    import pandas as pd

    output_dir = Path(output_dir)

    def table_figure(frame, title, path):
        fig, ax = plt.subplots(figsize=(max(10, len(frame.columns) * 1.45), max(2.4, len(frame) * 0.42 + 1.2)))
        ax.axis('off')
        ax.set_title(('SMOKE CHECK | ' if final_trials[0]['config'].get('limit_docs') is not None else '') + title, pad=16)
        table = ax.table(cellText=frame.values, colLabels=frame.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    for name, trials in stages.items():
        rows = []
        for trial in trials:
            config = trial['config']
            rows.append({'LR': f"{config['learning_rate']:g}", 'Dropout': config['dropout'],
                         'Loss': config['loss'], 'Scheduler': config['scheduler'],
                         'Batch': config['batch_size'], 'WD': config['weight_decay'],
                         'Best epoch': trial['best_epoch'], 'Val F1 %': round(trial['validation_f1'] * 100, 2),
                         'Train s': round(trial['training_seconds'], 1),
                         'Val s': round(trial['validation_seconds'], 1)})
        folder = output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(folder / 'comparison.csv', index=False)
        table_figure(frame, name, folder / 'comparison.png')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for index, trial in enumerate(trials, 1):
            history = json.loads((Path(trial['path']) / 'history.json').read_text())
            epochs = [row['epoch'] for row in history]
            axes[0].plot(epochs, [row['validation_f1'] * 100 for row in history], label=f'Trial {index}')
            axes[1].plot(epochs, [row['training_loss'] for row in history], label=f'Trial {index}')
        axes[0].set(xlabel='Epoch', ylabel='Validation micro F1 (%)')
        axes[1].set(xlabel='Epoch', ylabel='Training loss')
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(folder / 'learning_curves.png', dpi=160)
        if show:
            plt.show()
        plt.close(fig)

    folder = output_dir / '5.1.5 Seeds and final results'
    final_rows = []
    test_by_seed = {test['seed']: test for test in tests}
    for trial in final_trials:
        seed = trial['config']['seed']
        for split, metrics in [('Validation', trial['validation'])] + ([('Test', test_by_seed[seed]['metrics'])] if seed in test_by_seed else []):
            final_rows.append({'Split': split, 'Seed': seed,
                               'Precision %': round(metrics['typed_exact']['precision'] * 100, 2),
                               'Recall %': round(metrics['typed_exact']['recall'] * 100, 2),
                               'Micro F1 %': round(metrics['typed_exact']['f1'] * 100, 2),
                               'Macro F1 %': round(metrics['typed_macro']['f1'] * 100, 2),
                               'Train s': round(trial['training_seconds'], 1)})
    for split in ('Validation', 'Test'):
        stats = summary[split.lower()]
        if stats:
            final_rows.append({'Split': split, 'Seed': 'Mean ± SD',
                               **{title: f"{stats[key]['mean'] * 100:.2f} ± {stats[key]['std'] * 100:.2f}" for title, key in [('Precision %', 'precision'), ('Recall %', 'recall'), ('Micro F1 %', 'f1'), ('Macro F1 %', 'macro_f1')]},
                               'Train s': f"{summary['training_seconds']['mean']:.1f} ± {summary['training_seconds']['std']:.1f}"})
    frame = pd.DataFrame(final_rows)
    frame.to_csv(folder / 'final_results.csv', index=False)
    table_figure(frame, 'E1 final results', folder / 'final_results.png')
    # Preserve all per type results for plotting and error analysis
    per_type = []
    for trial in final_trials:
        seed = trial['config']['seed']
        splits = [('Validation', trial['validation'])]
        if seed in test_by_seed:
            splits.append(('Test', test_by_seed[seed]['metrics']))
        for split, metrics in splits:
            for kind, values in metrics['per_type'].items():
                per_type.append({'Split': split, 'Seed': seed, 'Type': kind,
                                 'Precision %': round(values['precision'] * 100, 2),
                                 'Recall %': round(values['recall'] * 100, 2),
                                 'F1 %': round(values['f1'] * 100, 2), 'Support': values['support']})
    pd.DataFrame(per_type).to_csv(folder / 'per_type_results.csv', index=False)
    settings = {key: final_trials[0]['config'][key] for key in
                ('learning_rate', 'dropout', 'loss', 'scheduler', 'batch_size', 'weight_decay', 'epochs', 'patience')}
    table_figure(pd.DataFrame(settings.items(), columns=['Setting', 'Selected value']),
                 'E1 selected configuration', folder / 'selected_configuration.png')


# Hide routine model hub messages and download progress
def configure_output():
    from huggingface_hub.utils import disable_progress_bars
    from huggingface_hub.utils import logging as hub_logging
    from transformers.utils import logging as transformers_logging

    hub_logging.set_verbosity_error()
    disable_progress_bars()
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()


# Execute ordered searches before opening the test split
def run_experiment(config, trial_runner=None, prepare_data=None, test_runner=None, render=True):
    configure_output()
    if config.epochs < 1 or config.patience < 1 or config.batch_size < 1:
        raise ValueError('Epochs patience and batch size must be positive')
    if config.limit_docs is not None and config.limit_docs < 1:
        raise ValueError('Document limit must be positive')
    if config.limit_docs is not None and config.evaluate_test:
        raise ValueError('Smoke runs must disable final test evaluation')
    if (config.batch_size, config.weight_decay, config.scheduler, config.patience,
        config.loss, config.seed, config.dropout) != (8, 0.01, 'constant', 3, 'cross_entropy', 9444, 0.1):
        raise ValueError('Search must start from the fixed E1 baseline settings')
    output_dir = config.root / config.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Choose an empty output directory to preserve earlier results')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'E1 results: {output_dir.resolve()}', flush=True)
    print('Loading and validating training and validation data', flush=True)
    trial_runner = trial_runner or run_trial
    test_runner = test_runner or evaluate_final_trial
    if prepare_data is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-cased', use_fast=True)
        train_records, train_windows = load_e1_split(config, tokenizer, 'training')
        valid_records, valid_windows = load_e1_split(config, tokenizer, 'validation')
        data = tokenizer, train_records, train_windows, valid_records, valid_windows
    else:
        data = prepare_data(config)
    save_json(output_dir / 'protocol.json', {
        'baseline': asdict(config), 'learning_rates': [5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3],
        'dropouts': [0.0, 0.1, 0.2], 'seeds': [42, 9444, 2026],
        'batch_gate': 'strictly greater validation F1 than batch size 8',
        'tie_break': 'first candidate with equal F1',
        'weighted_loss': 'inverse supervised training token frequency with absent classes weighted zero',
        'scheduler_budget': 'full epoch budget with ceil 10 percent warmup steps',
        'test_policy': 'all three validation selected seed checkpoints after config lock',
        'smoke_only': config.limit_docs is not None})
    stages = {}

    def search(name, candidates):
        print(f'\nStage: {name} | {len(candidates)} trials', flush=True)
        trials = [trial_runner(candidate, output_dir / name / f'trial_{index:02d}', data)
                  for index, candidate in enumerate(candidates, 1)]
        stages[name] = trials
        selected = best_trial(trials)
        print(f'Stage complete | best Val F1={selected["validation_f1"]:.2%} | '
              f'selected={selected["path"]}', flush=True)
        save_json(output_dir / name / 'selection.json', {'selected': selected, 'candidates': trials})
        return replace(config, **{key: selected['config'][key] for key in
                                 ('learning_rate', 'dropout', 'loss', 'scheduler', 'batch_size', 'weight_decay')})

    selected = search('5.1.1 Learning rate', [replace(config, learning_rate=lr) for lr in [5e-4, 7.5e-4, 1e-3, 1.5e-3, 2e-3]])
    # Put the incumbent first so ties retain the simpler existing choice
    selected = search('5.1.2 Dropout', [replace(selected, dropout=p) for p in [0.1, 0.0, 0.2]])
    selected = search('5.1.3 Loss and scheduler/Loss', [replace(selected, loss=loss) for loss in ['cross_entropy', 'weighted_cross_entropy']])
    selected = search('5.1.3 Loss and scheduler/Scheduler', [replace(selected, scheduler=name) for name in ['constant', 'linear', 'linear_warmup']])
    stage4 = output_dir / '5.1.4 Batch size and weight decay'
    stage4.mkdir(parents=True, exist_ok=True)
    if config.run_stage4:
        baseline_f1 = best_trial(stages['5.1.3 Loss and scheduler/Scheduler'])['validation_f1']
        baseline = selected
        selected = search('5.1.4 Batch size and weight decay/Batch size', [replace(selected, batch_size=size) for size in [8, 16, 32]])
        batch_f1 = best_trial(stages['5.1.4 Batch size and weight decay/Batch size'])['validation_f1']
        improved = batch_f1 > baseline_f1 + config.batch_improvement_threshold
        save_json(stage4 / 'gate.json', {'baseline_f1': baseline_f1, 'best_batch_f1': batch_f1,
                                        'threshold': config.batch_improvement_threshold, 'search_weight_decay': improved})
        if improved:
            selected = search('5.1.4 Batch size and weight decay/Weight decay', [replace(selected, weight_decay=wd) for wd in [0.01, 0.0, 0.05]])
        else:
            print('No batch size improvement | Skipping weight decay search', flush=True)
            selected = baseline
    else:
        save_json(stage4 / 'gate.json', {'status': 'skipped by configuration'})
    # Lock hyperparameters before independent seed training and test access
    save_json(output_dir / 'selected_config.json', asdict(selected))
    final_dir = output_dir / '5.1.5 Seeds and final results'
    print('\nStage: 5.1.5 Seeds and final results | seeds=42, 9444, 2026', flush=True)
    trials = [trial_runner(replace(selected, seed=seed), final_dir / f'seed_{seed}', data) for seed in [42, 9444, 2026]]
    save_json(final_dir / 'validation_selection.json', trials)
    tests = []
    if config.evaluate_test:
        test_data = load_e1_split(config, data[0], 'testing')
        for trial in trials:
            print(f'Final test evaluation | seed={trial["config"]["seed"]}', flush=True)
            tests.append(test_runner(replace(selected, seed=trial['config']['seed']), trial, data[0], test_data))
    summary = summarize_seeds(trials, tests)
    save_json(final_dir / 'summary.json', summary)
    if render:
        print('Generating result tables and figures', flush=True)
        render_results(output_dir, stages, trials, tests, summary)
    print(f'E1 complete | Results: {output_dir.resolve()}', flush=True)
    return {'selected_config': asdict(selected), 'stages': stages, 'trials': trials, 'tests': tests, 'summary': summary}


# Run directly after editing the settings above
if __name__ == '__main__':
    run_experiment(E1Config())
