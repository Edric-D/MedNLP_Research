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


def set_counts(gold_values, predicted_values):
    return (
        len(gold_values & predicted_values),
        len(predicted_values - gold_values),
        len(gold_values - predicted_values),
    )


def boundary_counts(gold_values, predicted_values):
    gold_counter = Counter(gold_values)
    predicted_counter = Counter(predicted_values)
    return (
        sum((gold_counter & predicted_counter).values()),
        sum((predicted_counter - gold_counter).values()),
        sum((gold_counter - predicted_counter).values()),
    )


def spans_to_characters(spans):
    characters = set()
    for start, end in spans:
        characters.update(range(start, end))
    return characters


def add_counts(total, current):
    for index, value in enumerate(current):
        total[index] += value


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


def select_device(torch):
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def main():
    import argparse
    import random
    import subprocess
    import time
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer, BertConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'Dataset/TokenizedData/BioBERT')
    parser.add_argument('--gold-dir', type=Path, default=Path(__file__).resolve().parent / 'Dataset/ProcessedData')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results/E1_head_only')
    parser.add_argument('--model', default='dmis-lab/biobert-base-cased-v1.1')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--evaluate-checkpoint', type=Path, help='Evaluate a frozen selected checkpoint without training; requires --test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit-docs', type=int, help='Smoke test only; not a formal experiment')
    parser.add_argument('--test', action='store_true', help='Evaluate Test after selecting the checkpoint on Validation')
    args = parser.parse_args()
    if args.evaluate_checkpoint and not args.test:
        parser.error('--evaluate-checkpoint requires --test')
    if min(args.epochs, args.batch_size, args.patience) < 1 or (args.limit_docs is not None and args.limit_docs < 1):
        parser.error('epochs, batch-size, patience and limit-docs must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output-dir must be empty; choose a new directory for each run')
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
        raise ValueError('Checkpoint label mapping differs from E1')
    if args.max_length > model.config.max_position_embeddings:
        parser.error('max-length exceeds model position embeddings')
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert {name for name, p in model.named_parameters() if p.requires_grad} == {'classifier.weight', 'classifier.bias'}
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

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
    config.update(experiment='E1', device=str(device), labels=LABELS,
                  trainable_parameters=sum(p.numel() for p in trainable),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  torch_version=torch.__version__, frozen_scope='entire BioBERT backbone',
                  selection='maximum validation typed exact-span micro F1')
    config['git_commit'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip()
    config['git_dirty'] = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=Path(__file__).parent, capture_output=True, text=True).stdout.strip())
    save_json('config.json', config)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    best_f1, stale, history, best_epoch = -1, 0, [], None
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            model.base_model.eval()  # Keep the frozen encoder deterministic; train head dropout.
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


if __name__ == '__main__':
    main()
