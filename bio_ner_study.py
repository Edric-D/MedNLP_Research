"""Train and compare six BioBERT disease NER experiments."""
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import csv
import json
import math
import random
import statistics
import time

import torch
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoModelForTokenClassification, BertConfig

# In the notebook these are the existing evaluator and window helpers.
if "__file__" in globals():
    from e1_head_only import (
        LABELS, decode_document, evaluate_standardized_records, load_jsonl_records,
        make_windows,
    )

BASE_MODEL = "dmis-lab/biobert-base-cased-v1.1"
BASE_REVISION = "924f12e0c3db7f156a765ad53fb6b11e7afedbc8"
TOKENIZER_REVISION = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
SCOPES = dict(E1="head", E2="last4", E3="full", E4="full", E5="full", E6="full")
STRATEGIES = dict(E4="token", E5="word", E6="entity")
DEFAULT_SEEDS = (42, 43, 44)
WINDOW = 512
OVERLAP = 64


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def precision_context(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def shuffled_batches(rows, size, rng=None):
    # Sorting within shuffled pools reduces padding while retaining different
    # training batches in each epoch. Evaluation order is length-sorted.
    indices = list(range(len(rows)))
    if rng is not None:
        rng.shuffle(indices)
    pools = [indices[i:i + size * 8] for i in range(0, len(indices), size * 8)]
    batches = []
    for pool in pools:
        pool.sort(key=lambda i: len(rows[i]["input_ids"]))
        batches.extend(pool[i:i + size] for i in range(0, len(pool), size))
    if rng is not None:
        rng.shuffle(batches)
    for batch in batches:
        yield [rows[i] for i in batch]


def collate(rows, tokenizer, device):
    length = math.ceil(max(len(row["input_ids"]) for row in rows) / 8) * 8
    ids, attention, labels = [], [], []
    for row in rows:
        pad = length - len(row["input_ids"])
        ids.append(row["input_ids"] + [tokenizer.pad_token_id] * pad)
        attention.append([1] * len(row["input_ids"]) + [0] * pad)
        if "labels" in row:
            labels.append(row["labels"] + [-100] * pad)
    inputs = {
        "input_ids": torch.tensor(ids, device=device),
        "attention_mask": torch.tensor(attention, device=device),
    }
    if labels:
        inputs["labels"] = torch.tensor(labels, device=device)
    return inputs


def read_ner_split(project, split, tokenizer, limit=None):
    cached = load_jsonl_records(project / "Dataset/TokenizedData/BioBERT" / f"NCBI_{split}.jsonl")
    gold = load_jsonl_records(project / "Dataset/ProcessedData" / f"NCBI_{split}.jsonl")
    if len(cached) != len(gold):
        raise ValueError(f"Tokenized and gold record counts differ for {split}")
    records = []
    for source, reference in zip(cached[:limit], gold[:limit], strict=True):
        if any(source[key] != reference[key] for key in ("record_id", "pmid", "text")):
            raise ValueError(f"Unaligned record {source['record_id']}")
        row = dict(source, entities=reference["entities"])
        if split != "training":
            row["bio_tags"] = ["O"] * len(row["input_ids"])
        records.append(row)
    return records, make_windows(records, tokenizer, WINDOW, OVERLAP)


def ner_config():
    return BertConfig.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )


def load_ner(scope, seed, device, adapted=None):
    set_seed(seed)
    source = str(adapted) if adapted else BASE_MODEL
    kwargs = {} if adapted else {"revision": BASE_REVISION}
    model = AutoModelForTokenClassification.from_pretrained(source, config=ner_config(), **kwargs)
    # Match the new NER head across adapted and unadapted models exactly.
    set_seed(seed)
    model.classifier.reset_parameters()
    head_hash = sha256(
        model.classifier.weight.detach().numpy().tobytes()
        + model.classifier.bias.detach().numpy().tobytes()
    ).hexdigest()
    if scope != "full":
        model.bert.requires_grad_(False)
        if scope == "last4":
            for layer in model.bert.encoder.layer[-4:]:
                layer.requires_grad_(True)
    return model.to(device), head_hash


@torch.inference_mode()
def score_ner(model, split, tokenizer, device, batch_size):
    records, windows = split
    model.eval()
    sums = [torch.zeros((len(r["input_ids"]), len(LABELS))) for r in records]
    counts = [torch.zeros(len(r["input_ids"])) for r in records]
    first = [torch.full((len(r["input_ids"]),), -1) for r in records]
    conflicts = [torch.zeros(len(r["input_ids"]), dtype=torch.bool) for r in records]
    for rows in shuffled_batches(windows, batch_size):
        inputs = collate(rows, tokenizer, device)
        inputs.pop("labels")
        with precision_context(device):
            logits = model(**inputs).logits.float().cpu()
        for row, values in zip(rows, logits, strict=True):
            doc, start, end = row["doc"], row["start"], row["end"]
            region = slice(start, end)
            values = values[1:1 + end - start]
            tag = values.argmax(-1)
            old = first[doc][region]
            conflicts[doc][region] |= (old >= 0) & (old != tag)
            first[doc][region] = tag
            sums[doc][region] += values
            counts[doc][region] += 1
    predictions, illegal = [], 0
    for record, total, count in zip(records, sums, counts, strict=True):
        if not torch.all(count > 0):
            raise ValueError(f"Uncovered document tokens in {record['record_id']}")
        pred, bad = decode_document(record, (total / count[:, None]).argmax(-1))
        predictions.append(pred)
        illegal += bad
    result = evaluate_standardized_records(records, predictions)
    result.update(
        illegal_bio_transitions=illegal,
        window_overlap_token_conflicts=sum(int(flags.sum()) for flags in conflicts),
    )
    return result, predictions


def train_ner(folder, experiment, seed, tokenizer, train, validation, device,
              encoder_lr, head_lr, epochs, min_epochs, patience, batch_size, adapted=None,
              evaluate_train=True):
    scope = SCOPES[experiment]
    model, head_hash = load_ner(scope, seed, device, adapted)
    encoder = [p for p in model.bert.parameters() if p.requires_grad]
    head = list(model.classifier.parameters())
    parameters = encoder + head
    groups = [{"params": head, "lr": head_lr}]
    if encoder:
        groups.append({"params": encoder, "lr": encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01, fused=device.type == "cuda")
    config = dict(
        experiment=experiment, seed=seed, scope=scope, base_model=BASE_MODEL,
        base_revision=BASE_REVISION, tokenizer_revision=TOKENIZER_REVISION,
        adapted=str(adapted) if adapted else None, head_init_sha256=head_hash,
        encoder_lr=encoder_lr if encoder else None, head_lr=head_lr, epochs=epochs,
        min_epochs=min_epochs, patience=patience, batch_size=batch_size,
        max_length=WINDOW, overlap=OVERLAP, optimizer="AdamW",
        weight_decay=0.01, gradient_clip=1.0, device=str(device),
        trainable_parameters=sum(p.numel() for p in parameters),
    )
    write_json(folder / "config.json", config)
    best_state, best, best_epoch, stale = None, -1, 0, 0
    history, optimization_seconds = [], 0.0
    wall_start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        if scope == "head":
            model.bert.eval()
        elif scope == "last4":
            model.bert.embeddings.eval()
            for layer in model.bert.encoder.layer[:-4]:
                layer.eval()
        set_seed(seed + 1000 * epoch)
        losses = torch.zeros((), device=device)
        tokens = 0
        train_start = time.perf_counter()
        for rows in shuffled_batches(train[1], batch_size, random.Random(seed + epoch)):
            inputs = collate(rows, tokenizer, device)
            count = sum(sum(tag != -100 for tag in row["labels"]) for row in rows)
            optimizer.zero_grad(set_to_none=True)
            with precision_context(device):
                loss = model(**inputs).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses += loss.detach() * count
            tokens += count
        synchronize(device)
        optimization_seconds += time.perf_counter() - train_start
        validation_metrics, predictions = score_ner(model, validation, tokenizer, device, batch_size)
        f1 = validation_metrics["typed_exact"]["f1"]
        train_loss = losses.item() / tokens
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Nonfinite NER loss in {experiment} seed {seed}")
        history.append(dict(epoch=epoch, train_loss=train_loss, validation_f1=f1))
        write_json(folder / "history.json", history)
        if f1 > best:
            best, best_epoch, stale = f1, epoch, 0
            best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
            write_json(folder / "validation_metrics.json", validation_metrics)
            write_json(folder / "validation_predictions.json", predictions)
        else:
            stale += 1
        print(f"{experiment} seed={seed} epoch={epoch:02} loss={train_loss:.4f} "
              f"val_f1={f1:.4f}", flush=True)
        if epoch >= min_epochs and stale >= patience:
            break
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    model.load_state_dict(best_state)
    checkpoint = folder / "best_checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    wall_seconds = time.perf_counter() - wall_start
    if evaluate_train:
        metrics, _ = score_ner(model, train, tokenizer, device, batch_size)
        write_json(folder / "training_metrics.json", metrics)
    summary = dict(
        best_epoch=best_epoch, completed_epochs=len(history), validation_f1=best,
        optimization_seconds=optimization_seconds, wall_seconds=wall_seconds,
        peak_cuda_bytes=peak, trainable_parameters=config["trainable_parameters"],
    )
    write_json(folder / "summary.json", summary)
    return summary


def make_mlm_splits(records, tokenizer, seed, block_size=256):
    by_pmid = {}
    for record in records:
        pmid = record["pmid"]
        if pmid in by_pmid:
            previous = by_pmid[pmid]
            if any(record[k] != previous[k] for k in ("text", "entities")):
                raise ValueError(f"Conflicting records for duplicate PMID {pmid}")
        by_pmid.setdefault(pmid, record)
    pmids = sorted(by_pmid)
    random.Random(seed).shuffle(pmids)
    cutoff = max(1, round(len(pmids) * 0.1))
    groups = dict(training=pmids[cutoff:], validation=pmids[:cutoff])
    blocks = {}
    for split, keys in groups.items():
        blocks[split] = []
        for pmid in keys:
            record = by_pmid[pmid]
            # Tokenizer offsets and word IDs agree with the full NER input.
            encoded = tokenizer(record["text"], add_special_tokens=False, return_offsets_mapping=True,
                                verbose=False)
            if encoded["input_ids"] != record["input_ids"]:
                raise ValueError(f"MLM tokenization mismatch for {pmid}")
            word_ids = encoded.word_ids()
            words = []
            for index, word_id in enumerate(word_ids):
                if index == 0 or word_id != word_ids[index - 1]:
                    words.append([])
                words[-1].append(index)
            # BIO spans use the same corrected training offsets as NER.
            entities, active = [], []
            for i, tag in enumerate(record["bio_tags"]):
                if not tag.startswith("I-") and active:
                    entities.append(active)
                    active = []
                if tag != "O":
                    active.append(i)
            if active:
                entities.append(active)
            allowed_ends = {group[-1] + 1 for group in words}
            start = 0
            while start < len(word_ids):
                end = min(start + block_size - 2, len(word_ids))
                while end > start and end not in allowed_ends:
                    end -= 1
                if end == start:
                    raise ValueError(f"Word longer than MLM capacity for {pmid}")
                blocks[split].append(dict(
                    input_ids=[tokenizer.cls_token_id] + record["input_ids"][start:end]
                    + [tokenizer.sep_token_id],
                    words=[[i - start + 1 for i in g] for g in words if start <= g[0] < end],
                    entities=[[i - start + 1 for i in g] for g in entities
                              if start <= g[0] and g[-1] < end],
                    entity_tokens={i - start + 1 for g in entities for i in g if start <= i < end},
                ))
                start = end
    return blocks, groups


def fit_groups(groups, budget, rng):
    """Select whole groups without exceeding budget, preferring a full budget."""
    groups = list(groups)
    rng.shuffle(groups)
    picked = []
    for group in groups:
        if len(group) <= budget:
            picked.extend(group)
            budget -= len(group)
            if budget == 0:
                break
    return picked


def mask_row(row, strategy, tokenizer, rng):
    ids = row["input_ids"]
    budget = max(1, round((len(ids) - 2) * 0.15))
    if strategy == "token":
        picked = rng.sample(range(1, len(ids) - 1), budget)
        entity_count = 0
    elif strategy == "word" or not row["entities"]:
        picked = fit_groups(row["words"], budget, rng)
        entity_count = 0
    else:
        entities = row["entities"]
        entity_tokens = row["entity_tokens"]
        non_entity_words = [group for group in row["words"] if entity_tokens.isdisjoint(group)]
        # Prefer half the budget from complete mentions and backfill with
        # non-entity words. Fallback can include additional complete mentions.
        picked = fit_groups(entities, round(budget / 2), rng)
        entity_count = len(picked)
        picked += fit_groups(non_entity_words, budget - len(picked), rng)
        picked_set = set(picked)
        unused = [span for span in entities if picked_set.isdisjoint(span)]
        extra = fit_groups(unused, budget - len(picked), rng)
        picked += extra
        entity_count += len(extra)
    if not picked:
        # Whole groups can exceed the nominal budget for a very short block.
        candidates = row["words"] if strategy != "entity" else [
            *row["entities"], *[g for g in row["words"]
                                 if row["entity_tokens"].isdisjoint(g)]
        ]
        picked = min(candidates, key=len)
        entity_count = len(picked) if strategy == "entity" and picked in row["entities"] else 0
    inputs, labels = ids.copy(), [-100] * len(ids)
    for i in picked:
        labels[i] = ids[i]
        draw = rng.random()
        if draw < 0.8:
            inputs[i] = tokenizer.mask_token_id
        elif draw < 0.9:
            inputs[i] = rng.randrange(len(tokenizer))
    return dict(input_ids=inputs, labels=labels), len(picked), entity_count


def mlm_loss(model, inputs):
    labels = inputs.pop("labels")
    selected = labels != -100
    hidden = model.bert(**inputs).last_hidden_state
    # Apply the vocabulary projection only at positions that contribute to loss.
    logits = model.cls(hidden[selected])
    return F.cross_entropy(logits, labels[selected])


def run_mlm_epoch(model, rows, strategy, tokenizer, device, seed, batch_size,
                  optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    rng = random.Random(seed)
    loss_sum = torch.zeros((), device=device)
    total, targeted, eligible = 0, 0, 0
    context = nullcontext() if training else torch.inference_mode()
    with context:
        for batch in shuffled_batches(rows, batch_size, rng if training else None):
            masked = [mask_row(row, strategy, tokenizer, rng) for row in batch]
            inputs = collate([item[0] for item in masked], tokenizer, device)
            count = sum(item[1] for item in masked)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with precision_context(device):
                loss = mlm_loss(model, inputs)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            loss_sum += loss.detach() * count
            total += count
            targeted += sum(item[2] for item in masked)
            eligible += sum(len(row["input_ids"]) - 2 for row in batch)
    return dict(loss=loss_sum.item() / total, masked=total, eligible=eligible,
                mask_fraction=total / eligible, entity_targeted=targeted)


def train_mlm(folder, experiment, seed, tokenizer, data, groups, device, epochs, batch_size):
    strategy = STRATEGIES[experiment]
    set_seed(seed)
    config = BertConfig.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    # The original checkpoint contains separate decoder and embedding tensors.
    config.tie_word_embeddings = False
    model, loading = AutoModelForMaskedLM.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, config=config, output_loading_info=True,
    )
    if loading["missing_keys"] or loading["mismatched_keys"]:
        raise RuntimeError(f"Incomplete BioBERT MLM initialization: {loading}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01,
                                  fused=device.type == "cuda")
    steps = epochs * math.ceil(len(data["training"]) / batch_size)
    warmup = max(1, round(steps * 0.10))
    def lr_multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (steps - step) / max(1, steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    write_json(folder / "config.json", dict(
        experiment=experiment, strategy=strategy, seed=seed, epochs=epochs,
        base_model=BASE_MODEL, revision=BASE_REVISION, lr=5e-5, weight_decay=0.01,
        batch_size=batch_size, optimizer_steps=steps, block_size=256, target_mask_fraction=0.15,
        checkpoint_policy="fixed final optimization step", split=groups,
    ))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history, optimization_seconds = [], 0.0
    for epoch in range(1, epochs + 1):
        set_seed(seed + 1000 * epoch)
        train_start = time.perf_counter()
        train = run_mlm_epoch(model, data["training"], strategy, tokenizer, device,
                              seed + epoch, batch_size, optimizer, scheduler)
        synchronize(device)
        optimization_seconds += time.perf_counter() - train_start
        valid = run_mlm_epoch(model, data["validation"], strategy, tokenizer, device,
                              seed + 100_000, batch_size)
        if not all(math.isfinite(item["loss"]) for item in (train, valid)):
            raise FloatingPointError(f"Nonfinite MLM loss in {experiment} seed {seed}")
        history.append(dict(epoch=epoch, training=train, validation=valid))
        write_json(folder / "history.json", history)
        print(f"{experiment} MLM seed={seed} epoch={epoch:02} train_loss={train['loss']:.4f} "
              f"val_loss={valid['loss']:.4f} mask={train['mask_fraction']:.4f}", flush=True)
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    checkpoint = folder / "final_checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    write_json(folder / "summary.json", dict(
        optimization_seconds=optimization_seconds, wall_seconds=time.perf_counter() - started,
        peak_cuda_bytes=peak, optimizer_steps=steps,
        mask_fraction=sum(item["training"]["masked"] for item in history)
        / sum(item["training"]["eligible"] for item in history),
        entity_mask_fraction=sum(item["training"]["entity_targeted"] for item in history)
        / sum(item["training"]["masked"] for item in history),
    ))
    return checkpoint


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(run_dir, seeds):
    rows = []
    for experiment in SCOPES:
        for seed in seeds:
            folder = run_dir / experiment / str(seed)
            result = read_json(folder / "test_metrics.json")
            summary = read_json(folder / "summary.json")
            train = read_json(folder / "training_metrics.json")
            strict = result["typed_exact"]
            rows.append(dict(
                experiment=experiment, seed=seed, precision=strict["precision"],
                recall=strict["recall"], micro_f1=strict["f1"],
                macro_f1=result["typed_macro"]["f1"], span_f1=result["span_only_exact"]["f1"],
                validation_f1=summary["validation_f1"], training_f1=train["typed_exact"]["f1"],
                illegal_bio=result["illegal_bio_transitions"],
                overlap_conflicts=result["window_overlap_token_conflicts"],
                best_epoch=summary["best_epoch"], ner_seconds=summary["optimization_seconds"],
                ner_wall_seconds=summary["wall_seconds"],
                ner_peak_cuda_bytes=summary["peak_cuda_bytes"],
                trainable_parameters=summary["trainable_parameters"],
                **({f"mlm_{k}": v for k, v in read_json(folder / "mlm/summary.json").items()}
                   if experiment in STRATEGIES else {}),
            ))
    reference = {row["seed"]: row["micro_f1"] for row in rows if row["experiment"] == "E3"}
    for row in rows:
        row["delta_vs_e3"] = row["micro_f1"] - reference[row["seed"]]
    stats = {}
    for exp in SCOPES:
        group = [row for row in rows if row["experiment"] == exp]
        stats[exp] = {
            key: dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values) > 1 else None)
            for key in group[0] if key not in {"experiment", "seed"}
            if (values := [row[key] for row in group if row[key] is not None])
        }
    write_json(run_dir / "results.json", dict(seeds=list(seeds), runs=rows, summary=stats))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return stats


def run_study(project, tokenizer, run_dir, *, seeds=DEFAULT_SEEDS, smoke=False, source_sha=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    batch_size = 16 if device.type == "cuda" else 4
    mlm_batch_size = 32 if device.type == "cuda" else 4
    # Fixed budgets preserve matched downstream settings for E3–E6.
    ner_epochs, min_epochs, patience = (1, 1, 1) if smoke else (30, 8, 4)
    mlm_epochs = 1 if smoke else 10
    seeds = tuple(seeds)
    limit = 8 if smoke else None
    train = read_ner_split(project, "training", tokenizer, limit)
    val = read_ner_split(project, "validation", tokenizer, limit)
    study = dict(
        seeds=list(seeds), smoke=smoke, source_sha256=source_sha, device=str(device),
        torch_version=str(torch.__version__),
        transformers_version=__import__("transformers").__version__,
        device_name=torch.cuda.get_device_name() if device.type == "cuda" else str(device),
        precision="bfloat16 mixed" if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else "float32",
        ner_epochs=ner_epochs, min_epochs=min_epochs, patience=patience,
        batch_size=batch_size, mlm_epochs=mlm_epochs, mlm_batch_size=mlm_batch_size,
        base_revision=BASE_REVISION, tokenizer_revision=TOKENIZER_REVISION,
        raw_sha256={path.name: sha256(path.read_bytes()).hexdigest() for path in
                    sorted((project / "Dataset/RawData").glob("NCBI_*.txt"))},
    )
    if (run_dir / "study.json").exists() and read_json(run_dir / "study.json") != study:
        raise ValueError("Existing run directory uses different study settings")
    write_json(run_dir / "study.json", study)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    shared = dict(tokenizer=tokenizer, train=train, validation=val, device=device,
                  epochs=ner_epochs, min_epochs=min_epochs, patience=patience, batch_size=batch_size)
    # E1 uses the original validation-selected head LR. E2/E3 receive identical
    # validation search budgets, and E3's selection is locked for E4–E6.
    choices = [2e-5] if smoke else [1e-5, 2e-5, 3e-5]
    selected = {}
    for experiment in ("E2", "E3"):
        trials = {}
        for lr in choices:
            folder = run_dir / "tuning" / experiment / f"{lr:g}"
            if (folder / "summary.json").exists():
                summary = read_json(folder / "summary.json")
            else:
                summary = train_ner(folder, experiment, seeds[0], encoder_lr=lr,
                                    head_lr=1e-4, evaluate_train=True, **shared)
            trials[lr] = summary["validation_f1"]
        selected[experiment] = max(choices, key=trials.get)
        write_json(run_dir / "tuning" / f"{experiment}.json", dict(selected=selected[experiment],
                                                                  validation_f1=trials))
    for seed in seeds:
        # Same deterministic document partition for the three MLM strategies.
        mlm_data, groups = make_mlm_splits(train[0], tokenizer, seed)
        for experiment in SCOPES:
            folder = run_dir / experiment / str(seed)
            if (folder / "summary.json").exists():
                continue
            adapted = None
            if experiment in STRATEGIES:
                mlm_dir = folder / "mlm"
                if (mlm_dir / "summary.json").exists():
                    adapted = mlm_dir / "final_checkpoint"
                else:
                    adapted = train_mlm(mlm_dir, experiment, seed, tokenizer, mlm_data,
                                        groups, device, mlm_epochs, mlm_batch_size)
            if experiment in selected and seed == seeds[0]:
                # Reuse the validation-selected run. The same candidate would
                # otherwise be retrained on the same seed with identical settings.
                import shutil
                source = run_dir / "tuning" / experiment / f"{selected[experiment]:g}"
                shutil.copytree(source, folder, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("summary.json"))
                write_json(folder / "summary.json", read_json(source / "summary.json"))
            else:
                train_ner(folder, experiment, seed, encoder_lr=selected.get(experiment, selected["E3"]),
                          head_lr=1e-3 if experiment == "E1" else 1e-4, adapted=adapted, **shared)
    write_json(run_dir / "selection_frozen.json", dict(
        rule="highest validation typed exact-span micro F1", selected_encoder_lr=selected,
        head_lr=dict(E1=1e-3, E2_to_E6=1e-4), seeds=list(seeds),
    ))
    # Test is first loaded here, after every experiment has been trained and selected.
    test = read_ner_split(project, "testing", tokenizer, limit)
    for seed in seeds:
        for experiment in SCOPES:
            folder = run_dir / experiment / str(seed)
            if (folder / "test_metrics.json").exists():
                continue
            model = AutoModelForTokenClassification.from_pretrained(folder / "best_checkpoint").to(device)
            metrics, predictions = score_ner(model, test, tokenizer, device, batch_size)
            write_json(folder / "test_predictions.json", predictions)
            write_json(folder / "test_metrics.json", metrics)
            print(f"TEST {experiment} seed={seed} micro_f1={metrics['typed_exact']['f1']:.4f}",
                  flush=True)
            del model
    return aggregate(run_dir, seeds)


def study_cli():
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    project = Path(__file__).resolve().parent
    run_dir = args.run_dir or project / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased",
                                              revision=TOKENIZER_REVISION, use_fast=True)
    source_sha = sha256(Path(__file__).read_bytes()).hexdigest()
    run_study(project, tokenizer, run_dir.resolve(), seeds=args.seeds, smoke=args.smoke,
              source_sha=source_sha)


if globals().get("__file__", "").endswith("bio_ner_study.py") and __name__ == "__main__":
    study_cli()
