#!/usr/bin/env python3
"""Add frozen-transformer embedding + LR baselines to the unified method table."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    data_path: Path
    positive_label: str


DATASETS = [
    DatasetSpec("sms_spam", "SMS Spam", Path("data/sms_spam_5000.csv"), "spam"),
    DatasetSpec("sst2", "SST-2", Path("data/sst2_5000.csv"), "positive"),
    DatasetSpec("ade", "ADE", Path("data/ade_5000.csv"), "ade"),
    DatasetSpec("disaster_tweets", "Disaster Tweets", Path("data/disaster_5000.csv"), "real disaster"),
]


MODEL_DISPLAY_NAMES = {
    "distilbert-base-uncased": "DistilBERT",
    "roberta-base": "RoBERTa",
}


def method_name(model_name: str) -> str:
    display = MODEL_DISPLAY_NAMES.get(model_name, model_name.split("/")[-1])
    if model_name == "roberta-base":
        return f"{display} embedding + LR"
    return f"Frozen {display} embedding + LR"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def y_values(rows: list[dict[str, str]], positive_label: str) -> list[int]:
    return [1 if row["label"] == positive_label else 0 for row in rows]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def metric_values(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro")),
        "Positive F1": float(f1_score(y_true, y_pred, pos_label=1)),
    }


def fmt_mean_std(values: list[float]) -> str:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"


def encode_transformer_embeddings(
    *,
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            attention = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * attention).sum(dim=1) / attention.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.detach().cpu().numpy())
            if (start // batch_size + 1) % 20 == 0:
                print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)} texts")
    return np.vstack(embeddings).astype(np.float32)


def load_or_create_embeddings(
    *,
    spec: DatasetSpec,
    rows: list[dict[str, str]],
    cache_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    refresh: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{spec.key}_{model_name.replace('/', '_')}_mean_pool_max{max_length}.npz"
    if cache_path.exists() and not refresh:
        data = np.load(cache_path)
        return data["embeddings"].astype(np.float32)
    print(f"Encoding {spec.display_name} with {model_name}")
    embeddings = encode_transformer_embeddings(
        texts=[row["text"] for row in rows],
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )
    np.savez_compressed(cache_path, embeddings=embeddings, ids=np.array([row["id"] for row in rows]))
    return embeddings


def run_dataset(
    *,
    spec: DatasetSpec,
    seeds: list[int],
    cache_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    refresh_embeddings: bool,
) -> list[dict[str, object]]:
    rows = read_rows(spec.data_path)
    method = method_name(model_name)
    embeddings = load_or_create_embeddings(
        spec=spec,
        rows=rows,
        cache_dir=cache_dir,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        refresh=refresh_embeddings,
    )
    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    test_indices = [index for index, row in enumerate(rows) if row["split"] == "test"]
    train_rows = [rows[index] for index in train_indices]
    test_rows = [rows[index] for index in test_indices]
    x_train = embeddings[train_indices]
    x_test = embeddings[test_indices]
    y_train = y_values(train_rows, spec.positive_label)
    y_test = y_values(test_rows, spec.positive_label)

    output_rows: list[dict[str, object]] = []
    for seed in seeds:
        set_seed(seed)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test).tolist()
        metrics = metric_values(y_test, y_pred)
        output_rows.append(
            {
                "Dataset": spec.display_name,
                "Input": "raw text",
                "Method": method,
                "Embedding model": model_name,
                "seed": seed,
                **metrics,
            }
        )
    return output_rows


def summarize(per_seed_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in per_seed_rows:
        key = (str(row["Dataset"]), str(row["Method"]), str(row["Embedding model"]))
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, object]] = []
    for dataset in ["SMS Spam", "SST-2", "ADE", "Disaster Tweets"]:
        dataset_keys = sorted(key for key in grouped if key[0] == dataset)
        for _, method, model_name in dataset_keys:
            values = grouped[(dataset, method, model_name)]
            rows.append(
                {
                    "Dataset": dataset,
                    "Input": "raw text",
                    "Method": method,
                    "Embedding model": model_name,
                    "Accuracy": fmt_mean_std([float(row["Accuracy"]) for row in values]),
                    "Macro-F1": fmt_mean_std([float(row["Macro-F1"]) for row in values]),
                    "Positive F1": fmt_mean_std([float(row["Positive F1"]) for row in values]),
                }
            )
    return rows


def merge_into_method_table(main_table: Path, embedding_rows: list[dict[str, object]]) -> None:
    if not main_table.exists():
        return
    with main_table.open(encoding="utf-8", newline="") as f:
        existing_rows = list(csv.DictReader(f))
    embedding_by_dataset: dict[str, list[dict[str, object]]] = {}
    for row in embedding_rows:
        embedding_by_dataset.setdefault(str(row["Dataset"]), []).append(row)
    embedding_methods = {str(row["Method"]) for row in embedding_rows}
    merged: list[dict[str, object]] = []
    inserted: set[str] = set()
    for row in existing_rows:
        if row["Method"] in embedding_methods:
            continue
        merged.append(row)
        if row["Method"] == "TF-IDF + SVM" and row["Dataset"] in embedding_by_dataset:
            merged.extend(embedding_by_dataset[row["Dataset"]])
            inserted.add(row["Dataset"])
    for dataset, rows in embedding_by_dataset.items():
        if dataset not in inserted:
            merged.extend(rows)
    write_csv(main_table, merged, ["Dataset", "Input", "Method", "Accuracy", "Macro-F1", "Positive F1"])


def parse_model_names(values: list[str]) -> list[str]:
    model_names: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item and item not in model_names:
                model_names.append(item)
    return model_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="paper_tables/table_embedding_lr_baseline.csv")
    parser.add_argument("--per-seed-output", default="output/results/embedding_lr_baseline_by_seed.csv")
    parser.add_argument("--merge-table", default="paper_tables/table_unified_method_performance.csv")
    parser.add_argument("--cache-dir", default="output/embeddings")
    parser.add_argument("--model-name", default=None, help="Backward-compatible alias for --model-names")
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=None,
        help="One or more HuggingFace model names, comma-separated or space-separated.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--dataset", choices=["all", *[dataset.key for dataset in DATASETS]], default="all")
    parser.add_argument("--refresh-embeddings", action="store_true")
    parser.add_argument("--no-merge", action="store_true")
    args = parser.parse_args()

    raw_model_names = args.model_names or ([args.model_name] if args.model_name else ["distilbert-base-uncased"])
    model_names = parse_model_names(raw_model_names)
    specs = DATASETS if args.dataset == "all" else [spec for spec in DATASETS if spec.key == args.dataset]
    per_seed_rows: list[dict[str, object]] = []
    for model_name in model_names:
        for spec in specs:
            per_seed_rows.extend(
                run_dataset(
                    spec=spec,
                    seeds=args.seeds,
                    cache_dir=Path(args.cache_dir),
                    model_name=model_name,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    refresh_embeddings=args.refresh_embeddings,
                )
            )

    write_csv(
        Path(args.per_seed_output),
        per_seed_rows,
        ["Dataset", "Input", "Method", "Embedding model", "seed", "Accuracy", "Macro-F1", "Positive F1"],
    )
    summary_rows = summarize(per_seed_rows)
    write_csv(
        Path(args.output),
        summary_rows,
        ["Dataset", "Input", "Method", "Embedding model", "Accuracy", "Macro-F1", "Positive F1"],
    )
    if not args.no_merge:
        merge_into_method_table(Path(args.merge_table), summary_rows)
    print(f"Wrote {args.per_seed_output}")
    print(f"Wrote {args.output}")
    if not args.no_merge:
        print(f"Updated {args.merge_table}")


if __name__ == "__main__":
    main()
