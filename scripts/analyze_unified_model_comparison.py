#!/usr/bin/env python3
"""Compare LLM feature-extractor models using ASB-LR (ours) downstream metrics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    feature_config: Path
    positive_label: str


DATASETS = [
    DatasetSpec("sms_spam", "SMS Spam", Path("config/human20/sms_spam_human20_features.json"), "spam"),
    DatasetSpec("sst2", "SST-2", Path("config/human20/sst2_human20_features.json"), "positive"),
    DatasetSpec("ade", "ADE", Path("config/human20/ade_human20_features.json"), "ade"),
    DatasetSpec(
        "disaster_tweets",
        "Disaster Tweets",
        Path("config/human20/disaster_tweets_human20_features.json"),
        "real disaster",
    ),
]

MODELS = [
    ("deepseek_v4_flash", "DeepSeek V4 Flash"),
    ("gpt_4_1_mini", "GPT-4.1 mini"),
    ("qwen_plus", "Qwen Plus"),
    ("claude_sonnet_4", "Claude Sonnet 4"),
    ("gemini_2_5_flash", "Gemini 2.5 Flash"),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_feature_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [feature["id"] for feature in json.load(f)]


def validate(rows: list[dict[str, str]], feature_ids: list[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = {"id", "split", "label", *feature_ids}.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for row in rows:
        for feature_id in feature_ids:
            if row[feature_id] not in {"0", "1"}:
                raise ValueError(f"{path}: invalid value for {row['id']} {feature_id}: {row[feature_id]}")


def y_values(rows: list[dict[str, str]], positive_label: str) -> list[int]:
    return [1 if row["label"] == positive_label else 0 for row in rows]


def feature_matrix(rows: list[dict[str, str]], feature_ids: list[str]) -> list[list[int]]:
    return [[int(row[feature_id]) for feature_id in feature_ids] for row in rows]


def evaluate(rows: list[dict[str, str]], feature_ids: list[str], positive_label: str, random_state: int) -> dict[str, float]:
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    y_test = y_values(test_rows, positive_label)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    model.fit(feature_matrix(train_rows, feature_ids), y_values(train_rows, positive_label))
    y_pred = model.predict(feature_matrix(test_rows, feature_ids)).tolist()
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro-F1": f1_score(y_test, y_pred, average="macro"),
        "Positive F1": f1_score(y_test, y_pred, pos_label=1),
    }


def feature_file(features_dir: Path, dataset_key: str, model_key: str) -> Path:
    return features_dir / f"{dataset_key}_5000_human20_{model_key}.csv"


def select_datasets(dataset_arg: str, datasets_arg: str | None) -> list[DatasetSpec]:
    if datasets_arg:
        requested = [item.strip() for item in datasets_arg.split(",") if item.strip()]
        known = {dataset.key: dataset for dataset in DATASETS}
        unknown = sorted(set(requested).difference(known))
        if unknown:
            raise ValueError(f"Unknown dataset keys: {unknown}")
        return [known[key] for key in requested]
    if dataset_arg == "all":
        return DATASETS
    return [dataset for dataset in DATASETS if dataset.key == dataset_arg]


def select_models(model_arg: str, models_arg: str | None) -> list[tuple[str, str]]:
    if models_arg:
        requested = [item.strip() for item in models_arg.split(",") if item.strip()]
        known = dict(MODELS)
        unknown = sorted(set(requested).difference(known))
        if unknown:
            raise ValueError(f"Unknown model keys: {unknown}")
        return [(key, known[key]) for key in requested]
    if model_arg == "all":
        return MODELS
    return [model for model in MODELS if model[0] == model_arg]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="output/features/unified_human20_models")
    parser.add_argument("--output", default="paper_tables/table_model_comparison_asb_lr.csv")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dataset", choices=["all", *[dataset.key for dataset in DATASETS]], default="all")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset keys, e.g. ade,disaster_tweets. Overrides --dataset.",
    )
    parser.add_argument("--model", choices=["all", *[model[0] for model in MODELS]], default="all")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys, e.g. deepseek_v4_flash,gpt_4_1_mini,qwen_plus.",
    )
    args = parser.parse_args()

    datasets = select_datasets(args.dataset, args.datasets)
    models = select_models(args.model, args.models)

    output_rows: list[dict[str, object]] = []
    for dataset in datasets:
        feature_ids = load_feature_ids(dataset.feature_config)
        for model_key, model_name in models:
            path = feature_file(Path(args.features_dir), dataset.key, model_key)
            if not path.exists():
                raise FileNotFoundError(path)
            rows = read_rows(path)
            validate(rows, feature_ids, path)
            metrics = evaluate(rows, feature_ids, dataset.positive_label, args.random_state)
            output_rows.append(
                {
                    "Dataset": dataset.display_name,
                    "Feature extractor": model_name,
                    "Classifier": "ASB-LR (ours)",
                    "Accuracy": f"{metrics['Accuracy']:.4f}",
                    "Macro-F1": f"{metrics['Macro-F1']:.4f}",
                    "Positive F1": f"{metrics['Positive F1']:.4f}",
                }
            )

    write_csv(
        Path(args.output),
        output_rows,
        ["Dataset", "Feature extractor", "Classifier", "Accuracy", "Macro-F1", "Positive F1"],
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
