#!/usr/bin/env python3
"""Run condensed and full ASB feature-group ablations across datasets."""

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
    feature_file: Path
    feature_config: Path
    group_config: Path
    positive_label: str
    condensed_groups: tuple[str, ...]


def dataset_specs(features_dir: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            key="sms_spam",
            display_name="SMS Spam",
            feature_file=features_dir / "sms_spam_5000_human20_strict.csv",
            feature_config=Path("config/human20/sms_spam_human20_features.json"),
            group_config=Path("config/human20/sms_spam_human20_feature_groups.json"),
            positive_label="spam",
            condensed_groups=("deception", "money", "urgency", "action_request", "personal_normal"),
        ),
        DatasetSpec(
            key="sst2",
            display_name="SST-2",
            feature_file=features_dir / "sst2_5000_human20_strict.csv",
            feature_config=Path("config/human20/sst2_human20_features.json"),
            group_config=Path("config/human20/sst2_human20_feature_groups.json"),
            positive_label="positive",
            condensed_groups=("positive_sentiment", "negative_sentiment", "aspect_evaluation"),
        ),
        DatasetSpec(
            key="ade",
            display_name="ADE",
            feature_file=features_dir / "ade_5000_human20_strict.csv",
            feature_config=Path("config/human20/ade_human20_features.json"),
            group_config=Path("config/human20/ade_human20_feature_groups.json"),
            positive_label="ade",
            condensed_groups=("causality_temporality", "clinical_event", "non_ade_context"),
        ),
        DatasetSpec(
            key="disaster_tweets",
            display_name="Disaster Tweets",
            feature_file=features_dir / "disaster_tweets_5000_human20_strict.csv",
            feature_config=Path("config/human20/disaster_tweets_human20_features.json"),
            group_config=Path("config/human20/disaster_tweets_human20_feature_groups.json"),
            positive_label="real disaster",
            condensed_groups=("disaster_event", "grounding", "severity_damage", "figurative_non_disaster"),
        ),
    ]


DISPLAY_NAMES = {
    "action_request": "action request",
    "aspect_evaluation": "aspect evaluation",
    "causality_temporality": "causality/temporality",
    "clinical_event": "clinical event",
    "deception": "deception",
    "disaster_event": "disaster event",
    "emergency_response": "emergency response",
    "figurative_non_disaster": "figurative/non-disaster",
    "grounding": "grounding",
    "medication": "medication",
    "money": "money",
    "negative_sentiment": "negative sentiment",
    "negation_uncertainty": "negation/uncertainty",
    "non_ade_context": "non-ADE context",
    "personal_normal": "personal/normal",
    "positive_sentiment": "positive sentiment",
    "recommendation": "recommendation",
    "severity_damage": "severity/damage",
    "uncertainty_irony": "uncertainty/irony",
    "urgency": "urgency",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate(rows: list[dict[str, str]], feature_ids: list[str], groups: dict[str, list[str]]) -> None:
    if not rows:
        raise ValueError("No rows found")
    missing = {"id", "split", "label", *feature_ids}.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    grouped = [feature for features in groups.values() for feature in features]
    if set(grouped) != set(feature_ids):
        raise ValueError("Feature groups do not cover exactly the feature schema")
    duplicates = sorted({feature for feature in grouped if grouped.count(feature) > 1})
    if duplicates:
        raise ValueError(f"Duplicate grouped features: {duplicates}")
    for row in rows:
        for feature_id in feature_ids:
            if row[feature_id] not in {"0", "1"}:
                raise ValueError(f"Invalid feature value for {row['id']} {feature_id}: {row[feature_id]}")


def y_values(rows: list[dict[str, str]], positive_label: str) -> list[int]:
    return [1 if row["label"] == positive_label else 0 for row in rows]


def feature_matrix(rows: list[dict[str, str]], feature_ids: list[str]) -> list[list[int]]:
    return [[int(row[feature_id]) for feature_id in feature_ids] for row in rows]


def evaluate_feature_set(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_ids: list[str],
    positive_label: str,
    random_state: int,
) -> dict[str, float]:
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    model.fit(feature_matrix(train_rows, feature_ids), y_values(train_rows, positive_label))
    y_test = y_values(test_rows, positive_label)
    y_pred = model.predict(feature_matrix(test_rows, feature_ids)).tolist()
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro-F1": f1_score(y_test, y_pred, average="macro"),
        "Positive F1": f1_score(y_test, y_pred, pos_label=1),
    }


def metric_row(
    dataset: str,
    feature_group_used: str,
    feature_ids: list[str],
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "Dataset": dataset,
        "Feature group used": feature_group_used,
        "#Features": len(feature_ids),
        "Accuracy": f"{metrics['Accuracy']:.4f}",
        "Macro-F1": f"{metrics['Macro-F1']:.4f}",
        "Positive F1": f"{metrics['Positive F1']:.4f}",
    }


def run_dataset(spec: DatasetSpec, random_state: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    features = load_json(spec.feature_config)
    feature_ids = [feature["id"] for feature in features]
    groups = load_json(spec.group_config)
    rows = read_rows(spec.feature_file)
    validate(rows, feature_ids, groups)

    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]

    all_metrics = evaluate_feature_set(train_rows, test_rows, feature_ids, spec.positive_label, random_state)
    full_rows = [metric_row(spec.display_name, "All features", feature_ids, all_metrics)]
    condensed_rows = [metric_row(spec.display_name, "All features", feature_ids, all_metrics)]

    for group_name, group_features in groups.items():
        display = f"Only {DISPLAY_NAMES.get(group_name, group_name.replace('_', ' '))}"
        metrics = evaluate_feature_set(train_rows, test_rows, group_features, spec.positive_label, random_state)
        row = metric_row(spec.display_name, display, group_features, metrics)
        full_rows.append(row)
        if group_name in spec.condensed_groups:
            condensed_rows.append(row)

    return condensed_rows, full_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="output/features/unified_human20")
    parser.add_argument("--output", default="paper_tables/table_feature_group_ablation_condensed.csv")
    parser.add_argument("--full-output", default="output/results/unified_feature_group_ablation_full.csv")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dataset", choices=["all", "sms_spam", "sst2", "ade", "disaster_tweets"], default="all")
    args = parser.parse_args()

    specs = dataset_specs(Path(args.features_dir))
    if args.dataset != "all":
        specs = [spec for spec in specs if spec.key == args.dataset]

    condensed_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []
    for spec in specs:
        condensed, full = run_dataset(spec, args.random_state)
        condensed_rows.extend(condensed)
        full_rows.extend(full)

    fieldnames = ["Dataset", "Feature group used", "#Features", "Accuracy", "Macro-F1", "Positive F1"]
    write_csv(Path(args.output), condensed_rows, fieldnames)
    write_csv(Path(args.full_output), full_rows, fieldnames)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.full_output}")


if __name__ == "__main__":
    main()
