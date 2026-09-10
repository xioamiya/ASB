#!/usr/bin/env python3
"""Evaluate ASB-LR predictions after human concept-value intervention."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    feature_path: Path
    audit_wide_path: Path
    feature_config: Path
    positive_label: str


DATASETS = [
    DatasetSpec(
        key="sms_spam",
        display_name="SMS Spam",
        feature_path=Path("output/features/unified_human20/sms_spam_5000_human20_strict.csv"),
        audit_wide_path=Path("output/concept_audit/wide/sms_spam_concept_audit_wide.csv"),
        feature_config=Path("config/human20/sms_spam_human20_features.json"),
        positive_label="spam",
    ),
    DatasetSpec(
        key="sst2",
        display_name="SST-2",
        feature_path=Path("output/features/unified_human20/sst2_5000_human20_strict.csv"),
        audit_wide_path=Path("output/concept_audit/wide/sst2_concept_audit_wide.csv"),
        feature_config=Path("config/human20/sst2_human20_features.json"),
        positive_label="positive",
    ),
    DatasetSpec(
        key="ade",
        display_name="ADE",
        feature_path=Path("output/features/unified_human20/ade_5000_human20_strict.csv"),
        audit_wide_path=Path("output/concept_audit/wide/ade_concept_audit_wide.csv"),
        feature_config=Path("config/human20/ade_human20_features.json"),
        positive_label="ade",
    ),
    DatasetSpec(
        key="disaster_tweets",
        display_name="Disaster Tweets",
        feature_path=Path("output/features/unified_human20/disaster_tweets_5000_human20_strict.csv"),
        audit_wide_path=Path("output/concept_audit/wide/disaster_tweets_concept_audit_wide.csv"),
        feature_config=Path("config/human20/disaster_tweets_human20_features.json"),
        positive_label="real disaster",
    ),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_feature_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [feature["id"] for feature in json.load(f)]


def y_values(rows: list[dict[str, str]], positive_label: str) -> list[int]:
    return [1 if row["label"] == positive_label else 0 for row in rows]


def matrix_from_rows(rows: list[dict[str, str]], feature_ids: list[str]) -> np.ndarray:
    return np.array([[int(row[feature_id]) for feature_id in feature_ids] for row in rows], dtype=np.float32)


def as_label(value: int, positive_label: str, negative_label: str) -> str:
    return positive_label if value == 1 else negative_label


def negative_label(rows: list[dict[str, str]], positive_label: str) -> str:
    labels = sorted({row["label"] for row in rows})
    negatives = [label for label in labels if label != positive_label]
    if len(negatives) != 1:
        raise ValueError(f"Expected one negative label, found {labels}")
    return negatives[0]


def validate_rows(rows: list[dict[str, str]], feature_ids: list[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"id", "split", "label", "text", *feature_ids}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for row in rows:
        for feature_id in feature_ids:
            if row[feature_id] not in {"0", "1"}:
                raise ValueError(f"{path}: invalid {feature_id}={row[feature_id]!r} for {row['id']}")


def validate_audit_rows(rows: list[dict[str, str]], feature_ids: list[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"id", *feature_ids}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for row in rows:
        for feature_id in feature_ids:
            if row[feature_id] not in {"0", "1"}:
                raise ValueError(f"{path}: invalid human value {feature_id}={row[feature_id]!r} for {row['id']}")


def metric_summary(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=1)),
    }


def run_dataset(spec: DatasetSpec, random_state: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    feature_ids = load_feature_ids(spec.feature_config)
    feature_rows = read_rows(spec.feature_path)
    audit_rows = read_rows(spec.audit_wide_path)
    validate_rows(feature_rows, feature_ids, spec.feature_path)
    validate_audit_rows(audit_rows, feature_ids, spec.audit_wide_path)

    train_rows = [row for row in feature_rows if row["split"] == "train"]
    feature_by_id = {row["id"]: row for row in feature_rows}
    neg_label = negative_label(feature_rows, spec.positive_label)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    model.fit(matrix_from_rows(train_rows, feature_ids), y_values(train_rows, spec.positive_label))

    y_true: list[int] = []
    original_pred: list[int] = []
    human_pred: list[int] = []
    changed_predictions = 0
    fixed_errors = 0
    introduced_errors = 0
    total_feature_cells = 0
    disagreed_feature_cells = 0
    example_rows: list[dict[str, object]] = []

    for audit_row in audit_rows:
        item_id = audit_row["id"]
        feature_row = feature_by_id.get(item_id)
        if feature_row is None:
            raise ValueError(f"{spec.audit_wide_path}: audit id not found in feature matrix: {item_id}")
        if feature_row["split"] != "test":
            raise ValueError(f"{spec.audit_wide_path}: audit id is not from test split: {item_id}")

        llm_values = [int(feature_row[feature_id]) for feature_id in feature_ids]
        human_values = [int(audit_row[feature_id]) for feature_id in feature_ids]
        row_true = 1 if feature_row["label"] == spec.positive_label else 0
        row_original = int(model.predict(np.array([llm_values], dtype=np.float32))[0])
        row_human = int(model.predict(np.array([human_values], dtype=np.float32))[0])
        feature_changes = sum(1 for llm, human in zip(llm_values, human_values) if llm != human)

        y_true.append(row_true)
        original_pred.append(row_original)
        human_pred.append(row_human)
        changed_predictions += int(row_original != row_human)
        fixed_errors += int(row_original != row_true and row_human == row_true)
        introduced_errors += int(row_original == row_true and row_human != row_true)
        total_feature_cells += len(feature_ids)
        disagreed_feature_cells += feature_changes

        if row_original != row_human or row_original != row_true:
            example_rows.append(
                {
                    "Dataset": spec.display_name,
                    "id": item_id,
                    "label": feature_row["label"],
                    "original_pred": as_label(row_original, spec.positive_label, neg_label),
                    "human_corrected_pred": as_label(row_human, spec.positive_label, neg_label),
                    "original_correct": int(row_original == row_true),
                    "human_corrected_correct": int(row_human == row_true),
                    "changed_feature_count": feature_changes,
                    "text": feature_row.get("text", ""),
                }
            )

    original_metrics = metric_summary(y_true, original_pred)
    human_metrics = metric_summary(y_true, human_pred)
    summary = {
        "Dataset": spec.display_name,
        "Audit examples": len(audit_rows),
        "Feature cells": total_feature_cells,
        "Feature disagreement rate": f"{disagreed_feature_cells / total_feature_cells:.4f}",
        "Original accuracy": f"{original_metrics['accuracy']:.4f}",
        "Human-corrected accuracy": f"{human_metrics['accuracy']:.4f}",
        "Delta accuracy": f"{human_metrics['accuracy'] - original_metrics['accuracy']:.4f}",
        "Original Macro-F1": f"{original_metrics['macro_f1']:.4f}",
        "Human-corrected Macro-F1": f"{human_metrics['macro_f1']:.4f}",
        "Delta Macro-F1": f"{human_metrics['macro_f1'] - original_metrics['macro_f1']:.4f}",
        "Changed predictions": changed_predictions,
        "Errors fixed": fixed_errors,
        "Errors introduced": introduced_errors,
        "Net fixed errors": fixed_errors - introduced_errors,
    }
    return summary, example_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/results/human_concept_intervention_by_dataset.csv")
    parser.add_argument("--examples-output", default="output/results/human_concept_intervention_examples.csv")
    parser.add_argument("--paper-table", default="paper_tables/table_human_concept_intervention.csv")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    summary_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    for spec in DATASETS:
        summary, examples = run_dataset(spec, args.random_state)
        summary_rows.append(summary)
        example_rows.extend(examples)

    summary_fields = [
        "Dataset",
        "Audit examples",
        "Feature cells",
        "Feature disagreement rate",
        "Original accuracy",
        "Human-corrected accuracy",
        "Delta accuracy",
        "Original Macro-F1",
        "Human-corrected Macro-F1",
        "Delta Macro-F1",
        "Changed predictions",
        "Errors fixed",
        "Errors introduced",
        "Net fixed errors",
    ]
    example_fields = [
        "Dataset",
        "id",
        "label",
        "original_pred",
        "human_corrected_pred",
        "original_correct",
        "human_corrected_correct",
        "changed_feature_count",
        "text",
    ]
    write_csv(Path(args.output), summary_rows, summary_fields)
    write_csv(Path(args.examples_output), example_rows, example_fields)
    write_csv(Path(args.paper_table), summary_rows, summary_fields)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.examples_output}")
    print(f"Wrote {args.paper_table}")


if __name__ == "__main__":
    main()
