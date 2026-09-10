#!/usr/bin/env python3
"""Compute counterfactual feature-edit coverage for unified ASB LR models."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    feature_file: Path
    feature_config: Path
    positive_label: str


def dataset_specs(features_dir: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            key="sms_spam",
            display_name="SMS Spam",
            feature_file=features_dir / "sms_spam_5000_human20_strict.csv",
            feature_config=Path("config/human20/sms_spam_human20_features.json"),
            positive_label="spam",
        ),
        DatasetSpec(
            key="sst2",
            display_name="SST-2",
            feature_file=features_dir / "sst2_5000_human20_strict.csv",
            feature_config=Path("config/human20/sst2_human20_features.json"),
            positive_label="positive",
        ),
        DatasetSpec(
            key="ade",
            display_name="ADE",
            feature_file=features_dir / "ade_5000_human20_strict.csv",
            feature_config=Path("config/human20/ade_human20_features.json"),
            positive_label="ade",
        ),
        DatasetSpec(
            key="disaster_tweets",
            display_name="Disaster Tweets",
            feature_file=features_dir / "disaster_tweets_5000_human20_strict.csv",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_feature_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [feature["id"] for feature in json.load(f)]


def validate(rows: list[dict[str, str]], feature_ids: list[str]) -> None:
    if not rows:
        raise ValueError("No rows found")
    missing = {"id", "split", "label", *feature_ids}.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for row in rows:
        for feature_id in feature_ids:
            if row[feature_id] not in {"0", "1"}:
                raise ValueError(f"Invalid feature value for {row['id']} {feature_id}: {row[feature_id]}")


def y_value(row: dict[str, str], positive_label: str) -> int:
    return 1 if row["label"] == positive_label else 0


def feature_vector(row: dict[str, str], feature_ids: list[str]) -> list[int]:
    return [int(row[feature_id]) for feature_id in feature_ids]


def score(model: LogisticRegression, vector: list[int]) -> float:
    return float(model.intercept_[0]) + sum(float(coef) * value for coef, value in zip(model.coef_[0], vector))


def predict_from_score(value: float) -> int:
    return 1 if value >= 0 else 0


def find_minimal_edit_distance(
    *,
    model: LogisticRegression,
    vector: list[int],
    max_edits: int,
) -> int | None:
    original_score = score(model, vector)
    original_pred = predict_from_score(original_score)
    coefficients = [float(value) for value in model.coef_[0]]

    candidate_indices = []
    for index, (value, coefficient) in enumerate(zip(vector, coefficients)):
        edited_value = 1 - value
        delta = coefficient * (edited_value - value)
        if predict_from_score(original_score + delta) != original_pred:
            return 1
        candidate_indices.append((index, delta))

    target_direction = -1 if original_pred == 1 else 1
    candidate_indices.sort(key=lambda item: target_direction * item[1], reverse=True)
    ordered_indices = [index for index, _ in candidate_indices]
    for size in range(2, max_edits + 1):
        for combo in itertools.combinations(ordered_indices, size):
            edited = list(vector)
            for index in combo:
                edited[index] = 1 - edited[index]
            if predict_from_score(score(model, edited)) != original_pred:
                return size
    return None


def format_rate(count: int, total: int) -> str:
    if total == 0:
        return "0.0000"
    return f"{count / total:.4f}"


def run_dataset(
    spec: DatasetSpec,
    max_edits: int,
    random_state: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    feature_ids = load_feature_ids(spec.feature_config)
    rows = read_rows(spec.feature_file)
    validate(rows, feature_ids)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    model.fit(
        [feature_vector(row, feature_ids) for row in train_rows],
        [y_value(row, spec.positive_label) for row in train_rows],
    )

    detail_rows: list[dict[str, object]] = []
    edit_distances: list[int | None] = []
    for row in test_rows:
        vector = feature_vector(row, feature_ids)
        y_true = y_value(row, spec.positive_label)
        original_score = score(model, vector)
        original_pred = predict_from_score(original_score)
        if original_pred == y_true:
            continue
        edit_distance = find_minimal_edit_distance(model=model, vector=vector, max_edits=max_edits)
        edit_distances.append(edit_distance)
        detail_rows.append(
            {
                "Dataset": spec.display_name,
                "id": row["id"],
                "label": row["label"],
                "y_true": y_true,
                "original_pred": original_pred,
                "original_score": f"{original_score:.4f}",
                "minimal_edit_distance": "" if edit_distance is None else edit_distance,
                "text": row.get("text", ""),
            }
        )

    test_errors = len(edit_distances)
    covered_distances = [distance for distance in edit_distances if distance is not None]
    summary_row = {
        "Dataset": spec.display_name,
        "Model": "ASB-LR (ours)",
        "Test errors": test_errors,
        "<=1 edit": format_rate(sum(1 for distance in covered_distances if distance <= 1), test_errors),
        "<=2 edits": format_rate(sum(1 for distance in covered_distances if distance <= 2), test_errors),
        "<=3 edits": format_rate(sum(1 for distance in covered_distances if distance <= 3), test_errors),
        "Median edit distance": (
            "" if not covered_distances else f"{statistics.median(covered_distances):.1f}"
        ),
    }
    return summary_row, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="output/features/unified_human20")
    parser.add_argument("--output", default="paper_tables/table_counterfactual_feature_edit_coverage.csv")
    parser.add_argument("--details-output", default="output/results/unified_counterfactual_feature_edits.csv")
    parser.add_argument("--max-edits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dataset", choices=["all", "sms_spam", "sst2", "ade", "disaster_tweets"], default="all")
    args = parser.parse_args()

    specs = dataset_specs(Path(args.features_dir))
    if args.dataset != "all":
        specs = [spec for spec in specs if spec.key == args.dataset]

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for spec in specs:
        summary, details = run_dataset(spec, args.max_edits, args.random_state)
        summary_rows.append(summary)
        detail_rows.extend(details)

    write_csv(
        Path(args.output),
        summary_rows,
        ["Dataset", "Model", "Test errors", "<=1 edit", "<=2 edits", "<=3 edits", "Median edit distance"],
    )
    write_csv(
        Path(args.details_output),
        detail_rows,
        ["Dataset", "id", "label", "y_true", "original_pred", "original_score", "minimal_edit_distance", "text"],
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.details_output}")


if __name__ == "__main__":
    main()
