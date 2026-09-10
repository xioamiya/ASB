#!/usr/bin/env python3
"""Resample 20-concept panels from the fixed original+alternative ASB pool.

This is a diagnostic over already materialized feature matrices.  It does not
generate or select a new schema using held-out performance: every sampled
panel is reported, and the test split is used only to summarize its operating
point.
"""

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
    positive_label: str
    original_features: Path
    original_config: Path
    alternative_features: Path
    alternative_config: Path


def dataset_specs(original_dir: Path, alternative_dir: Path, config_dir: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            "sms_spam", "SMS Spam", "spam",
            original_dir / "sms_spam_5000_human20_strict.csv",
            Path("config/human20/sms_spam_human20_features.json"),
            alternative_dir / "sms_spam_5000_alt20_strict.csv",
            config_dir / "sms_spam_alt20_features.json",
        ),
        DatasetSpec(
            "sst2", "SST-2", "positive",
            original_dir / "sst2_5000_human20_strict.csv",
            Path("config/human20/sst2_human20_features.json"),
            alternative_dir / "sst2_5000_alt20_strict.csv",
            config_dir / "sst2_alt20_features.json",
        ),
        DatasetSpec(
            "ade", "ADE", "ade",
            original_dir / "ade_5000_human20_strict.csv",
            Path("config/human20/ade_human20_features.json"),
            alternative_dir / "ade_5000_alt20_strict.csv",
            config_dir / "ade_alt20_features.json",
        ),
        DatasetSpec(
            "disaster_tweets", "Disaster Tweets", "real disaster",
            original_dir / "disaster_tweets_5000_human20_strict.csv",
            Path("config/human20/disaster_tweets_human20_features.json"),
            alternative_dir / "disaster_tweets_5000_alt20_strict.csv",
            config_dir / "disaster_tweets_alt20_features.json",
        ),
    ]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_feature_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [feature["id"] for feature in json.load(handle)]


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_and_pair(
    original_rows: list[dict[str, str]],
    alternative_rows: list[dict[str, str]],
    original_ids: list[str],
    alternative_ids: list[str],
    spec: DatasetSpec,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    if len(original_rows) != 5000 or len(alternative_rows) != 5000:
        raise ValueError(f"{spec.display_name}: expected 5000 rows in each feature matrix")
    original_by_id = {row["id"]: row for row in original_rows}
    alternative_by_id = {row["id"]: row for row in alternative_rows}
    if len(original_by_id) != 5000 or original_by_id.keys() != alternative_by_id.keys():
        raise ValueError(f"{spec.display_name}: feature matrices do not contain the same unique ids")
    for row in original_rows:
        for feature_id in original_ids:
            if row.get(feature_id) not in {"0", "1"}:
                raise ValueError(f"{spec.display_name}: invalid original value for {row['id']} {feature_id}")
    for row in alternative_rows:
        for feature_id in alternative_ids:
            if row.get(feature_id) not in {"0", "1"}:
                raise ValueError(f"{spec.display_name}: invalid alternative value for {row['id']} {feature_id}")
    pairs = []
    for row_id in sorted(original_by_id):
        left = original_by_id[row_id]
        right = alternative_by_id[row_id]
        if (left["split"], left["label"]) != (right["split"], right["label"]):
            raise ValueError(f"{spec.display_name}: metadata mismatch for {row_id}")
        pairs.append((left, right))
    return pairs


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro")),
        float(f1_score(y_true, y_pred, pos_label=1)),
    )


def run_dataset(
    spec: DatasetSpec,
    panel_count: int,
    panel_size: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    original_ids = read_feature_ids(spec.original_config)
    alternative_ids = read_feature_ids(spec.alternative_config)
    if set(original_ids) & set(alternative_ids):
        raise ValueError(f"{spec.display_name}: original and alternative feature ids overlap")
    pooled_ids = [f"original:{item}" for item in original_ids] + [f"alternative:{item}" for item in alternative_ids]
    if panel_size >= len(pooled_ids):
        raise ValueError("Panel size must be smaller than the pooled feature count")

    pairs = validate_and_pair(
        read_rows(spec.original_features), read_rows(spec.alternative_features),
        original_ids, alternative_ids, spec,
    )
    split = np.asarray([left["split"] for left, _ in pairs])
    labels = np.asarray([1 if left["label"] == spec.positive_label else 0 for left, _ in pairs], dtype=np.int64)
    original_matrix = np.asarray(
        [[int(left[feature_id]) for feature_id in original_ids] for left, _ in pairs], dtype=np.float32
    )
    alternative_matrix = np.asarray(
        [[int(right[feature_id]) for feature_id in alternative_ids] for _, right in pairs], dtype=np.float32
    )
    pooled_matrix = np.concatenate([original_matrix, alternative_matrix], axis=1)
    train_mask = split == "train"
    test_mask = split == "test"
    if int(train_mask.sum()) != 4000 or int(test_mask.sum()) != 1000:
        raise ValueError(f"{spec.display_name}: expected a 4000/1000 train/test split")

    baseline_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    baseline_model.fit(original_matrix[train_mask], labels[train_mask])
    _, baseline_macro, _ = metrics(labels[test_mask], baseline_model.predict(original_matrix[test_mask]))

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    macro_values: list[float] = []
    for panel_index in range(panel_count):
        selected = np.sort(rng.choice(len(pooled_ids), size=panel_size, replace=False))
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(pooled_matrix[train_mask][:, selected], labels[train_mask])
        accuracy, macro, positive = metrics(
            labels[test_mask], model.predict(pooled_matrix[test_mask][:, selected])
        )
        macro_values.append(macro)
        rows.append(
            {
                "Dataset": spec.display_name,
                "Panel": panel_index + 1,
                "Sampling seed": seed,
                "Original features": int(np.sum(selected < len(original_ids))),
                "Alternative features": int(np.sum(selected >= len(original_ids))),
                "Accuracy": f"{accuracy:.4f}",
                "Macro-F1": f"{macro:.4f}",
                "Positive F1": f"{positive:.4f}",
                "Delta Macro-F1 vs original": f"{macro - baseline_macro:+.4f}",
                "Feature ids": "; ".join(pooled_ids[index] for index in selected),
            }
        )

    values = np.asarray(macro_values)
    summary = {
        "Dataset": spec.display_name,
        "Panels": panel_count,
        "Pool size": len(pooled_ids),
        "Panel size": panel_size,
        "Original Macro-F1": f"{baseline_macro:.4f}",
        "Mean Macro-F1": f"{float(values.mean()):.4f}",
        "SD Macro-F1": f"{float(values.std(ddof=1)):.4f}",
        "Minimum Macro-F1": f"{float(values.min()):.4f}",
        "5th percentile": f"{float(np.quantile(values, 0.05)):.4f}",
        "Median Macro-F1": f"{float(np.median(values)):.4f}",
        "95th percentile": f"{float(np.quantile(values, 0.95)):.4f}",
        "Maximum Macro-F1": f"{float(values.max()):.4f}",
        "Panels at least original": f"{float(np.mean(values >= baseline_macro)):.2f}",
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-features-dir", default="output/features/unified_human20")
    parser.add_argument("--alternative-features-dir", default="output/features/schema_sensitivity")
    parser.add_argument("--alternative-config-dir", default="config/schema_sensitivity")
    parser.add_argument("--panels", type=int, default=100)
    parser.add_argument("--panel-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="paper_tables/table_schema_resampling_summary.csv")
    parser.add_argument("--details-output", default="output/results/schema_resampling_panels.csv")
    args = parser.parse_args()

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for offset, spec in enumerate(dataset_specs(
        Path(args.original_features_dir), Path(args.alternative_features_dir), Path(args.alternative_config_dir)
    )):
        details, summary = run_dataset(spec, args.panels, args.panel_size, args.seed + offset)
        detail_rows.extend(details)
        summary_rows.append(summary)

    write_rows(
        Path(args.details_output), detail_rows,
        ["Dataset", "Panel", "Sampling seed", "Original features", "Alternative features", "Accuracy",
         "Macro-F1", "Positive F1", "Delta Macro-F1 vs original", "Feature ids"],
    )
    write_rows(
        Path(args.output), summary_rows,
        ["Dataset", "Panels", "Pool size", "Panel size", "Original Macro-F1", "Mean Macro-F1",
         "SD Macro-F1", "Minimum Macro-F1", "5th percentile", "Median Macro-F1", "95th percentile",
         "Maximum Macro-F1", "Panels at least original"],
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.details_output}")


if __name__ == "__main__":
    main()
