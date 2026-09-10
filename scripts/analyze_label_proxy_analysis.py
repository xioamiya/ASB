#!/usr/bin/env python3
"""Quantify train-only label association for ASB semantic features."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif
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
            "sms_spam",
            "SMS Spam",
            features_dir / "sms_spam_5000_human20_strict.csv",
            Path("config/human20/sms_spam_human20_features.json"),
            "spam",
        ),
        DatasetSpec(
            "sst2",
            "SST-2",
            features_dir / "sst2_5000_human20_strict.csv",
            Path("config/human20/sst2_human20_features.json"),
            "positive",
        ),
        DatasetSpec(
            "ade",
            "ADE",
            features_dir / "ade_5000_human20_strict.csv",
            Path("config/human20/ade_human20_features.json"),
            "ade",
        ),
        DatasetSpec(
            "disaster_tweets",
            "Disaster Tweets",
            features_dir / "disaster_tweets_5000_human20_strict.csv",
            Path("config/human20/disaster_tweets_human20_features.json"),
            "real disaster",
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


def load_features(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate(rows: list[dict[str, str]], feature_ids: list[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = {"id", "split", "label", *feature_ids}.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def feature_matrix(rows: list[dict[str, str]], feature_ids: list[str]) -> np.ndarray:
    return np.array([[int(row[feature_id]) for feature_id in feature_ids] for row in rows], dtype=np.float32)


def y_values(rows: list[dict[str, str]], positive_label: str) -> np.ndarray:
    return np.array([1 if row["label"] == positive_label else 0 for row in rows], dtype=np.int64)


def proxy_strength(abs_gap: float, mi: float) -> str:
    if abs_gap >= 0.50 or mi >= 0.10:
        return "high"
    if abs_gap >= 0.20 or mi >= 0.02:
        return "medium"
    return "low"


def proxy_taxonomy(strength: str, direction: str, positive_label: str) -> str:
    if direction != positive_label:
        return "counter-proxy / negative evidence"
    if strength == "high":
        return "high-signal semantic proxy"
    return "mechanistic evidence"


def run_dataset(spec: DatasetSpec, random_state: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    features = load_features(spec.feature_config)
    feature_ids = [feature["id"] for feature in features]
    feature_by_id = {feature["id"]: feature for feature in features}
    rows = read_rows(spec.feature_file)
    validate(rows, feature_ids, spec.feature_file)
    train_rows = [row for row in rows if row["split"] == "train"]
    x_train = feature_matrix(train_rows, feature_ids)
    y_train = y_values(train_rows, spec.positive_label)
    mi_scores = mutual_info_classif(x_train, y_train, discrete_features=True, random_state=random_state)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    lr.fit(x_train, y_train)

    detail_rows: list[dict[str, object]] = []
    for index, feature_id in enumerate(feature_ids):
        feature_values = x_train[:, index]
        positive_yes_rate = float(feature_values[y_train == 1].mean())
        negative_yes_rate = float(feature_values[y_train == 0].mean())
        gap = positive_yes_rate - negative_yes_rate
        direction = spec.positive_label if gap >= 0 else f"not {spec.positive_label}"
        abs_gap = abs(gap)
        mi = float(mi_scores[index])
        strength = proxy_strength(abs_gap, mi)
        taxonomy = proxy_taxonomy(strength, direction, spec.positive_label)
        detail_rows.append(
            {
                "Dataset": spec.display_name,
                "Feature": feature_id,
                "Category": feature_by_id[feature_id].get("category", ""),
                "Question": feature_by_id[feature_id]["question"],
                "Positive label yes-rate": f"{positive_yes_rate:.4f}",
                "Negative label yes-rate": f"{negative_yes_rate:.4f}",
                "Absolute yes-rate gap": f"{abs_gap:.4f}",
                "Direction": direction,
                "Mutual information": f"{mi:.6f}",
                "LR coefficient": f"{float(lr.coef_[0][index]):.6f}",
                "Proxy strength": strength,
                "Proxy taxonomy": taxonomy,
            }
        )

    sorted_rows = sorted(
        detail_rows,
        key=lambda row: (-float(row["Absolute yes-rate gap"]), -float(row["Mutual information"]), str(row["Feature"])),
    )
    counts = {strength: sum(1 for row in detail_rows if row["Proxy strength"] == strength) for strength in ["high", "medium", "low"]}
    taxonomy_counts: dict[str, int] = {}
    for row in detail_rows:
        taxonomy_counts[str(row["Proxy taxonomy"])] = taxonomy_counts.get(str(row["Proxy taxonomy"]), 0) + 1
    top_examples = "; ".join(
        f"{row['Feature']} ({row['Direction']}, gap={row['Absolute yes-rate gap']})" for row in sorted_rows[:3]
    )
    summary_row = {
        "Dataset": spec.display_name,
        "High proxy features": counts["high"],
        "Medium proxy features": counts["medium"],
        "Low proxy features": counts["low"],
        "High-signal semantic proxy": taxonomy_counts.get("high-signal semantic proxy", 0),
        "Mechanistic evidence": taxonomy_counts.get("mechanistic evidence", 0),
        "Counter-proxy / negative evidence": taxonomy_counts.get("counter-proxy / negative evidence", 0),
        "Top associated features": top_examples,
    }
    return detail_rows, summary_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="output/features/unified_human20")
    parser.add_argument("--output", default="paper_tables/table_label_proxy_summary.csv")
    parser.add_argument("--details-output", default="output/results/label_proxy_feature_details.csv")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for spec in dataset_specs(Path(args.features_dir)):
        dataset_details, dataset_summary = run_dataset(spec, args.random_state)
        detail_rows.extend(dataset_details)
        summary_rows.append(dataset_summary)

    write_csv(
        Path(args.output),
        summary_rows,
        [
            "Dataset",
            "High proxy features",
            "Medium proxy features",
            "Low proxy features",
            "High-signal semantic proxy",
            "Mechanistic evidence",
            "Counter-proxy / negative evidence",
            "Top associated features",
        ],
    )
    write_csv(
        Path(args.details_output),
        detail_rows,
        [
            "Dataset",
            "Feature",
            "Category",
            "Question",
            "Positive label yes-rate",
            "Negative label yes-rate",
            "Absolute yes-rate gap",
            "Direction",
            "Mutual information",
            "LR coefficient",
            "Proxy strength",
            "Proxy taxonomy",
        ],
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.details_output}")


if __name__ == "__main__":
    main()
