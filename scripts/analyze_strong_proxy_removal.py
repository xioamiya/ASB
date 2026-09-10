#!/usr/bin/env python3
"""Evaluate ASB-LR after removing train-only high-strength label proxies."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


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
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate row ids")
    for row in rows:
        for feature_id in feature_ids:
            if row[feature_id] not in {"0", "1"}:
                raise ValueError(f"{path}: invalid value for {row['id']} {feature_id}: {row[feature_id]}")


def x_values(rows: list[dict[str, str]], feature_ids: list[str]) -> np.ndarray:
    return np.asarray([[int(row[feature_id]) for feature_id in feature_ids] for row in rows], dtype=np.float32)


def y_values(rows: list[dict[str, str]], positive_label: str) -> np.ndarray:
    return np.asarray([1 if row["label"] == positive_label else 0 for row in rows], dtype=np.int64)


def evaluate(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_ids: list[str],
    positive_label: str,
    random_state: int,
) -> dict[str, float]:
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    model.fit(x_values(train_rows, feature_ids), y_values(train_rows, positive_label))
    y_test = y_values(test_rows, positive_label)
    y_pred = model.predict(x_values(test_rows, feature_ids))
    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Macro-F1": float(f1_score(y_test, y_pred, average="macro")),
        "Positive F1": float(f1_score(y_test, y_pred, pos_label=1)),
    }


def is_high_proxy(abs_gap: float, mutual_information: float) -> bool:
    """Use the same pre-specified rule as analyze_label_proxy_analysis.py."""
    return abs_gap >= 0.50 or mutual_information >= 0.10


def run_dataset(
    spec: DatasetSpec,
    random_state: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    features = load_features(spec.feature_config)
    feature_ids = [feature["id"] for feature in features]
    feature_by_id = {feature["id"]: feature for feature in features}
    rows = read_rows(spec.feature_file)
    validate(rows, feature_ids, spec.feature_file)
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(train_rows) != 4000 or len(test_rows) != 1000:
        raise ValueError(f"{spec.feature_file}: expected 4000 train and 1000 test rows")

    x_train = x_values(train_rows, feature_ids)
    y_train = y_values(train_rows, spec.positive_label)
    mi_scores = mutual_info_classif(
        x_train,
        y_train,
        discrete_features=True,
        random_state=random_state,
    )

    detail_rows: list[dict[str, object]] = []
    removed: list[str] = []
    for index, feature_id in enumerate(feature_ids):
        values = x_train[:, index]
        positive_yes_rate = float(values[y_train == 1].mean())
        negative_yes_rate = float(values[y_train == 0].mean())
        gap = positive_yes_rate - negative_yes_rate
        abs_gap = abs(gap)
        mi = float(mi_scores[index])
        if not is_high_proxy(abs_gap, mi):
            continue
        removed.append(feature_id)
        detail_rows.append(
            {
                "Dataset": spec.display_name,
                "Feature": feature_id,
                "Category": feature_by_id[feature_id].get("category", ""),
                "Question": feature_by_id[feature_id]["question"],
                "Direction": spec.positive_label if gap >= 0 else f"not {spec.positive_label}",
                "Positive label yes-rate": f"{positive_yes_rate:.4f}",
                "Negative label yes-rate": f"{negative_yes_rate:.4f}",
                "Absolute yes-rate gap": f"{abs_gap:.4f}",
                "Mutual information": f"{mi:.6f}",
            }
        )

    retained = [feature_id for feature_id in feature_ids if feature_id not in set(removed)]
    if not retained:
        raise ValueError(f"{spec.display_name}: proxy removal left no features")
    baseline = evaluate(train_rows, test_rows, feature_ids, spec.positive_label, random_state)
    reduced = evaluate(train_rows, test_rows, retained, spec.positive_label, random_state)
    summary = {
        "Dataset": spec.display_name,
        "Removal rule": "train abs yes-rate gap >= 0.50 or train MI >= 0.10",
        "Removed features": len(removed),
        "Remaining features": len(retained),
        "Baseline Accuracy": f"{baseline['Accuracy']:.4f}",
        "After removal Accuracy": f"{reduced['Accuracy']:.4f}",
        "Delta Accuracy": f"{reduced['Accuracy'] - baseline['Accuracy']:+.4f}",
        "Baseline Macro-F1": f"{baseline['Macro-F1']:.4f}",
        "After removal Macro-F1": f"{reduced['Macro-F1']:.4f}",
        "Delta Macro-F1": f"{reduced['Macro-F1'] - baseline['Macro-F1']:+.4f}",
        "Baseline Positive F1": f"{baseline['Positive F1']:.4f}",
        "After removal Positive F1": f"{reduced['Positive F1']:.4f}",
        "Delta Positive F1": f"{reduced['Positive F1'] - baseline['Positive F1']:+.4f}",
        "Removed feature ids": "; ".join(removed),
    }
    return summary, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="output/features/unified_human20")
    parser.add_argument("--output", default="paper_tables/table_strong_proxy_removal.csv")
    parser.add_argument("--details-output", default="output/results/strong_proxy_removal_details.csv")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for spec in dataset_specs(Path(args.features_dir)):
        summary, details = run_dataset(spec, args.random_state)
        summary_rows.append(summary)
        detail_rows.extend(details)

    write_csv(
        Path(args.output),
        summary_rows,
        [
            "Dataset",
            "Removal rule",
            "Removed features",
            "Remaining features",
            "Baseline Accuracy",
            "After removal Accuracy",
            "Delta Accuracy",
            "Baseline Macro-F1",
            "After removal Macro-F1",
            "Delta Macro-F1",
            "Baseline Positive F1",
            "After removal Positive F1",
            "Delta Positive F1",
            "Removed feature ids",
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
            "Direction",
            "Positive label yes-rate",
            "Negative label yes-rate",
            "Absolute yes-rate gap",
            "Mutual information",
        ],
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.details_output}")


if __name__ == "__main__":
    main()
