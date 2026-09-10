#!/usr/bin/env python3
"""Train ASB-LR on one extractor and test it on another extractor."""

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

EXTRACTORS = [
    ("deepseek_v4_flash", "DeepSeek V4 Flash"),
    ("gpt_4_1_mini", "GPT-4.1 mini"),
    ("qwen_plus", "Qwen Plus"),
    ("gemini_2_5_flash", "Gemini 2.5 Flash"),
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


def feature_file(features_dir: Path, dataset_key: str, extractor_key: str) -> Path:
    return features_dir / f"{dataset_key}_5000_human20_{extractor_key}.csv"


def validate(rows: list[dict[str, str]], feature_ids: list[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = {"id", "split", "label", *feature_ids}.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if len(rows) != 5000 or len({row["id"] for row in rows}) != 5000:
        raise ValueError(f"{path}: expected 5000 unique rows")
    if sum(row["split"] == "train" for row in rows) != 4000:
        raise ValueError(f"{path}: expected 4000 training rows")
    if sum(row["split"] == "test" for row in rows) != 1000:
        raise ValueError(f"{path}: expected 1000 test rows")
    for row in rows:
        for feature_id in feature_ids:
            if row[feature_id] not in {"0", "1"}:
                raise ValueError(f"{path}: invalid value for {row['id']} {feature_id}: {row[feature_id]}")


def ordered_rows(rows_by_id: dict[str, dict[str, str]], ids: list[str]) -> list[dict[str, str]]:
    return [rows_by_id[row_id] for row_id in ids]


def x_values(rows: list[dict[str, str]], feature_ids: list[str]) -> list[list[int]]:
    return [[int(row[feature_id]) for feature_id in feature_ids] for row in rows]


def y_values(rows: list[dict[str, str]], positive_label: str) -> list[int]:
    return [1 if row["label"] == positive_label else 0 for row in rows]


def evaluate_dataset(
    spec: DatasetSpec,
    features_dir: Path,
    random_state: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    feature_ids = load_feature_ids(spec.feature_config)
    extractor_rows: dict[str, dict[str, dict[str, str]]] = {}
    canonical_metadata: dict[str, tuple[str, str]] | None = None

    for extractor_key, _ in EXTRACTORS:
        path = feature_file(features_dir, spec.key, extractor_key)
        rows = read_rows(path)
        validate(rows, feature_ids, path)
        rows_by_id = {row["id"]: row for row in rows}
        metadata = {row_id: (row["split"], row["label"]) for row_id, row in rows_by_id.items()}
        if canonical_metadata is None:
            canonical_metadata = metadata
        elif metadata != canonical_metadata:
            raise ValueError(f"{path}: ids, splits, or labels do not align with the other extractors")
        extractor_rows[extractor_key] = rows_by_id

    assert canonical_metadata is not None
    train_ids = sorted(row_id for row_id, meta in canonical_metadata.items() if meta[0] == "train")
    test_ids = sorted(row_id for row_id, meta in canonical_metadata.items() if meta[0] == "test")
    display_by_key = dict(EXTRACTORS)
    raw_results: dict[tuple[str, str], dict[str, float]] = {}

    for source_key, _ in EXTRACTORS:
        source_train = ordered_rows(extractor_rows[source_key], train_ids)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
        model.fit(x_values(source_train, feature_ids), y_values(source_train, spec.positive_label))
        for target_key, _ in EXTRACTORS:
            target_test = ordered_rows(extractor_rows[target_key], test_ids)
            y_test = y_values(target_test, spec.positive_label)
            y_pred = model.predict(x_values(target_test, feature_ids)).tolist()
            raw_results[(source_key, target_key)] = {
                "Accuracy": float(accuracy_score(y_test, y_pred)),
                "Macro-F1": float(f1_score(y_test, y_pred, average="macro")),
                "Positive F1": float(f1_score(y_test, y_pred, pos_label=1)),
            }

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for source_key, source_name in EXTRACTORS:
        source_in_domain = raw_results[(source_key, source_key)]["Macro-F1"]
        off_diagonal: list[tuple[str, float]] = []
        for target_key, target_name in EXTRACTORS:
            metrics = raw_results[(source_key, target_key)]
            target_in_domain = raw_results[(target_key, target_key)]["Macro-F1"]
            detail_rows.append(
                {
                    "Dataset": spec.display_name,
                    "Source extractor": source_name,
                    "Target extractor": target_name,
                    "Train rows": len(train_ids),
                    "Test rows": len(test_ids),
                    "Accuracy": f"{metrics['Accuracy']:.4f}",
                    "Macro-F1": f"{metrics['Macro-F1']:.4f}",
                    "Positive F1": f"{metrics['Positive F1']:.4f}",
                    "Source in-domain Macro-F1": f"{source_in_domain:.4f}",
                    "Delta vs source in-domain": f"{metrics['Macro-F1'] - source_in_domain:+.4f}",
                    "Target in-domain Macro-F1": f"{target_in_domain:.4f}",
                    "Delta vs target in-domain": f"{metrics['Macro-F1'] - target_in_domain:+.4f}",
                }
            )
            if source_key != target_key:
                off_diagonal.append((target_key, metrics["Macro-F1"]))

        mean_cross = sum(value for _, value in off_diagonal) / len(off_diagonal)
        worst_target_key, worst_value = min(off_diagonal, key=lambda item: item[1])
        best_target_key, best_value = max(off_diagonal, key=lambda item: item[1])
        summary_rows.append(
            {
                "Dataset": spec.display_name,
                "Source extractor": source_name,
                "In-domain Macro-F1": f"{source_in_domain:.4f}",
                "Mean cross-extractor Macro-F1": f"{mean_cross:.4f}",
                "Mean delta vs source in-domain": f"{mean_cross - source_in_domain:+.4f}",
                "Worst target": display_by_key[worst_target_key],
                "Worst cross-extractor Macro-F1": f"{worst_value:.4f}",
                "Worst delta vs source in-domain": f"{worst_value - source_in_domain:+.4f}",
                "Best target": display_by_key[best_target_key],
                "Best cross-extractor Macro-F1": f"{best_value:.4f}",
            }
        )
    return detail_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="output/features/unified_human20_models")
    parser.add_argument("--output", default="paper_tables/table_cross_extractor_transfer_summary.csv")
    parser.add_argument("--details-output", default="output/results/cross_extractor_transfer_details.csv")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for spec in DATASETS:
        dataset_details, dataset_summary = evaluate_dataset(spec, Path(args.features_dir), args.random_state)
        detail_rows.extend(dataset_details)
        summary_rows.extend(dataset_summary)

    write_csv(
        Path(args.output),
        summary_rows,
        [
            "Dataset",
            "Source extractor",
            "In-domain Macro-F1",
            "Mean cross-extractor Macro-F1",
            "Mean delta vs source in-domain",
            "Worst target",
            "Worst cross-extractor Macro-F1",
            "Worst delta vs source in-domain",
            "Best target",
            "Best cross-extractor Macro-F1",
        ],
    )
    write_csv(
        Path(args.details_output),
        detail_rows,
        [
            "Dataset",
            "Source extractor",
            "Target extractor",
            "Train rows",
            "Test rows",
            "Accuracy",
            "Macro-F1",
            "Positive F1",
            "Source in-domain Macro-F1",
            "Delta vs source in-domain",
            "Target in-domain Macro-F1",
            "Delta vs target in-domain",
        ],
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.details_output}")


if __name__ == "__main__":
    main()
