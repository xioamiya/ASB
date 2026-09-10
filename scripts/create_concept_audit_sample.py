#!/usr/bin/env python3
"""Create blind human annotation sheets for the ASB concept audit."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path


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


def validate_feature_rows(rows: list[dict[str, str]], feature_ids: list[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No feature rows found in {path}")
    missing = {"id", "split", "label", "text", *feature_ids}.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def balanced_test_sample(
    rows: list[dict[str, str]],
    positive_label: str,
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    test_rows = [row for row in rows if row["split"] == "test"]
    positives = [row for row in test_rows if row["label"] == positive_label]
    negatives = [row for row in test_rows if row["label"] != positive_label]
    per_class = sample_size // 2
    positive_take = min(per_class, len(positives))
    negative_take = min(sample_size - positive_take, len(negatives))
    if positive_take + negative_take < sample_size:
        remaining = [row for row in test_rows if row not in positives[:positive_take] + negatives[:negative_take]]
        extra = min(sample_size - positive_take - negative_take, len(remaining))
    else:
        extra = 0

    sampled = rng.sample(positives, positive_take) + rng.sample(negatives, negative_take)
    if extra:
        sampled.extend(rng.sample([row for row in test_rows if row not in sampled], extra))
    rng.shuffle(sampled)
    return sampled


def build_rows(
    spec: DatasetSpec,
    sample_rows: list[dict[str, str]],
    features: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    blind_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    for row in sample_rows:
        for feature in features:
            feature_id = feature["id"]
            blind_rows.append(
                {
                    "dataset": spec.display_name,
                    "dataset_key": spec.key,
                    "id": row["id"],
                    "text": row["text"],
                    "feature_id": feature_id,
                    "feature_category": feature.get("category", ""),
                    "feature_question": feature["question"],
                    "human_value": "",
                    "notes": "",
                }
            )
            key_rows.append(
                {
                    "dataset": spec.display_name,
                    "dataset_key": spec.key,
                    "id": row["id"],
                    "label": row["label"],
                    "feature_id": feature_id,
                    "llm_value": row[feature_id],
                }
            )
    return blind_rows, key_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="output/features/unified_human20")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--annotation-output", default="output/concept_audit/concept_audit_annotations.csv")
    parser.add_argument("--key-output", default="output/concept_audit/concept_audit_key.csv")
    parser.add_argument("--feature-guide-output", default="output/concept_audit/concept_audit_feature_guide.csv")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    annotation_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    guide_rows: list[dict[str, object]] = []
    for spec in dataset_specs(Path(args.features_dir)):
        features = load_features(spec.feature_config)
        feature_ids = [feature["id"] for feature in features]
        rows = read_rows(spec.feature_file)
        validate_feature_rows(rows, feature_ids, spec.feature_file)
        sampled = balanced_test_sample(rows, spec.positive_label, args.sample_size, rng)
        dataset_annotation_rows, dataset_key_rows = build_rows(spec, sampled, features)
        annotation_rows.extend(dataset_annotation_rows)
        key_rows.extend(dataset_key_rows)
        for feature in features:
            guide_rows.append(
                {
                    "dataset": spec.display_name,
                    "dataset_key": spec.key,
                    "feature_id": feature["id"],
                    "feature_category": feature.get("category", ""),
                    "feature_question": feature["question"],
                }
            )

    write_csv(
        Path(args.annotation_output),
        annotation_rows,
        [
            "dataset",
            "dataset_key",
            "id",
            "text",
            "feature_id",
            "feature_category",
            "feature_question",
            "human_value",
            "notes",
        ],
    )
    write_csv(Path(args.key_output), key_rows, ["dataset", "dataset_key", "id", "label", "feature_id", "llm_value"])
    write_csv(
        Path(args.feature_guide_output),
        guide_rows,
        ["dataset", "dataset_key", "feature_id", "feature_category", "feature_question"],
    )
    print(f"Wrote blind annotation sheet: {args.annotation_output}")
    print(f"Wrote key file for analysis: {args.key_output}")
    print(f"Wrote feature guide: {args.feature_guide_output}")


if __name__ == "__main__":
    main()
