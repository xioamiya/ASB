#!/usr/bin/env python3
"""Analyze ASB schema sensitivity using original and alternative feature matrices."""

from __future__ import annotations

import argparse
import csv
import json
import re
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
    original_features: Path
    original_config: Path
    alternative_features: Path
    alternative_config: Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_features(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def feature_ids(path: Path) -> list[str]:
    return [feature["id"] for feature in load_features(path)]


def validate_rows(rows: list[dict[str, str]], ids: list[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found: {path}")
    required = {"id", "split", "label", *ids}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for row in rows:
        for feature_id in ids:
            if row[feature_id] not in {"0", "1"}:
                raise ValueError(f"{path}: invalid value for {row['id']} {feature_id}: {row[feature_id]!r}")


def y_values(rows: list[dict[str, str]], positive_label: str) -> list[int]:
    return [1 if row["label"] == positive_label else 0 for row in rows]


def matrix(rows: list[dict[str, str]], ids: list[str]) -> np.ndarray:
    return np.asarray([[int(row[feature_id]) for feature_id in ids] for row in rows], dtype=np.float32)


def metric_values(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro")),
        "Positive F1": float(f1_score(y_true, y_pred, pos_label=1)),
    }


def evaluate_schema(
    *,
    rows: list[dict[str, str]],
    ids: list[str],
    positive_label: str,
    seed: int,
) -> dict[str, float]:
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    x_train = matrix(train_rows, ids)
    x_test = matrix(test_rows, ids)
    y_train = y_values(train_rows, positive_label)
    y_test = y_values(test_rows, positive_label)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    model.fit(x_train, y_train)
    return metric_values(y_test, model.predict(x_test).tolist())


def summarize_metrics(per_seed: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in per_seed:
        grouped.setdefault((str(row["Dataset"]), str(row["Schema"])), []).append(row)

    original_macro = {
        dataset: float(np.mean([float(row["Macro-F1"]) for row in rows]))
        for (dataset, schema), rows in grouped.items()
        if schema == "original"
    }
    out = []
    for dataset in ["SMS Spam", "SST-2", "ADE", "Disaster Tweets"]:
        for schema in ["original", "alternative"]:
            rows = grouped.get((dataset, schema), [])
            if not rows:
                continue
            accuracy = float(np.mean([float(row["Accuracy"]) for row in rows]))
            macro = float(np.mean([float(row["Macro-F1"]) for row in rows]))
            positive = float(np.mean([float(row["Positive F1"]) for row in rows]))
            out.append(
                {
                    "Dataset": dataset,
                    "Schema": schema,
                    "Accuracy": f"{accuracy:.4f}",
                    "Macro-F1": f"{macro:.4f}",
                    "Positive F1": f"{positive:.4f}",
                    "Delta Macro-F1 vs original": f"{macro - original_macro[dataset]:+.4f}",
                }
            )
    return out


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in {"does", "the", "a", "an", "or", "and", "to", "of", "in", "is", "it", "any"}
    }


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def schema_overlap(
    *,
    dataset: str,
    original_config: Path,
    alternative_config: Path,
    original_rows: list[dict[str, str]],
    alternative_rows: list[dict[str, str]],
) -> dict[str, object]:
    original = load_features(original_config)
    alternative = load_features(alternative_config)
    original_ids = [feature["id"] for feature in original]
    alternative_ids = [feature["id"] for feature in alternative]

    original_tokens = [tokenize(feature["question"]) for feature in original]
    alt_tokens = [tokenize(feature["question"]) for feature in alternative]
    max_jaccards = [max(jaccard(alt, orig) for orig in original_tokens) for alt in alt_tokens]

    original_by_id = {row["id"]: row for row in original_rows}
    paired_original = []
    paired_alternative = []
    for row in alternative_rows:
        if row["id"] in original_by_id:
            paired_original.append(original_by_id[row["id"]])
            paired_alternative.append(row)
    if not paired_original:
        raise ValueError(f"No paired rows for overlap: {dataset}")

    x_original = matrix(paired_original, original_ids)
    x_alternative = matrix(paired_alternative, alternative_ids)
    corr = np.zeros((len(alternative_ids), len(original_ids)), dtype=np.float32)
    for alt_idx in range(len(alternative_ids)):
        for orig_idx in range(len(original_ids)):
            corr[alt_idx, orig_idx] = abs(safe_corr(x_alternative[:, alt_idx], x_original[:, orig_idx]))
    max_corr = corr.max(axis=1)

    return {
        "Dataset": dataset,
        "Mean max lexical Jaccard": f"{float(np.mean(max_jaccards)):.4f}",
        "Mean max abs feature correlation": f"{float(np.mean(max_corr)):.4f}",
        "Alternative features with max corr > 0.5": int(np.sum(max_corr > 0.5)),
        "Paired rows": len(paired_original),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alt-features-dir", default="output/features/schema_sensitivity")
    parser.add_argument("--alt-config-dir", default="config/schema_sensitivity")
    parser.add_argument("--original-features-dir", default="output/features/unified_human20")
    parser.add_argument("--output", default="paper_tables/table_schema_sensitivity.csv")
    parser.add_argument("--per-seed-output", default="output/results/schema_sensitivity_by_seed.csv")
    parser.add_argument("--overlap-output", default="paper_tables/table_schema_overlap.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args()

    alt_features_dir = Path(args.alt_features_dir)
    alt_config_dir = Path(args.alt_config_dir)
    original_features_dir = Path(args.original_features_dir)
    specs = [
        DatasetSpec(
            key="sms_spam",
            display_name="SMS Spam",
            data_path=Path("data/sms_spam_5000.csv"),
            positive_label="spam",
            original_features=original_features_dir / "sms_spam_5000_human20_strict.csv",
            original_config=Path("config/human20/sms_spam_human20_features.json"),
            alternative_features=alt_features_dir / "sms_spam_5000_alt20_strict.csv",
            alternative_config=alt_config_dir / "sms_spam_alt20_features.json",
        ),
        DatasetSpec(
            key="sst2",
            display_name="SST-2",
            data_path=Path("data/sst2_5000.csv"),
            positive_label="positive",
            original_features=original_features_dir / "sst2_5000_human20_strict.csv",
            original_config=Path("config/human20/sst2_human20_features.json"),
            alternative_features=alt_features_dir / "sst2_5000_alt20_strict.csv",
            alternative_config=alt_config_dir / "sst2_alt20_features.json",
        ),
        DatasetSpec(
            key="ade",
            display_name="ADE",
            data_path=Path("data/ade_5000.csv"),
            positive_label="ade",
            original_features=original_features_dir / "ade_5000_human20_strict.csv",
            original_config=Path("config/human20/ade_human20_features.json"),
            alternative_features=alt_features_dir / "ade_5000_alt20_strict.csv",
            alternative_config=alt_config_dir / "ade_alt20_features.json",
        ),
        DatasetSpec(
            key="disaster_tweets",
            display_name="Disaster Tweets",
            data_path=Path("data/disaster_5000.csv"),
            positive_label="real disaster",
            original_features=original_features_dir / "disaster_tweets_5000_human20_strict.csv",
            original_config=Path("config/human20/disaster_tweets_human20_features.json"),
            alternative_features=alt_features_dir / "disaster_tweets_5000_alt20_strict.csv",
            alternative_config=alt_config_dir / "disaster_tweets_alt20_features.json",
        ),
    ]

    per_seed_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    for spec in specs:
        required_paths = [
            spec.original_features,
            spec.original_config,
            spec.alternative_features,
            spec.alternative_config,
        ]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            print(f"Skipping {spec.display_name}: missing {missing}")
            continue
        original_ids = feature_ids(spec.original_config)
        alternative_ids = feature_ids(spec.alternative_config)
        original_rows = read_rows(spec.original_features)
        alternative_rows = read_rows(spec.alternative_features)
        validate_rows(original_rows, original_ids, spec.original_features)
        validate_rows(alternative_rows, alternative_ids, spec.alternative_features)
        for schema_name, rows, ids in [
            ("original", original_rows, original_ids),
            ("alternative", alternative_rows, alternative_ids),
        ]:
            for seed in args.seeds:
                metrics = evaluate_schema(
                    rows=rows,
                    ids=ids,
                    positive_label=spec.positive_label,
                    seed=seed,
                )
                per_seed_rows.append(
                    {
                        "Dataset": spec.display_name,
                        "Schema": schema_name,
                        "seed": seed,
                        **metrics,
                    }
                )
        overlap_rows.append(
            schema_overlap(
                dataset=spec.display_name,
                original_config=spec.original_config,
                alternative_config=spec.alternative_config,
                original_rows=original_rows,
                alternative_rows=alternative_rows,
            )
        )

    write_csv(
        Path(args.per_seed_output),
        per_seed_rows,
        ["Dataset", "Schema", "seed", "Accuracy", "Macro-F1", "Positive F1"],
    )
    write_csv(
        Path(args.output),
        summarize_metrics(per_seed_rows),
        ["Dataset", "Schema", "Accuracy", "Macro-F1", "Positive F1", "Delta Macro-F1 vs original"],
    )
    write_csv(
        Path(args.overlap_output),
        overlap_rows,
        [
            "Dataset",
            "Mean max lexical Jaccard",
            "Mean max abs feature correlation",
            "Alternative features with max corr > 0.5",
            "Paired rows",
        ],
    )
    print(f"Wrote {args.per_seed_output}")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.overlap_output}")


if __name__ == "__main__":
    main()
