#!/usr/bin/env python3
"""Analyze human-vs-LLM agreement for the ASB concept audit."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score


VALID_HUMAN_VALUES = {"0", "1", "U", "u"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None) -> str:
    if value is None or np.isnan(value):
        return ""
    return f"{value:.4f}"


def positive_agreement(human: list[int], llm: list[int]) -> float | None:
    tp = sum(1 for h, l in zip(human, llm) if h == 1 and l == 1)
    fp = sum(1 for h, l in zip(human, llm) if h == 0 and l == 1)
    fn = sum(1 for h, l in zip(human, llm) if h == 1 and l == 0)
    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return None
    return 2 * tp / denominator


def kappa(human: list[int], llm: list[int]) -> float | None:
    if len(set(human)) < 2 and len(set(llm)) < 2:
        return None
    return float(cohen_kappa_score(human, llm))


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def normalize_human_value(raw_value: str, row_id: str, feature_id: str) -> str:
    value = raw_value.strip()
    if value == "":
        return ""
    if value not in VALID_HUMAN_VALUES:
        raise ValueError(f"Invalid human_value for {row_id} {feature_id}: {raw_value!r}; use 0, 1, or U")
    return value.upper()


def load_joined_rows(annotation_path: Path, key_path: Path) -> list[dict[str, str]]:
    annotation_rows = read_rows(annotation_path)
    key_rows = read_rows(key_path)
    key_by_item = {(row["dataset_key"], row["id"], row["feature_id"]): row for row in key_rows}

    joined: list[dict[str, str]] = []
    for row in annotation_rows:
        item_key = (row["dataset_key"], row["id"], row["feature_id"])
        if item_key not in key_by_item:
            raise ValueError(f"Missing key row for {item_key}")
        human_value = normalize_human_value(row.get("human_value", ""), row["id"], row["feature_id"])
        if human_value == "":
            continue
        key_row = key_by_item[item_key]
        joined.append({**row, **key_row, "human_value": human_value})
    if not joined:
        raise ValueError(
            f"No filled annotations found in {annotation_path}. Fill human_value with 0, 1, or U before analysis."
        )
    return joined


def metric_row(rows: list[dict[str, str]], dataset: str, feature_id: str | None = None) -> dict[str, object]:
    valid = [row for row in rows if row["human_value"] in {"0", "1"}]
    unclear = [row for row in rows if row["human_value"] == "U"]
    human = [int(row["human_value"]) for row in valid]
    llm = [int(row["llm_value"]) for row in valid]
    agreements = sum(1 for h, l in zip(human, llm) if h == l)
    fp = sum(1 for h, l in zip(human, llm) if h == 0 and l == 1)
    fn = sum(1 for h, l in zip(human, llm) if h == 1 and l == 0)
    human_negative = sum(1 for h in human if h == 0)
    human_positive = sum(1 for h in human if h == 1)

    return {
        "Dataset": dataset,
        "Feature": feature_id or "ALL",
        "Annotated entries": len(rows),
        "Scored entries": len(valid),
        "Unclear entries": len(unclear),
        "Unclear rate": fmt(rate(len(unclear), len(rows))),
        "Agreement": fmt(rate(agreements, len(valid))),
        "Cohen kappa": fmt(kappa(human, llm)),
        "Positive agreement": fmt(positive_agreement(human, llm)),
        "LLM false-positive rate": fmt(rate(fp, human_negative)),
        "LLM false-negative rate": fmt(rate(fn, human_positive)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="output/concept_audit/concept_audit_annotations.csv")
    parser.add_argument("--key", default="output/concept_audit/concept_audit_key.csv")
    parser.add_argument("--summary-output", default="paper_tables/table_concept_audit_summary.csv")
    parser.add_argument("--feature-output", default="output/results/concept_audit_feature_level.csv")
    args = parser.parse_args()

    rows = load_joined_rows(Path(args.annotations), Path(args.key))
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_feature: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset = row["dataset"]
        by_dataset[dataset].append(row)
        by_feature[(dataset, row["feature_id"])].append(row)

    summary_rows = [metric_row(by_dataset[dataset], dataset) for dataset in sorted(by_dataset)]
    feature_rows = [
        metric_row(feature_rows, dataset, feature_id)
        for (dataset, feature_id), feature_rows in sorted(by_feature.items())
    ]
    write_csv(
        Path(args.summary_output),
        summary_rows,
        [
            "Dataset",
            "Feature",
            "Annotated entries",
            "Scored entries",
            "Unclear entries",
            "Unclear rate",
            "Agreement",
            "Cohen kappa",
            "Positive agreement",
            "LLM false-positive rate",
            "LLM false-negative rate",
        ],
    )
    write_csv(
        Path(args.feature_output),
        feature_rows,
        [
            "Dataset",
            "Feature",
            "Annotated entries",
            "Scored entries",
            "Unclear entries",
            "Unclear rate",
            "Agreement",
            "Cohen kappa",
            "Positive agreement",
            "LLM false-positive rate",
            "LLM false-negative rate",
        ],
    )
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.feature_output}")


if __name__ == "__main__":
    main()
