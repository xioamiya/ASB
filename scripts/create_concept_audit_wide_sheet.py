#!/usr/bin/env python3
"""Convert the long concept-audit sheet into per-dataset wide annotation sheets."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--long-annotations", default="output/concept_audit/concept_audit_annotations.csv")
    parser.add_argument("--output-dir", default="output/concept_audit/wide")
    args = parser.parse_args()

    rows = read_rows(Path(args.long_annotations))
    grouped: dict[tuple[str, str], OrderedDict[str, dict[str, object]]] = {}
    feature_order: dict[str, list[str]] = {}
    for row in rows:
        dataset_key = row["dataset_key"]
        grouped.setdefault((row["dataset"], dataset_key), OrderedDict())
        feature_order.setdefault(dataset_key, [])
        if row["feature_id"] not in feature_order[dataset_key]:
            feature_order[dataset_key].append(row["feature_id"])
        item = grouped[(row["dataset"], dataset_key)].setdefault(
            row["id"],
            {
                "dataset": row["dataset"],
                "dataset_key": dataset_key,
                "id": row["id"],
                "text": row["text"],
            },
        )
        item[row["feature_id"]] = row.get("human_value", "")

    for (dataset, dataset_key), items in grouped.items():
        fields = ["dataset", "dataset_key", "id", "text", *feature_order[dataset_key]]
        output_path = Path(args.output_dir) / f"{dataset_key}_concept_audit_wide.csv"
        write_csv(output_path, list(items.values()), fields)
        print(f"Wrote {dataset}: {output_path}")


if __name__ == "__main__":
    main()
