#!/usr/bin/env python3
"""Copy filled per-dataset wide concept-audit values back into the long sheet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


VALID_VALUES = {"", "0", "1", "U", "u"}


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
    parser.add_argument("--long-template", default="output/concept_audit/concept_audit_annotations.csv")
    parser.add_argument("--wide-dir", default="output/concept_audit/wide")
    parser.add_argument("--output", default="output/concept_audit/concept_audit_annotations.csv")
    args = parser.parse_args()

    long_rows = read_rows(Path(args.long_template))
    values: dict[tuple[str, str, str], str] = {}
    for wide_path in sorted(Path(args.wide_dir).glob("*_concept_audit_wide.csv")):
        for row in read_rows(wide_path):
            dataset_key = row["dataset_key"]
            item_id = row["id"]
            for feature_id, raw_value in row.items():
                if feature_id in {"dataset", "dataset_key", "id", "text"}:
                    continue
                value = raw_value.strip()
                if value not in VALID_VALUES:
                    raise ValueError(f"{wide_path}: invalid value for {item_id} {feature_id}: {raw_value!r}")
                if value:
                    values[(dataset_key, item_id, feature_id)] = value.upper()

    updated = 0
    for row in long_rows:
        key = (row["dataset_key"], row["id"], row["feature_id"])
        if key in values:
            row["human_value"] = values[key]
            updated += 1

    write_csv(
        Path(args.output),
        long_rows,
        ["dataset", "dataset_key", "id", "text", "feature_id", "feature_category", "feature_question", "human_value", "notes"],
    )
    print(f"Updated {updated} filled audit values in {args.output}")


if __name__ == "__main__":
    main()
