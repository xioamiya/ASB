#!/usr/bin/env python3
"""Prepare unified 5000-row stratified datasets and the construction summary table."""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, DownloadConfig, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    task: str
    positive_label: str
    positive_label_display: str
    output_path: Path


SPECS = [
    DatasetSpec(
        key="sms_spam",
        display_name="SMS Spam",
        task="spam detection",
        positive_label="spam",
        positive_label_display="spam",
        output_path=Path("data/sms_spam_5000.csv"),
    ),
    DatasetSpec(
        key="sst2",
        display_name="SST-2",
        task="sentiment",
        positive_label="positive",
        positive_label_display="positive",
        output_path=Path("data/sst2_5000.csv"),
    ),
    DatasetSpec(
        key="ade",
        display_name="ADE",
        task="adverse drug event detection",
        positive_label="ade",
        positive_label_display="ADE",
        output_path=Path("data/ade_5000.csv"),
    ),
    DatasetSpec(
        key="disaster",
        display_name="Disaster Tweets",
        task="disaster detection",
        positive_label="real disaster",
        positive_label_display="real disaster",
        output_path=Path("data/disaster_5000.csv"),
    ),
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def deduplicate_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        text = row["text"].strip()
        if not text:
            continue
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        output.append({**row, "text": text})
    return output


def proportional_allocations(label_counts: Counter[str], total: int) -> dict[str, int]:
    available_total = sum(label_counts.values())
    if total > available_total:
        raise ValueError(f"Requested {total} rows, but only {available_total} cleaned rows are available")

    raw = {label: total * count / available_total for label, count in label_counts.items()}
    allocations = {label: int(value) for label, value in raw.items()}
    remainder = total - sum(allocations.values())
    labels_by_fraction = sorted(
        label_counts,
        key=lambda label: (raw[label] - allocations[label], label),
        reverse=True,
    )
    for label in labels_by_fraction:
        if remainder <= 0:
            break
        allocations[label] += 1
        remainder -= 1

    for label, allocation in allocations.items():
        if allocation > label_counts[label]:
            raise ValueError(f"Cannot sample {allocation} rows for {label}; only {label_counts[label]} available")
    return allocations


def stratified_sample(rows: list[dict[str, str]], total: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    allocations = proportional_allocations(Counter(row["label"] for row in rows), total)
    sampled: list[dict[str, str]] = []
    for label in sorted(allocations):
        sampled.extend(rng.sample(by_label[label], allocations[label]))
    rng.shuffle(sampled)
    return sampled


def add_stratified_split(rows: list[dict[str, str]], train_size: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    label_counts = Counter(row["label"] for row in rows)
    train_allocations = proportional_allocations(label_counts, train_size)

    output: list[dict[str, str]] = []
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    for label in sorted(by_label):
        label_rows = list(by_label[label])
        rng.shuffle(label_rows)
        cutoff = train_allocations[label]
        output.extend({**row, "split": "train"} for row in label_rows[:cutoff])
        output.extend({**row, "split": "test"} for row in label_rows[cutoff:])

    rng.shuffle(output)
    return output


def write_dataset(rows: list[dict[str, str]], spec: DatasetSpec) -> None:
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "source_id", "split", "label", "text"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            writer.writerow(
                {
                    "id": f"{spec.key}_{index:05d}",
                    "source_id": row["source_id"],
                    "split": row["split"],
                    "label": row["label"],
                    "text": row["text"],
                }
            )


def load_sms_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="latin-1", newline="") as f:
        for index, row in enumerate(csv.DictReader(f)):
            label = (row.get("v1") or "").strip()
            text = (row.get("v2") or "").strip()
            if label in {"ham", "spam"}:
                rows.append({"source_id": str(index), "label": label, "text": text})
    return rows


def load_sst2_rows(local_files_only: bool) -> list[dict[str, str]]:
    cache_dir = (
        Path.home()
        / ".cache/huggingface/datasets/glue/sst2/0.0.0"
        / "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
    )
    if local_files_only and cache_dir.exists():
        dataset = {
            "train": Dataset.from_file(str(cache_dir / "glue-train.arrow")),
            "validation": Dataset.from_file(str(cache_dir / "glue-validation.arrow")),
        }
    else:
        dataset = load_dataset(
            "glue",
            "sst2",
            download_config=DownloadConfig(local_files_only=local_files_only),
        )
    labels = {0: "negative", 1: "positive"}
    rows: list[dict[str, str]] = []
    for split_name in ["train", "validation"]:
        for index, item in enumerate(dataset[split_name]):
            rows.append(
                {
                    "source_id": f"{split_name}_{index}",
                    "label": labels[int(item["label"])],
                    "text": item["sentence"],
                }
            )
    return rows


def load_ade_rows(local_files_only: bool) -> list[dict[str, str]]:
    cache_dir = (
        Path.home()
        / ".cache/huggingface/datasets/SetFit___ade_corpus_v2_classification/default/0.0.0"
        / "0d5751865d26618e2141fe0aecf06477d93d0955"
    )
    if local_files_only and cache_dir.exists():
        dataset = {
            "train": Dataset.from_file(str(cache_dir / "ade_corpus_v2_classification-train.arrow")),
            "test": Dataset.from_file(str(cache_dir / "ade_corpus_v2_classification-test.arrow")),
        }
    else:
        dataset = load_dataset(
            "SetFit/ade_corpus_v2_classification",
            download_config=DownloadConfig(local_files_only=local_files_only),
        )
    labels = {"Not-Related": "non_ade", "Related": "ade"}
    rows: list[dict[str, str]] = []
    for split_name in ["train", "test"]:
        for index, item in enumerate(dataset[split_name]):
            label = labels.get(str(item["label_text"]))
            if label:
                rows.append(
                    {
                        "source_id": f"{split_name}_{index}",
                        "label": label,
                        "text": item["text"],
                    }
                )
    return rows


def load_disaster_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for index, row in enumerate(csv.DictReader(f)):
            target = (row.get("target") or "").strip()
            if target in {"0", "1"}:
                label = "real disaster" if target == "1" else "not disaster"
                source_id = (row.get("id") or "").strip() or str(index)
                rows.append({"source_id": source_id, "label": label, "text": row.get("text") or ""})
    return rows


def format_ratio(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_summary(path: Path, records: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Dataset",
        "Task",
        "Original size",
        "Cleaned size",
        "Sample size",
        "Train",
        "Test",
        "Positive label",
        "Positive ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def prepare_dataset(
    *,
    spec: DatasetSpec,
    raw_rows: list[dict[str, str]],
    sample_size: int,
    train_size: int,
    seed: int,
) -> dict[str, str | int]:
    cleaned_rows = deduplicate_rows(raw_rows)
    sampled_rows = stratified_sample(cleaned_rows, sample_size, seed)
    split_rows = add_stratified_split(sampled_rows, train_size, seed)
    write_dataset(split_rows, spec)

    positive_count = sum(row["label"] == spec.positive_label for row in split_rows)
    return {
        "Dataset": spec.display_name,
        "Task": spec.task,
        "Original size": len(raw_rows),
        "Cleaned size": len(cleaned_rows),
        "Sample size": len(split_rows),
        "Train": sum(row["split"] == "train" for row in split_rows),
        "Test": sum(row["split"] == "test" for row in split_rows),
        "Positive label": spec.positive_label_display,
        "Positive ratio": format_ratio(positive_count / len(split_rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sms-input", default="data/spam.csv")
    parser.add_argument("--disaster-input", default="data/disaster.csv")
    parser.add_argument("--summary-output", default="paper_tables/table_dataset_construction_summary.csv")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    local_files_only = not args.allow_download
    raw_by_key = {
        "sms_spam": load_sms_rows(Path(args.sms_input)),
        "sst2": load_sst2_rows(local_files_only),
        "ade": load_ade_rows(local_files_only),
        "disaster": load_disaster_rows(Path(args.disaster_input)),
    }

    records = [
        prepare_dataset(
            spec=spec,
            raw_rows=raw_by_key[spec.key],
            sample_size=args.sample_size,
            train_size=args.train_size,
            seed=args.seed,
        )
        for spec in SPECS
    ]
    write_summary(Path(args.summary_output), records)

    for record in records:
        print(record)


if __name__ == "__main__":
    main()
