#!/usr/bin/env python3
"""Create per-dataset Markdown/CSV packets for concept-audit annotation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def existing_answer(row: dict[str, str], features: list[str]) -> str:
    values = [row.get(feature, "").strip() for feature in features]
    return ",".join(values) if any(values) else ",".join([""] * len(features))


def selected_dataset_keys(dataset: str, datasets: str | None) -> set[str] | None:
    if datasets:
        return {item.strip() for item in datasets.split(",") if item.strip()}
    if dataset == "all":
        return None
    return {dataset}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-dir", default="output/concept_audit/wide")
    parser.add_argument("--feature-guide", default="output/concept_audit/concept_audit_feature_guide.csv")
    parser.add_argument("--output-dir", default="output/concept_audit/annotation_packets")
    parser.add_argument("--dataset", choices=["all", "sms_spam", "sst2", "ade", "disaster_tweets"], default="all")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset keys. Overrides --dataset, e.g. sms_spam,sst2.",
    )
    parser.add_argument(
        "--blank-answers",
        action="store_true",
        help="Leave answer fields empty for independent blind annotation.",
    )
    args = parser.parse_args()

    guide_rows = read_rows(Path(args.feature_guide))
    question_by_dataset_feature = {
        (row["dataset_key"], row["feature_id"]): row["feature_question"] for row in guide_rows
    }
    requested_dataset_keys = selected_dataset_keys(args.dataset, args.datasets)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for wide_path in sorted(Path(args.wide_dir).glob("*_concept_audit_wide.csv")):
        dataset_key = wide_path.name.replace("_concept_audit_wide.csv", "")
        if requested_dataset_keys is not None and dataset_key not in requested_dataset_keys:
            continue
        rows = read_rows(wide_path)
        if not rows:
            continue
        dataset = rows[0]["dataset"]
        features = [field for field in rows[0] if field not in {"dataset", "dataset_key", "id", "text"}]
        md_path = output_dir / f"{dataset_key}_annotation_packet.md"
        csv_path = output_dir / f"{dataset_key}_answer_template.csv"

        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# {dataset} Concept Audit Annotation Packet\n\n")
            f.write(
                "填写规则：每条文本回复 20 个值，顺序必须与下方 feature order 一致。"
                "`1`=是，`0`=否，`U`=不确定。\n\n"
            )
            f.write("## Feature Order\n\n")
            for index, feature_id in enumerate(features, start=1):
                question = question_by_dataset_feature.get((dataset_key, feature_id), "")
                f.write(f"{index}. `{feature_id}`: {question}\n")
            f.write("\n## Items\n\n")
            for index, row in enumerate(rows, start=1):
                answer = "" if args.blank_answers else existing_answer(row, features)
                f.write(f"### {index}. {row['id']}\n\n")
                f.write(f"Text: {row['text']}\n\n")
                if args.blank_answers:
                    f.write("Answer:\n\n")
                else:
                    f.write(f"Answer: `{answer}`\n\n")

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "id", "text", "answer_20_values"], lineterminator="\n")
            writer.writeheader()
            for index, row in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "index": index,
                        "id": row["id"],
                        "text": row["text"],
                        "answer_20_values": "" if args.blank_answers else existing_answer(row, features),
                    }
                )
        print(f"Wrote {md_path}")
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
