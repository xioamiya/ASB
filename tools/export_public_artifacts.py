#!/usr/bin/env python3
"""Export reproducibility artifacts without redistributing source text.

Run this script from a private, complete ASB workspace. It copies only the
paper-facing feature matrices, predictions, audit annotations, result details,
and split identifiers. Free-text fields and annotator notes are removed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


RESULT_FILES = (
    "concept_audit_feature_level.csv",
    "cross_extractor_transfer_details.csv",
    "embedding_lr_baseline_by_seed.csv",
    "human_concept_intervention_by_dataset.csv",
    "human_concept_intervention_examples.csv",
    "label_proxy_feature_details.csv",
    "schema_resampling_panels.csv",
    "schema_sensitivity_by_seed.csv",
    "second_annotator_agreement_feature_level.csv",
    "second_annotator_annotations_long.csv",
    "strong_proxy_removal_details.csv",
    "unified_api_cost_summary_details.csv",
    "unified_counterfactual_feature_edits.csv",
    "unified_feature_group_ablation_full.csv",
    "unified_method_performance_by_seed.csv",
    "unified_mi_feature_selection_ranking.csv",
    "unified_mi_feature_selection_sensitivity_details.csv",
)

AUDIT_FILES = (
    "concept_audit_annotations.csv",
    "concept_audit_feature_guide.csv",
    "concept_audit_key.csv",
)

DATA_FILES = (
    "ade_5000.csv",
    "disaster_5000.csv",
    "sms_spam_5000.csv",
    "sst2_5000.csv",
)

DROP_COLUMNS = {"text", "notes"}


def export_csv(source: Path, destination: Path, destination_root: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {source}")
        fields = [name for name in reader.fieldnames if name not in DROP_COLUMNS]
        rows = list(reader)

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {
        "path": destination.relative_to(destination_root).as_posix(),
        "rows": len(rows),
        "columns": fields,
        "removed_columns": sorted(set(reader.fieldnames) & DROP_COLUMNS),
        "sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    destination_root = args.destination_root.resolve()
    manifest: list[dict[str, object]] = []

    for name in DATA_FILES:
        source = source_root / "data" / name
        manifest.append(export_csv(source, destination_root / "splits" / name, destination_root))

    for relative_dir in (
        Path("output/features/unified_human20"),
        Path("output/features/unified_human20_models"),
        Path("output/features/schema_sensitivity"),
        Path("output/llm_predictions/unified_human20"),
    ):
        for source in sorted((source_root / relative_dir).glob("*.csv")):
            relative = source.relative_to(source_root / "output")
            manifest.append(export_csv(source, destination_root / relative, destination_root))

    for name in RESULT_FILES:
        source = source_root / "output" / "results" / name
        manifest.append(export_csv(source, destination_root / "results" / name, destination_root))

    for name in AUDIT_FILES:
        source = source_root / "output" / "concept_audit" / name
        manifest.append(export_csv(source, destination_root / "concept_audit" / name, destination_root))

    manifest_path = destination_root / "manifest.json"
    manifest_path.write_text(
        json.dumps({"files": manifest}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(manifest)} files to {destination_root}")


if __name__ == "__main__":
    main()
