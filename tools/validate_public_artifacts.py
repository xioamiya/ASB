#!/usr/bin/env python3
"""Validate the committed ASB public-artifact manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts"
MANIFEST_PATH = ARTIFACT_ROOT / "manifest.json"
BANNED_COLUMNS = {"text", "notes"}


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest.get("files", [])
    if not entries:
        raise SystemExit("Manifest has no file entries")

    failures: list[str] = []
    for entry in entries:
        path = ARTIFACT_ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"missing: {entry['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            failures.append(f"hash mismatch: {entry['path']}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            row_count = sum(1 for _ in reader)
        if columns != entry["columns"]:
            failures.append(f"column mismatch: {entry['path']}")
        if row_count != entry["rows"]:
            failures.append(f"row-count mismatch: {entry['path']}")
        leaked = BANNED_COLUMNS.intersection(columns)
        if leaked:
            failures.append(f"excluded columns present in {entry['path']}: {sorted(leaked)}")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {len(entries)} public artifact files")


if __name__ == "__main__":
    main()
