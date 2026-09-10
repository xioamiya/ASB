# Public reproducibility artifacts

This directory contains the derived artifacts used to audit the reported
experiments without redistributing third-party source text.

```text
concept_audit/        Human audit values and the blinded audit key
features/             Binary ASB matrices for prompt and schema analyses
llm_predictions/      Zero- and four-shot predictions without source text
results/              Per-seed, per-feature, and robustness details
splits/               Exact row identifiers, labels, and train/test assignments
manifest.json         Columns, row counts, and SHA-256 hashes
```

The exporter removes `text` and free-form `notes` columns. Consequently these
files support inspection of the semantic matrices and reported statistics but
do not independently redistribute or reconstruct the source corpora. To rerun
raw-text and transformer baselines, obtain the datasets as described in
`data/README.md` and join source text by `id` after reproducing the fixed sample.

Maintainers can regenerate this directory from a complete private workspace:

```bash
python3 tools/export_public_artifacts.py --source-root /path/to/full/asb/workspace
```
