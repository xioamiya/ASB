# ASB: Auditable Semantic Bottlenecks

Official code and reproducibility artifacts for **ASB: Auditable Semantic
Bottlenecks for Reliable LLM-Generated Text Features**.

ASB converts each text into 20 binary, human-readable semantic variables and
trains a lightweight classifier on the resulting matrix. The paper evaluates
the predictive utility and reliability of this intermediate representation on
SMS spam, SST-2 sentiment, adverse drug event, and disaster tweet
classification.

## What is included

```text
artifacts/                 Text-free feature matrices, predictions, and details
config/human20/            Final 20-concept schemas and semantic groups
config/schema_sensitivity/ Alternative schemas used in sensitivity analysis
data/                      Source-data acquisition notes
latex/                     Paper source, figures, bibliography, and compiled PDF
paper_tables/              Exact CSV tables used in the paper
scripts/                   Dataset, extraction, modeling, and audit code
tools/                     Public-artifact export and validation utilities
```

The committed artifacts retain row identifiers, fixed splits, labels, semantic
features, predictions, and quantitative audit results. Source text and raw API
responses are intentionally excluded because the source datasets have their
own access and redistribution terms. See [artifacts/README.md](artifacts/README.md)
for the exact boundary.

## Quick inspection (no API calls)

The final paper tables are directly available under `paper_tables/`. The 80
questions used by ASB are under `config/human20/`, and figure source data are
under `latex/figures/source_data/`.

Validate the public artifacts:

```bash
python3 tools/validate_public_artifacts.py
```

Recompute and byte-compare 11 audit and robustness tables from the committed
text-free matrices (no API calls):

```bash
./scripts/reproduce_public_tables.sh
```

Build the paper:

```bash
cd latex
latexmk -pdf -interaction=nonstopmode -halt-on-error ASB.tex
```

## Environment

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The transformer baselines use locally cached Hugging Face models by default.
The reported runs use `distilbert-base-uncased`. A full rerun also requires
access to the four source datasets and to the selected LLM providers.

## Prepare the datasets

Read [data/README.md](data/README.md) first. After obtaining the source data,
the following command cleans duplicates and empty texts, draws a stratified
5,000-example sample, and creates a fixed 4,000/1,000 train/test split with
seed 42:

```bash
python3 scripts/prepare_unified_datasets.py \
  --allow-download \
  --summary-output paper_tables/table_dataset_construction_summary.csv
```

The generated files have this schema:

```text
id,source_id,split,label,text
```

The committed files under `artifacts/splits/` provide the exact identifiers,
labels, and split assignments used for the reported experiments.

## Configure API access

Copy the example file and add only the providers you intend to run:

```bash
cp config/api_key.example config/api_key.doc
```

`config/api_key.doc` is ignored by Git. Never commit credentials or raw API
responses.

## Run the pipeline

The main entry point is:

```bash
API_KEY_FILE=config/api_key.doc \
./scripts/run_unified_human20_pipeline.sh <phase>
```

Common phases are:

| Phase | Purpose | API required |
|---|---|---:|
| `datasets` | Prepare the four fixed datasets | No |
| `features` | Generate the strict ASB feature matrices | Yes |
| `prompt_features` | Generate direct and inference variants | Yes |
| `llm` | Generate zero- and four-shot LLM baselines | Yes |
| `table` | Train models and assemble the main performance table | No* |
| `prompt_table` | Measure prompt stability | No* |
| `group_ablation` | Run semantic-group ablations | No* |
| `counterfactuals` | Run feature-space edit diagnostics | No* |
| `mi_feature_selection` | Run train-only feature selection | No* |
| `schema_sensitivity` | Compare original and alternative schemas | No* |
| `label_proxy` | Measure train-only label association | No* |
| `model_table` | Compare feature extractors | No* |
| `all` | Run the complete pipeline | Yes |

`*` These phases do not make API calls but require locally generated feature
matrices, predictions, or source text. The public text-free artifacts support
result inspection; restore licensed source text by matching `id` before
rerunning text baselines.

Useful controls include:

```bash
PY=python3
WORKERS=6
SLEEP=0
MAX_RETRIES=8
BERT_EPOCHS=3
BERT_BATCH_SIZE=8
MODEL_DATASETS=sms_spam,sst2,ade,disaster_tweets
MODEL_EXTRACTORS=deepseek_v4_flash,gpt_4_1_mini,qwen_plus,gemini_2_5_flash
```

Some internal filenames retain `human20` for compatibility with the original
experiment runs; the paper-facing method names are ASB-LR and ASB-XGB.

## Paper outputs

The principal outputs are:

- `paper_tables/table_unified_method_performance.csv`
- `paper_tables/table_prompt_stability_summary.csv`
- `paper_tables/table_mi_feature_selection_sensitivity.csv`
- `paper_tables/table_feature_group_ablation_condensed.csv`
- `paper_tables/table_counterfactual_feature_edit_coverage.csv`
- `paper_tables/table_model_comparison_asb_lr.csv`
- `paper_tables/table_concept_audit_summary.csv`
- `paper_tables/table_strong_proxy_removal.csv`
- `paper_tables/table_cross_extractor_transfer_summary.csv`
- `paper_tables/table_schema_resampling_summary.csv`
- `paper_tables/table_human_concept_intervention.csv`

Supporting per-seed and per-feature results are under `artifacts/results/`.

## Reproducibility details

- Dataset construction seed: 42.
- Downstream model seeds: 1, 2, and 3 where applicable.
- Each task uses 20 binary questions in five semantic groups.
- Schema construction and label-proxy screening use training data only.
- Feature extraction is resumable by row identifier.
- Raw LLM responses are local-only and are not part of this repository.
- `artifacts/manifest.json` records row counts, columns, and SHA-256 hashes for
  every public derived artifact.

## Citation

Please cite the associated paper. Machine-readable metadata are provided in
`CITATION.cff`; proceedings metadata can be added there once assigned.
