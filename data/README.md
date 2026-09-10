# Source data

The paper uses four public text-classification datasets. Their text is not
redistributed in this repository; obtain each dataset from its original host
and comply with its terms.

- **SMS Spam Collection:** place the original CSV at `data/spam.csv`. The
  loader expects the common `v1` (label) and `v2` (message) columns.
- **SST-2:** loaded as the `sst2` configuration of Hugging Face `glue`.
- **ADE Corpus V2:** loaded from `SetFit/ade_corpus_v2_classification`.
- **Disaster Tweets:** place the Kaggle training CSV at
  `data/disaster.csv`; the loader uses the `id`, `text`, and `target` columns.

Then run:

```bash
python3 scripts/prepare_unified_datasets.py --allow-download
```

The preparation script removes empty and normalized duplicate texts, samples
5,000 examples per dataset, and creates a stratified 4,000/1,000 train/test
split with seed 42. Exact paper split identifiers are committed under
`artifacts/splits/`.
