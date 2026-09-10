#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python3}"
OUT="${OUT:-output/public_reproduction}"
mkdir -p "$OUT"

"$PY" scripts/analyze_unified_prompt_stability.py \
  --features-dir artifacts/features/unified_human20 \
  --output "$OUT/prompt.csv"
"$PY" scripts/analyze_unified_group_ablation.py \
  --features-dir artifacts/features/unified_human20 \
  --output "$OUT/group.csv" --full-output "$OUT/group_full.csv"
"$PY" scripts/analyze_unified_counterfactual_coverage.py \
  --features-dir artifacts/features/unified_human20 \
  --output "$OUT/counter.csv" --details-output "$OUT/counter_details.csv"
"$PY" scripts/analyze_unified_mi_feature_selection.py \
  --features-dir artifacts/features/unified_human20 \
  --output "$OUT/mi.csv" --details-output "$OUT/mi_details.csv" \
  --ranking-output "$OUT/mi_ranking.csv"
"$PY" scripts/analyze_schema_sensitivity.py \
  --alt-features-dir artifacts/features/schema_sensitivity \
  --alt-config-dir config/schema_sensitivity \
  --original-features-dir artifacts/features/unified_human20 \
  --output "$OUT/schema.csv" --per-seed-output "$OUT/schema_seeds.csv" \
  --overlap-output "$OUT/schema_overlap.csv"
"$PY" scripts/analyze_label_proxy_analysis.py \
  --features-dir artifacts/features/unified_human20 \
  --output "$OUT/proxy.csv" --details-output "$OUT/proxy_details.csv"
"$PY" scripts/analyze_unified_model_comparison.py \
  --features-dir artifacts/features/unified_human20_models \
  --datasets sms_spam,sst2,ade,disaster_tweets \
  --models deepseek_v4_flash,gpt_4_1_mini,qwen_plus,gemini_2_5_flash \
  --output "$OUT/model.csv"
"$PY" scripts/analyze_cross_extractor_transfer.py \
  --features-dir artifacts/features/unified_human20_models \
  --output "$OUT/cross.csv" --details-output "$OUT/cross_details.csv"
"$PY" scripts/analyze_schema_resampling.py \
  --original-features-dir artifacts/features/unified_human20 \
  --alternative-features-dir artifacts/features/schema_sensitivity \
  --alternative-config-dir config/schema_sensitivity \
  --output "$OUT/resampling.csv" --details-output "$OUT/resampling_details.csv"
"$PY" scripts/analyze_strong_proxy_removal.py \
  --features-dir artifacts/features/unified_human20 \
  --output "$OUT/strong.csv" --details-output "$OUT/strong_details.csv"

compare() {
  if ! cmp -s "$1" "$2"; then
    echo "Mismatch: $2" >&2
    return 1
  fi
  echo "Match: $2"
}

compare "$OUT/prompt.csv" paper_tables/table_prompt_stability_summary.csv
compare "$OUT/group.csv" paper_tables/table_feature_group_ablation_condensed.csv
compare "$OUT/counter.csv" paper_tables/table_counterfactual_feature_edit_coverage.csv
compare "$OUT/mi.csv" paper_tables/table_mi_feature_selection_sensitivity.csv
compare "$OUT/schema.csv" paper_tables/table_schema_sensitivity.csv
compare "$OUT/schema_overlap.csv" paper_tables/table_schema_overlap.csv
compare "$OUT/proxy.csv" paper_tables/table_label_proxy_summary.csv
compare "$OUT/model.csv" paper_tables/table_model_comparison_asb_lr.csv
compare "$OUT/cross.csv" paper_tables/table_cross_extractor_transfer_summary.csv
compare "$OUT/resampling.csv" paper_tables/table_schema_resampling_summary.csv
compare "$OUT/strong.csv" paper_tables/table_strong_proxy_removal.csv

echo "Public-artifact reproduction passed."
