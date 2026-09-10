#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python3}"
API_KEY_FILE="${API_KEY_FILE:-config/api_key.doc}"
WORKERS="${WORKERS:-6}"
SLEEP="${SLEEP:-0}"
MAX_RETRIES="${MAX_RETRIES:-8}"
BERT_EPOCHS="${BERT_EPOCHS:-3}"
BERT_BATCH_SIZE="${BERT_BATCH_SIZE:-8}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-distilbert-base-uncased}"
EMBEDDING_MODELS="${EMBEDDING_MODELS:-$EMBEDDING_MODEL}"
MODEL_DATASETS="${MODEL_DATASETS:-sms_spam,sst2,ade,disaster_tweets}"
MODEL_EXTRACTORS="${MODEL_EXTRACTORS:-deepseek_v4_flash,gpt_4_1_mini,qwen_plus,gemini_2_5_flash}"
SCHEMA_DATASETS="${SCHEMA_DATASETS:-sms_spam,ade}"
PHASE="${1:-all}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-tmp/mplconfig}"
mkdir -p "$MPLCONFIGDIR"

dataset_input() {
  case "$1" in
    sms_spam) echo "data/sms_spam_5000.csv" ;;
    sst2) echo "data/sst2_5000.csv" ;;
    ade) echo "data/ade_5000.csv" ;;
    disaster_tweets) echo "data/disaster_5000.csv" ;;
    *) echo "Unknown dataset: $1" >&2; exit 2 ;;
  esac
}

dataset_features() {
  case "$1" in
    sms_spam) echo "config/human20/sms_spam_human20_features.json" ;;
    sst2) echo "config/human20/sst2_human20_features.json" ;;
    ade) echo "config/human20/ade_human20_features.json" ;;
    disaster_tweets) echo "config/human20/disaster_tweets_human20_features.json" ;;
    *) echo "Unknown dataset: $1" >&2; exit 2 ;;
  esac
}

dataset_item_label() {
  case "$1" in
    sms_spam) echo "SMS message" ;;
    sst2) echo "Movie review sentence" ;;
    ade) echo "Biomedical sentence" ;;
    disaster_tweets) echo "Tweet" ;;
    *) echo "Unknown dataset: $1" >&2; exit 2 ;;
  esac
}

dataset_task_guard() {
  case "$1" in
    sms_spam) echo "Do not directly classify whether the SMS is spam or ham." ;;
    sst2) echo "Do not directly classify whether the text is positive or negative." ;;
    ade) echo "Do not directly classify whether the text is ADE-related or non-ADE." ;;
    disaster_tweets) echo "Do not directly classify whether the tweet is a real disaster or not disaster." ;;
    *) echo "Unknown dataset: $1" >&2; exit 2 ;;
  esac
}

dataset_alt_schema_features() {
  case "$1" in
    sms_spam) echo "config/schema_sensitivity/sms_spam_alt20_features.json" ;;
    sst2) echo "config/schema_sensitivity/sst2_alt20_features.json" ;;
    ade) echo "config/schema_sensitivity/ade_alt20_features.json" ;;
    disaster_tweets) echo "config/schema_sensitivity/disaster_tweets_alt20_features.json" ;;
    *) echo "Unknown dataset: $1" >&2; exit 2 ;;
  esac
}

extractor_provider() {
  case "$1" in
    deepseek_v4_flash) echo "deepseek" ;;
    gpt_4_1_mini) echo "openai" ;;
    qwen_plus) echo "qwen" ;;
    claude_sonnet_4) echo "anthropic" ;;
    gemini_2_5_flash) echo "gemini" ;;
    *) echo "Unknown extractor: $1" >&2; exit 2 ;;
  esac
}

run_datasets() {
  "$PY" scripts/prepare_unified_datasets.py \
    --summary-output paper_tables/table_dataset_construction_summary.csv
}

run_one_feature_file() {
  local dataset="$1"
  local variant="$2"
  local out_dir="${3:-output/features/unified_human20}"
  local raw_dir="${4:-output/raw_llm/unified_human20}"
  local provider="${5:-deepseek}"
  local suffix="${6:-$variant}"

  "$PY" scripts/generate_llm_features.py \
    --input "$(dataset_input "$dataset")" \
    --features "$(dataset_features "$dataset")" \
    --output "${out_dir}/${dataset}_5000_human20_${suffix}.csv" \
    --raw-output "${raw_dir}/${dataset}_5000_human20_${suffix}.jsonl" \
    --provider "$provider" \
    --prompt-variant "$variant" \
    --item-label "$(dataset_item_label "$dataset")" \
    --task-guard "$(dataset_task_guard "$dataset")" \
    --api-key-file "$API_KEY_FILE" \
    --workers "$WORKERS" \
    --sleep "$SLEEP" \
    --max-retries "$MAX_RETRIES"
}

run_features() {
  for dataset in sms_spam sst2 ade disaster_tweets; do
    run_one_feature_file "$dataset" strict
  done
}

run_prompt_features() {
  for dataset in sms_spam sst2 ade disaster_tweets; do
    run_one_feature_file "$dataset" direct
    run_one_feature_file "$dataset" inference
  done
}

run_llm_predictions() {
  "$PY" scripts/generate_llm_classification.py \
    --input data/sms_spam_5000.csv \
    --output output/llm_predictions/unified_human20/sms_spam_zero_shot.csv \
    --raw-output output/raw_llm/unified_human20/sms_spam_zero_shot.jsonl \
    --labels ham spam \
    --shots 0 \
    --item-label "SMS message" \
    --task-description "Classify the SMS message as ham or spam." \
    --api-key-file "$API_KEY_FILE" \
    --workers "$WORKERS" \
    --sleep "$SLEEP" \
    --max-retries "$MAX_RETRIES"
  for seed in 1 2 3; do
    "$PY" scripts/generate_llm_classification.py \
      --input data/sms_spam_5000.csv \
      --output "output/llm_predictions/unified_human20/sms_spam_4shot_seed${seed}.csv" \
      --raw-output "output/raw_llm/unified_human20/sms_spam_4shot_seed${seed}.jsonl" \
      --labels ham spam \
      --shots 4 \
      --example-seed "$seed" \
      --item-label "SMS message" \
      --task-description "Classify the SMS message as ham or spam." \
      --api-key-file "$API_KEY_FILE" \
      --workers "$WORKERS" \
      --sleep "$SLEEP" \
      --max-retries "$MAX_RETRIES"
  done

  "$PY" scripts/generate_llm_classification.py \
    --input data/sst2_5000.csv \
    --output output/llm_predictions/unified_human20/sst2_zero_shot.csv \
    --raw-output output/raw_llm/unified_human20/sst2_zero_shot.jsonl \
    --labels negative positive \
    --shots 0 \
    --item-label "Movie review sentence" \
    --task-description "Classify the movie review sentence sentiment as negative or positive." \
    --api-key-file "$API_KEY_FILE" \
    --workers "$WORKERS" \
    --sleep "$SLEEP" \
    --max-retries "$MAX_RETRIES"
  for seed in 1 2 3; do
    "$PY" scripts/generate_llm_classification.py \
      --input data/sst2_5000.csv \
      --output "output/llm_predictions/unified_human20/sst2_4shot_seed${seed}.csv" \
      --raw-output "output/raw_llm/unified_human20/sst2_4shot_seed${seed}.jsonl" \
      --labels negative positive \
      --shots 4 \
      --example-seed "$seed" \
      --item-label "Movie review sentence" \
      --task-description "Classify the movie review sentence sentiment as negative or positive." \
      --api-key-file "$API_KEY_FILE" \
      --workers "$WORKERS" \
      --sleep "$SLEEP" \
      --max-retries "$MAX_RETRIES"
  done

  "$PY" scripts/generate_llm_classification.py \
    --input data/ade_5000.csv \
    --output output/llm_predictions/unified_human20/ade_zero_shot.csv \
    --raw-output output/raw_llm/unified_human20/ade_zero_shot.jsonl \
    --labels non_ade ade \
    --shots 0 \
    --item-label "Biomedical sentence" \
    --task-description "Classify whether the biomedical sentence is related to an adverse drug event. Use ade if it is ADE-related and non_ade otherwise." \
    --api-key-file "$API_KEY_FILE" \
    --workers "$WORKERS" \
    --sleep "$SLEEP" \
    --max-retries "$MAX_RETRIES"
  for seed in 1 2 3; do
    "$PY" scripts/generate_llm_classification.py \
      --input data/ade_5000.csv \
      --output "output/llm_predictions/unified_human20/ade_4shot_seed${seed}.csv" \
      --raw-output "output/raw_llm/unified_human20/ade_4shot_seed${seed}.jsonl" \
      --labels non_ade ade \
      --shots 4 \
      --example-seed "$seed" \
      --item-label "Biomedical sentence" \
      --task-description "Classify whether the biomedical sentence is related to an adverse drug event. Use ade if it is ADE-related and non_ade otherwise." \
      --api-key-file "$API_KEY_FILE" \
      --workers "$WORKERS" \
      --sleep "$SLEEP" \
      --max-retries "$MAX_RETRIES"
  done

  "$PY" scripts/generate_llm_classification.py \
    --input data/disaster_5000.csv \
    --output output/llm_predictions/unified_human20/disaster_tweets_zero_shot.csv \
    --raw-output output/raw_llm/unified_human20/disaster_tweets_zero_shot.jsonl \
    --labels "not disaster" "real disaster" \
    --shots 0 \
    --item-label "Tweet" \
    --task-description "Classify whether the tweet reports a real disaster. Use real disaster for real disaster reports and not disaster otherwise." \
    --api-key-file "$API_KEY_FILE" \
    --workers "$WORKERS" \
    --sleep "$SLEEP" \
    --max-retries "$MAX_RETRIES"
  for seed in 1 2 3; do
    "$PY" scripts/generate_llm_classification.py \
      --input data/disaster_5000.csv \
      --output "output/llm_predictions/unified_human20/disaster_tweets_4shot_seed${seed}.csv" \
      --raw-output "output/raw_llm/unified_human20/disaster_tweets_4shot_seed${seed}.jsonl" \
      --labels "not disaster" "real disaster" \
      --shots 4 \
      --example-seed "$seed" \
      --item-label "Tweet" \
      --task-description "Classify whether the tweet reports a real disaster. Use real disaster for real disaster reports and not disaster otherwise." \
      --api-key-file "$API_KEY_FILE" \
      --workers "$WORKERS" \
      --sleep "$SLEEP" \
      --max-retries "$MAX_RETRIES"
  done
}

run_table() {
  "$PY" scripts/run_unified_method_table.py \
    --features-dir output/features/unified_human20 \
    --llm-predictions-dir output/llm_predictions/unified_human20 \
    --seeds 1 2 3 \
    --bert-epochs "$BERT_EPOCHS" \
    --bert-batch-size "$BERT_BATCH_SIZE" \
    --require-llm-predictions \
    --per-seed-output output/results/unified_method_performance_by_seed.csv \
    --output paper_tables/table_unified_method_performance.csv
  run_embedding_lr
}

run_embedding_lr() {
  "$PY" scripts/run_embedding_lr_baseline.py \
    --model-names "$EMBEDDING_MODELS" \
    --output paper_tables/table_embedding_lr_baseline.csv \
    --per-seed-output output/results/embedding_lr_baseline_by_seed.csv \
    --merge-table paper_tables/table_unified_method_performance.csv
}

run_model_features() {
  IFS=',' read -r -a datasets <<< "$MODEL_DATASETS"
  IFS=',' read -r -a extractors <<< "$MODEL_EXTRACTORS"
  for dataset in "${datasets[@]}"; do
    dataset="${dataset//[[:space:]]/}"
    for extractor in "${extractors[@]}"; do
      extractor="${extractor//[[:space:]]/}"
      run_one_feature_file \
        "$dataset" \
        strict \
        output/features/unified_human20_models \
        output/raw_llm/unified_human20_models \
        "$(extractor_provider "$extractor")" \
        "$extractor"
    done
  done
}

run_model_table() {
  "$PY" scripts/analyze_unified_model_comparison.py \
    --features-dir output/features/unified_human20_models \
    --datasets "$MODEL_DATASETS" \
    --models "$MODEL_EXTRACTORS" \
    --output paper_tables/table_model_comparison_asb_lr.csv
}

run_prompt_table() {
  "$PY" scripts/analyze_unified_prompt_stability.py \
    --features-dir output/features/unified_human20 \
    --output paper_tables/table_prompt_stability_summary.csv
}

run_group_ablation() {
  "$PY" scripts/analyze_unified_group_ablation.py \
    --features-dir output/features/unified_human20 \
    --output paper_tables/table_feature_group_ablation_condensed.csv \
    --full-output output/results/unified_feature_group_ablation_full.csv
}

run_counterfactuals() {
  "$PY" scripts/analyze_unified_counterfactual_coverage.py \
    --features-dir output/features/unified_human20 \
    --output paper_tables/table_counterfactual_feature_edit_coverage.csv \
    --details-output output/results/unified_counterfactual_feature_edits.csv
}

run_human_intervention() {
  "$PY" scripts/analyze_human_concept_intervention.py \
    --output output/results/human_concept_intervention_by_dataset.csv \
    --examples-output output/results/human_concept_intervention_examples.csv \
    --paper-table paper_tables/table_human_concept_intervention.csv
}

run_mi_feature_selection() {
  "$PY" scripts/analyze_unified_mi_feature_selection.py \
    --features-dir output/features/unified_human20 \
    --output paper_tables/table_mi_feature_selection_sensitivity.csv \
    --details-output output/results/unified_mi_feature_selection_sensitivity_details.csv \
    --ranking-output output/results/unified_mi_feature_selection_ranking.csv
}

run_schema_features() {
  IFS=',' read -r -a datasets <<< "$SCHEMA_DATASETS"
  for dataset in "${datasets[@]}"; do
    dataset="${dataset//[[:space:]]/}"
    "$PY" scripts/generate_llm_features.py \
      --input "$(dataset_input "$dataset")" \
      --features "$(dataset_alt_schema_features "$dataset")" \
      --output "output/features/schema_sensitivity/${dataset}_5000_alt20_strict.csv" \
      --raw-output "output/raw_llm/schema_sensitivity/${dataset}_5000_alt20_strict.jsonl" \
      --provider deepseek \
      --prompt-variant strict \
      --item-label "$(dataset_item_label "$dataset")" \
      --task-guard "$(dataset_task_guard "$dataset")" \
      --api-key-file "$API_KEY_FILE" \
      --workers "$WORKERS" \
      --sleep "$SLEEP" \
      --max-retries "$MAX_RETRIES"
  done
}

run_schema_sensitivity() {
  "$PY" scripts/analyze_schema_sensitivity.py \
    --alt-features-dir output/features/schema_sensitivity \
    --alt-config-dir config/schema_sensitivity \
    --original-features-dir output/features/unified_human20 \
    --output paper_tables/table_schema_sensitivity.csv \
    --per-seed-output output/results/schema_sensitivity_by_seed.csv \
    --overlap-output paper_tables/table_schema_overlap.csv
}

run_concept_audit_sample() {
  "$PY" scripts/create_concept_audit_sample.py \
    --features-dir output/features/unified_human20 \
    --annotation-output output/concept_audit/concept_audit_annotations.csv \
    --key-output output/concept_audit/concept_audit_key.csv \
    --feature-guide-output output/concept_audit/concept_audit_feature_guide.csv
}

run_concept_audit_table() {
  "$PY" scripts/analyze_concept_audit.py \
    --annotations output/concept_audit/concept_audit_annotations.csv \
    --key output/concept_audit/concept_audit_key.csv \
    --summary-output paper_tables/table_concept_audit_summary.csv \
    --feature-output output/results/concept_audit_feature_level.csv
}

run_label_proxy() {
  "$PY" scripts/analyze_label_proxy_analysis.py \
    --features-dir output/features/unified_human20 \
    --output paper_tables/table_label_proxy_summary.csv \
    --details-output output/results/label_proxy_feature_details.csv
}

run_api_cost() {
  "$PY" scripts/analyze_api_cost_summary.py \
    --raw-dir output/raw_llm \
    --features-dir output/features/unified_human20_models \
    --output paper_tables/table_api_cost_summary.csv \
    --details-output output/results/unified_api_cost_summary_details.csv
}

run_tables_from_cache() {
  run_table
  run_prompt_table
  run_group_ablation
  run_counterfactuals
  run_human_intervention
  run_mi_feature_selection
  run_schema_sensitivity
  run_label_proxy
  run_model_table
  run_api_cost
}

case "$PHASE" in
  datasets) run_datasets ;;
  features) run_features ;;
  prompt_features) run_prompt_features ;;
  llm) run_llm_predictions ;;
  table) run_table ;;
  embedding_lr) run_embedding_lr ;;
  prompt_table) run_prompt_table ;;
  group_ablation) run_group_ablation ;;
  counterfactuals) run_counterfactuals ;;
  human_intervention) run_human_intervention ;;
  mi_feature_selection) run_mi_feature_selection ;;
  schema_features) run_schema_features ;;
  schema_sensitivity) run_schema_sensitivity ;;
  concept_audit_sample) run_concept_audit_sample ;;
  concept_audit_table) run_concept_audit_table ;;
  label_proxy) run_label_proxy ;;
  model_features) run_model_features ;;
  model_table) run_model_table ;;
  api_cost) run_api_cost ;;
  tables_from_cache) run_tables_from_cache ;;
  all)
    run_datasets
    run_features
    run_prompt_features
    run_llm_predictions
    run_table
    run_prompt_table
    run_group_ablation
    run_counterfactuals
    run_human_intervention
    run_mi_feature_selection
    run_schema_sensitivity
    run_concept_audit_sample
    run_label_proxy
    run_model_features
    run_model_table
    run_api_cost
    ;;
  *)
    echo "Usage: $0 {datasets|features|prompt_features|llm|table|embedding_lr|prompt_table|group_ablation|counterfactuals|human_intervention|mi_feature_selection|schema_features|schema_sensitivity|concept_audit_sample|concept_audit_table|label_proxy|model_features|model_table|api_cost|tables_from_cache|all}" >&2
    exit 2
    ;;
esac
