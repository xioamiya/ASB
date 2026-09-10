#!/usr/bin/env python3
"""Estimate API usage and cost for the four-platform ASB extractor comparison."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PriceSpec:
    input_per_million: float
    output_per_million: float
    currency: str
    cached_input_per_million: float | None = None
    cache_miss_input_per_million: float | None = None


@dataclass(frozen=True)
class PlatformSpec:
    platform: str
    provider: str
    model: str
    datasets: tuple[str, ...]
    raw_paths: tuple[Path, ...]
    feature_paths: tuple[Path, ...]
    price: PriceSpec
    price_basis: str


MODEL_SCOPE = "ADE + Disaster Tweets ASB feature extraction"


def platform_specs(raw_dir: Path, feature_dir: Path) -> list[PlatformSpec]:
    return [
        PlatformSpec(
            platform="DeepSeek",
            provider="deepseek",
            model="deepseek-v4-flash",
            datasets=("ADE", "Disaster Tweets"),
            raw_paths=(
                raw_dir / "unified_human20/ade_5000_human20_strict.jsonl",
                raw_dir / "unified_human20/disaster_tweets_5000_human20_strict.jsonl",
            ),
            feature_paths=(
                feature_dir / "ade_5000_human20_deepseek_v4_flash.csv",
                feature_dir / "disaster_tweets_5000_human20_deepseek_v4_flash.csv",
            ),
            price=PriceSpec(
                input_per_million=0.14,
                output_per_million=0.28,
                cached_input_per_million=0.0028,
                cache_miss_input_per_million=0.14,
                currency="USD",
            ),
            price_basis="cache-aware v4-flash estimate",
        ),
        PlatformSpec(
            platform="OpenAI",
            provider="openai",
            model="gpt-4.1-mini",
            datasets=("ADE", "Disaster Tweets"),
            raw_paths=(
                raw_dir / "unified_human20_models/ade_5000_human20_gpt_4_1_mini.jsonl",
                raw_dir / "unified_human20_models/disaster_tweets_5000_human20_gpt_4_1_mini.jsonl",
            ),
            feature_paths=(
                feature_dir / "ade_5000_human20_gpt_4_1_mini.csv",
                feature_dir / "disaster_tweets_5000_human20_gpt_4_1_mini.csv",
            ),
            price=PriceSpec(
                input_per_million=0.40,
                output_per_million=1.60,
                cached_input_per_million=0.10,
                currency="USD",
            ),
            price_basis="GPT-4.1 mini text tokens",
        ),
        PlatformSpec(
            platform="Qwen",
            provider="qwen",
            model="qwen-plus",
            datasets=("ADE", "Disaster Tweets"),
            raw_paths=(
                raw_dir / "unified_human20_models/ade_5000_human20_qwen_plus.jsonl",
                raw_dir / "unified_human20_models/disaster_tweets_5000_human20_qwen_plus.jsonl",
            ),
            feature_paths=(
                feature_dir / "ade_5000_human20_qwen_plus.csv",
                feature_dir / "disaster_tweets_5000_human20_qwen_plus.csv",
            ),
            price=PriceSpec(
                input_per_million=0.115,
                output_per_million=0.287,
                currency="USD",
            ),
            price_basis="Alibaba Cloud qwen-plus 0-128K non-thinking estimate",
        ),
        PlatformSpec(
            platform="Google",
            provider="gemini",
            model="gemini-2.5-flash",
            datasets=("ADE", "Disaster Tweets"),
            raw_paths=(
                raw_dir / "unified_human20_models/ade_5000_human20_gemini_2_5_flash.jsonl",
                raw_dir / "unified_human20_models/disaster_tweets_5000_human20_gemini_2_5_flash.jsonl",
            ),
            feature_paths=(
                feature_dir / "ade_5000_human20_gemini_2_5_flash.csv",
                feature_dir / "disaster_tweets_5000_human20_gemini_2_5_flash.csv",
            ),
            price=PriceSpec(
                input_per_million=0.30,
                output_per_million=2.50,
                currency="USD",
            ),
            price_basis="Gemini 2.5 Flash; output includes thinking tokens",
        ),
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def usage_from_record(record: dict[str, Any], provider: str) -> dict[str, int]:
    response = record.get("response", {})
    if provider == "gemini":
        usage = response.get("provider_raw_response", {}).get("usageMetadata", {})
        prompt = int(usage.get("promptTokenCount", 0) or 0)
        visible_output = int(usage.get("candidatesTokenCount", 0) or 0)
        thinking = int(usage.get("thoughtsTokenCount", 0) or 0)
        total = int(usage.get("totalTokenCount", 0) or 0)
        billable_output = visible_output + thinking
        if not billable_output and total > prompt:
            billable_output = total - prompt
        return {
            "input_tokens": prompt,
            "cached_input_tokens": 0,
            "cache_miss_input_tokens": prompt,
            "output_tokens": billable_output,
            "visible_output_tokens": visible_output,
            "thinking_tokens": thinking,
            "total_tokens": total or prompt + billable_output,
        }

    usage = response.get("usage", {})
    prompt = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
    output = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
    total = int(usage.get("total_tokens", 0) or prompt + output)
    cached = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = max(cached, int(details.get("cached_tokens", 0) or 0))
    cache_miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
    if not cache_miss:
        cache_miss = max(prompt - cached, 0)
    return {
        "input_tokens": prompt,
        "cached_input_tokens": cached,
        "cache_miss_input_tokens": cache_miss,
        "output_tokens": output,
        "visible_output_tokens": output,
        "thinking_tokens": 0,
        "total_tokens": total,
    }


def estimate_cost(tokens: dict[str, int], price: PriceSpec) -> float:
    if price.cache_miss_input_per_million is not None or price.cached_input_per_million is not None:
        cache_miss_rate = price.cache_miss_input_per_million or price.input_per_million
        cached_rate = price.cached_input_per_million or price.input_per_million
        input_cost = (
            tokens["cache_miss_input_tokens"] * cache_miss_rate
            + tokens["cached_input_tokens"] * cached_rate
        ) / 1_000_000
    else:
        input_cost = tokens["input_tokens"] * price.input_per_million / 1_000_000
    output_cost = tokens["output_tokens"] * price.output_per_million / 1_000_000
    return input_cost + output_cost


def summarize_platform(spec: PlatformSpec) -> dict[str, object]:
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_miss_input_tokens": 0,
        "output_tokens": 0,
        "visible_output_tokens": 0,
        "thinking_tokens": 0,
        "total_tokens": 0,
    }
    api_records = 0
    for path in spec.raw_paths:
        records = read_jsonl(path)
        api_records += len(records)
        for record in records:
            usage = usage_from_record(record, spec.provider)
            for key in totals:
                totals[key] += usage[key]

    feature_rows = sum(count_csv_rows(path) for path in spec.feature_paths)
    manual_or_unbilled_rows = max(feature_rows - api_records, 0)
    cost = estimate_cost(totals, spec.price)
    return {
        "Scope": MODEL_SCOPE,
        "Platform": spec.platform,
        "Model": spec.model,
        "Datasets": "; ".join(spec.datasets),
        "Feature rows": feature_rows,
        "API responses": api_records,
        "Manual/unbilled rows": manual_or_unbilled_rows,
        "Input tokens": totals["input_tokens"],
        "Cached input tokens": totals["cached_input_tokens"],
        "Output tokens": totals["output_tokens"],
        "Visible output tokens": totals["visible_output_tokens"],
        "Thinking tokens": totals["thinking_tokens"],
        "Total tokens": totals["total_tokens"],
        "Currency": spec.price.currency,
        "Input price / 1M": f"{spec.price.input_per_million:.4f}",
        "Output price / 1M": f"{spec.price.output_per_million:.4f}",
        "Estimated cost": f"{cost:.4f}",
        "Price basis": spec.price_basis,
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="output/raw_llm")
    parser.add_argument("--features-dir", default="output/features/unified_human20_models")
    parser.add_argument("--output", default="paper_tables/table_api_cost_summary.csv")
    parser.add_argument("--details-output", default="output/results/unified_api_cost_summary_details.csv")
    args = parser.parse_args()

    rows = [
        summarize_platform(spec)
        for spec in platform_specs(Path(args.raw_dir), Path(args.features_dir))
    ]
    paper_fields = [
        "Platform",
        "Model",
        "Datasets",
        "Feature rows",
        "API responses",
        "Input tokens",
        "Output tokens",
        "Total tokens",
        "Currency",
        "Estimated cost",
    ]
    detail_fields = [
        "Scope",
        "Platform",
        "Model",
        "Datasets",
        "Feature rows",
        "API responses",
        "Manual/unbilled rows",
        "Input tokens",
        "Cached input tokens",
        "Output tokens",
        "Visible output tokens",
        "Thinking tokens",
        "Total tokens",
        "Currency",
        "Input price / 1M",
        "Output price / 1M",
        "Estimated cost",
        "Price basis",
    ]
    write_csv(Path(args.output), rows, paper_fields)
    write_csv(Path(args.details_output), rows, detail_fields)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.details_output}")


if __name__ == "__main__":
    main()
