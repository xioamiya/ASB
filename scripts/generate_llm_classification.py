#!/usr/bin/env python3
"""Generate zero-shot or few-shot LLM predictions for binary CSV datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a careful binary text classifier.
Return JSON only. Do not include markdown, explanations, or extra keys.
Choose exactly one label from the allowed labels."""


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def read_api_key(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            _, value = line.split("=", 1)
            return value.strip().strip('"').strip("'")
        return line.strip().strip('"').strip("'")
    return ""


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def select_examples(
    train_rows: list[dict[str, str]],
    labels: list[str],
    shots: int,
    seed: int,
) -> list[dict[str, str]]:
    if shots == 0:
        return []
    if shots % len(labels) != 0:
        raise ValueError(f"--shots must be divisible by the number of labels ({len(labels)})")
    per_label = shots // len(labels)
    rng = random.Random(seed)
    examples: list[dict[str, str]] = []
    for label in labels:
        candidates = [row for row in train_rows if row["label"] == label]
        if len(candidates) < per_label:
            raise ValueError(f"Cannot draw {per_label} examples for label {label}; found {len(candidates)}")
        examples.extend(rng.sample(candidates, per_label))
    rng.shuffle(examples)
    return examples


def build_user_prompt(
    *,
    text: str,
    item_label: str,
    labels: list[str],
    task_description: str,
    examples: list[dict[str, str]],
) -> str:
    example_block = ""
    if examples:
        lines = []
        for index, row in enumerate(examples, start=1):
            lines.append(
                f"Example {index}\n"
                f"{item_label}: {row['text']}\n"
                f"Label: {row['label']}"
            )
        example_block = "Labeled examples:\n" + "\n\n".join(lines) + "\n\n"
    return (
        f"Task: {task_description}\n"
        f"Allowed labels: {', '.join(labels)}\n\n"
        f"{example_block}"
        f"{item_label}:\n{text}\n\n"
        "Return JSON with exactly one key: label."
    )


def request_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    thinking: str,
    timeout: int,
    use_response_format: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    if thinking in {"enabled", "disabled"}:
        payload["thinking"] = {"type": thinking}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_label(content: str, labels: list[str]) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {content[:300]}") from exc
    candidate = str(parsed.get("label", "")).strip().lower()
    normalized = {label.lower(): label for label in labels}
    if candidate not in normalized:
        raise ValueError(f"Invalid label {parsed.get('label')!r}; expected one of {labels}")
    return normalized[candidate]


def process_row(
    *,
    row: dict[str, str],
    labels: list[str],
    item_label: str,
    task_description: str,
    examples: list[dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    thinking: str,
    timeout: int,
    max_retries: int,
    sleep: float,
    shots: int,
    example_seed: int,
    use_response_format: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                text=row["text"],
                item_label=item_label,
                labels=labels,
                task_description=task_description,
                examples=examples,
            ),
        },
    ]
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = request_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                thinking=thinking,
                timeout=timeout,
                use_response_format=use_response_format,
            )
            content = response["choices"][0]["message"]["content"]
            prediction = parse_label(content, labels)
            output_row = {
                "id": row["id"],
                "source_id": row.get("source_id", ""),
                "split": row.get("split", ""),
                "label": row["label"],
                "text": row["text"],
                "prediction": prediction,
                "shots": shots,
                "example_seed": example_seed,
            }
            raw_record = {
                "id": row["id"],
                "model": model,
                "temperature": temperature,
                "thinking": thinking,
                "shots": shots,
                "example_seed": example_seed,
                "examples": [{"id": item["id"], "label": item["label"]} for item in examples],
                "request": messages,
                "response": response,
            }
            if sleep:
                time.sleep(sleep)
            return output_row, raw_record
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
            last_error = exc
            wait_seconds = min(2**attempt, 30)
            print(
                f"[{row['id']}] attempt {attempt}/{max_retries} failed: {exc}; "
                f"retrying in {wait_seconds}s",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to process {row['id']} after retries") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--labels", nargs=2, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--shots", type=int, choices=[0, 4], default=0)
    parser.add_argument("--example-seed", type=int, default=1)
    parser.add_argument("--item-label", default="Text")
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0")))
    parser.add_argument("--thinking", choices=["auto", "enabled", "disabled"], default=os.getenv("LLM_THINKING", "disabled"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--disable-response-format", action="store_true")
    args = parser.parse_args()

    if not args.api_key and args.api_key_file:
        args.api_key = read_api_key(Path(args.api_key_file))
    if not args.api_key:
        sys.exit("Missing API key. Set DEEPSEEK_API_KEY, set LLM_API_KEY, pass --api-key, or pass --api-key-file.")

    all_rows = read_rows(Path(args.input))
    test_rows = [row for row in all_rows if row.get("split") == args.split]
    train_rows = [row for row in all_rows if row.get("split") == "train"]
    examples = select_examples(train_rows, args.labels, args.shots, args.example_seed)
    completed = read_completed_ids(Path(args.output))
    pending = [row for row in test_rows if row["id"] not in completed]
    print(f"Rows: {len(test_rows)}; completed: {len(completed)}; pending: {len(pending)}; shots={args.shots}")

    fieldnames = ["id", "source_id", "split", "label", "text", "prediction", "shots", "example_seed"]
    common_kwargs = {
        "labels": args.labels,
        "item_label": args.item_label,
        "task_description": args.task_description,
        "examples": examples,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "sleep": args.sleep,
        "shots": args.shots,
        "example_seed": args.example_seed,
        "use_response_format": not args.disable_response_format,
    }
    if args.workers <= 1:
        for index, row in enumerate(pending, start=1):
            output_row, raw_record = process_row(row=row, **common_kwargs)
            append_csv_row(Path(args.output), fieldnames, output_row)
            append_jsonl(Path(args.raw_output), raw_record)
            if index % 25 == 0 or index == len(pending):
                print(f"Processed {index}/{len(pending)} pending rows")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_row, row=row, **common_kwargs) for row in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                output_row, raw_record = future.result()
                append_csv_row(Path(args.output), fieldnames, output_row)
                append_jsonl(Path(args.raw_output), raw_record)
                if index % 25 == 0 or index == len(pending):
                    print(f"Processed {index}/{len(pending)} pending rows")


if __name__ == "__main__":
    main()
