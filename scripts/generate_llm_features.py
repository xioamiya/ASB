#!/usr/bin/env python3
"""Generate a 0/1 semantic feature table using an OpenAI-compatible chat API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SYSTEM_PROMPTS = {
    "strict": """You convert texts into semantic yes/no features.
Return JSON only. Do not include markdown, explanations, or extra keys.
For every feature id, return 1 for yes and 0 for no.
If the answer is unclear or not supported by the text, return 0.
{task_guard}""",
    "direct": """You convert texts into semantic yes/no features.
Return JSON only. Do not include markdown, explanations, or extra keys.
For every feature id, return 1 for yes and 0 for no.
Use your best judgment based on the text content.
{task_guard}""",
    "inference": """You convert texts into semantic yes/no features.
Return JSON only. Do not include markdown, explanations, or extra keys.
For every feature id, return 1 for yes and 0 for no.
Return 1 if the feature is explicitly stated or strongly implied by the message.
Return 0 only if the feature is absent or very unlikely.
{task_guard}""",
}

PROVIDER_DEFAULTS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "key_names": ["LLM_API_KEY", "DEEPSEEK_API_KEY", "deepseek_api"],
        "thinking": "disabled",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1-mini",
        "key_names": ["OPENAI_API_KEY"],
        "thinking": "auto",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "key_names": ["DASHSCOPE_API_KEY", "QWEN_API_KEY", "qwen_api"],
        "thinking": "auto",
    },
    "generic": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
        "key_names": ["LLM_API_KEY", "DEEPSEEK_API_KEY", "deepseek_api"],
        "thinking": "auto",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-20250514",
        "key_names": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "claude-sonnet-4-20250514"],
        "thinking": "auto",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.5-flash",
        "key_names": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini-2.5-flash"],
        "thinking": "auto",
    },
}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def read_api_key(path: Path, key_names: list[str] | None = None) -> str:
    def clean_value(value: str) -> str:
        return value.strip().strip('"').strip("'").strip("“").strip("”").strip()

    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            entries.append((key.strip(), clean_value(value)))
        else:
            entries.append(("", clean_value(line)))
    if key_names:
        wanted = {name.lower() for name in key_names}
        for key, value in entries:
            if key.lower() in wanted:
                return value
        return ""
    if entries:
        return entries[0][1]
    return ""


def build_user_prompt(text: str, features: list[dict[str, str]], item_label: str) -> str:
    feature_lines = "\n".join(
        f"- {feature['id']}: {feature['question']}" for feature in features
    )
    return (
        f"{item_label}:\n"
        f"{text}\n\n"
        "Feature questions:\n"
        f"{feature_lines}\n\n"
        "Return one JSON object with exactly these feature ids as keys and 0/1 integer values."
    )


def request_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    thinking: str,
    provider: str,
    use_response_format: bool,
    feature_ids: list[str],
    timeout: int,
) -> dict[str, Any]:
    if provider == "anthropic":
        system_prompt = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
        user_messages = [message for message in messages if message["role"] != "system"]
        payload = {
            "model": model,
            "max_tokens": 2048,
            "temperature": temperature,
            "system": system_prompt,
            "messages": user_messages,
        }
        req = urllib.request.Request(
            base_url.rstrip("/") + "/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        content = "".join(
            block.get("text", "")
            for block in raw.get("content", [])
            if block.get("type") == "text"
        )
        return {
            "choices": [{"message": {"content": content}}],
            "provider_raw_response": raw,
        }

    if provider == "gemini":
        system_prompt = "\n\n".join(message["content"] for message in messages if message["role"] == "system")
        user_prompt = "\n\n".join(message["content"] for message in messages if message["role"] != "system")
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
            },
        }
        if use_response_format:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            payload["generationConfig"]["responseSchema"] = {
                "type": "OBJECT",
                "properties": {
                    feature_id: {"type": "INTEGER"}
                    for feature_id in feature_ids
                },
                "required": feature_ids,
                "propertyOrdering": feature_ids,
            }
        url = base_url.rstrip("/") + f"/models/{model}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        if not raw.get("candidates"):
            raise ValueError(f"Gemini returned no candidates: {json.dumps(raw, ensure_ascii=False)[:1000]}")
        parts = raw["candidates"][0]["content"].get("parts", [])
        content = "".join(part.get("text", "") for part in parts)
        return {
            "choices": [{"message": {"content": content}}],
            "provider_raw_response": raw,
        }

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    if provider == "deepseek" and thinking in {"enabled", "disabled"}:
        payload["thinking"] = {"type": thinking}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_feature_json(content: str, feature_ids: list[str]) -> dict[str, int]:
    content = content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").strip()
        if content.endswith("```"):
            content = content[:-3].strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {content[:300]}") from exc

    output = {}
    for feature_id in feature_ids:
        value = parsed.get(feature_id)
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, str):
            value = value.strip().lower()
            if value in {"yes", "true", "1"}:
                value = 1
            elif value in {"no", "false", "0"}:
                value = 0
        if value not in {0, 1}:
            raise ValueError(f"Invalid value for {feature_id}: {parsed.get(feature_id)!r}")
        output[feature_id] = int(value)
    return output


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


def process_row(
    *,
    row: dict[str, str],
    features: list[dict[str, str]],
    feature_ids: list[str],
    system_prompt: str,
    item_label: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    thinking: str,
    timeout: int,
    max_retries: int,
    sleep: float,
    prompt_variant: str,
    task_guard: str,
    provider: str,
    use_response_format: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(row["text"], features, item_label)},
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
                provider=provider,
                use_response_format=use_response_format,
                feature_ids=feature_ids,
                timeout=timeout,
            )
            content = response["choices"][0]["message"]["content"]
            feature_values = parse_feature_json(content, feature_ids)
            output_row = {key: row.get(key, "") for key in ["id", "source_id", "split", "label", "text"]}
            output_row.update(feature_values)
            raw_record = {
                "id": row["id"],
                "model": model,
                "provider": provider,
                "temperature": temperature,
                "thinking": thinking,
                "prompt_version": f"semantic_features_{prompt_variant}_v1",
                "prompt_variant": prompt_variant,
                "item_label": item_label,
                "task_guard": task_guard,
                "request": messages,
                "response": response,
            }
            if sleep:
                time.sleep(sleep)
            return output_row, raw_record
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = ""
            body_note = f"; body: {error_body[:800]}" if error_body else ""
            last_error = RuntimeError(f"HTTP Error {exc.code}: {exc.reason}{body_note}")
            if 400 <= exc.code < 500 and exc.code not in {408, 409, 425, 429}:
                print(
                    f"[{row['id']}] non-retryable HTTP {exc.code}: {exc.reason}{body_note}",
                    file=sys.stderr,
                )
                break
            wait_seconds = min(2**attempt, 30)
            print(
                f"[{row['id']}] attempt {attempt}/{max_retries} failed: {last_error}; "
                f"retrying in {wait_seconds}s",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
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
    parser.add_argument("--input", default="data/sms_spam_500.csv")
    parser.add_argument("--features", default="config/sms_spam_20_features.json")
    parser.add_argument("--output", default="output/features/sms_spam_500_llm_features_run1.csv")
    parser.add_argument("--raw-output", default="output/raw_llm/sms_spam_500_llm_features_run1.jsonl")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_DEFAULTS),
        default=os.getenv("LLM_PROVIDER", "deepseek"),
        help="Provider preset for base URL, model, key lookup, and provider-specific payload fields.",
    )
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument(
        "--api-key-name",
        default=None,
        help="Variable name to read from --api-key-file, e.g. OPENAI_API_KEY or DASHSCOPE_API_KEY.",
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0")))
    parser.add_argument(
        "--prompt-variant",
        choices=sorted(SYSTEM_PROMPTS),
        default="strict",
        help="Prompt framing for feature extraction.",
    )
    parser.add_argument("--item-label", default="SMS message")
    parser.add_argument(
        "--task-guard",
        default="Do not directly classify whether the SMS is spam or ham.",
        help="Final instruction that prevents direct label prediction.",
    )
    parser.add_argument(
        "--thinking",
        choices=["auto", "enabled", "disabled"],
        default=os.getenv("LLM_THINKING"),
        help="DeepSeek V4 thinking mode. Disabled is cheaper and more deterministic for feature extraction.",
    )
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent API calls.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Before a concurrent run, process one pending row synchronously to catch authentication or payload errors.",
    )
    parser.add_argument(
        "--disable-response-format",
        action="store_true",
        help="Do not send OpenAI-style response_format. Use this if a compatible provider rejects that field.",
    )
    args = parser.parse_args()

    provider_defaults = PROVIDER_DEFAULTS[args.provider]
    if not args.base_url:
        args.base_url = provider_defaults["base_url"]
    if not args.model:
        args.model = provider_defaults["model"]
    if not args.thinking:
        args.thinking = provider_defaults["thinking"]
    if not args.api_key:
        env_key_names = ["LLM_API_KEY", *provider_defaults["key_names"]]
        for key_name in env_key_names:
            if os.getenv(key_name):
                args.api_key = os.getenv(key_name)
                break
    if not args.api_key and args.api_key_file:
        key_names = [args.api_key_name] if args.api_key_name else provider_defaults["key_names"]
        args.api_key = read_api_key(Path(args.api_key_file), key_names)
    if not args.api_key:
        sys.exit("Missing API key. Set env key, pass --api-key, or pass --api-key-file with --api-key-name.")
    if not args.model:
        sys.exit("Missing model. Set LLM_MODEL or pass --model.")

    rows = read_rows(Path(args.input))
    features = load_json(Path(args.features))
    feature_ids = [feature["id"] for feature in features]
    completed = read_completed_ids(Path(args.output))
    if args.limit is not None:
        rows = rows[: args.limit]

    fieldnames = ["id", "source_id", "split", "label", "text"] + feature_ids
    pending = [row for row in rows if row["id"] not in completed]
    print(f"Rows: {len(rows)}; completed: {len(completed)}; pending: {len(pending)}")

    system_prompt = SYSTEM_PROMPTS[args.prompt_variant].format(task_guard=args.task_guard)
    common_kwargs = {
        "features": features,
        "feature_ids": feature_ids,
        "system_prompt": system_prompt,
        "item_label": args.item_label,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model": args.model,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "timeout": args.timeout,
        "max_retries": args.max_retries,
        "sleep": args.sleep,
        "prompt_variant": args.prompt_variant,
        "task_guard": args.task_guard,
        "provider": args.provider,
        "use_response_format": not args.disable_response_format,
    }
    if args.preflight and pending and args.workers > 1:
        print(f"Preflight row: {pending[0]['id']}")
        output_row, raw_record = process_row(row=pending[0], **common_kwargs)
        append_csv_row(Path(args.output), fieldnames, output_row)
        append_jsonl(Path(args.raw_output), raw_record)
        pending = pending[1:]
        print(f"Preflight succeeded; remaining pending: {len(pending)}")
    if args.workers <= 1:
        for index, row in enumerate(pending, start=1):
            output_row, raw_record = process_row(row=row, **common_kwargs)
            append_csv_row(Path(args.output), fieldnames, output_row)
            append_jsonl(Path(args.raw_output), raw_record)
            if index % 25 == 0 or index == len(pending):
                print(f"Processed {index}/{len(pending)} pending rows")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_row, row=row, **common_kwargs)
                for row in pending
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                output_row, raw_record = future.result()
                append_csv_row(Path(args.output), fieldnames, output_row)
                append_jsonl(Path(args.raw_output), raw_record)
                if index % 25 == 0 or index == len(pending):
                    print(f"Processed {index}/{len(pending)} pending rows")


if __name__ == "__main__":
    main()
