#!/usr/bin/env python3
"""Build unified prompt-stability summary for ASB feature matrices."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


PROMPTS = {
    "strict": "Strict",
    "direct": "Direct",
    "inference": "Inference",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    feature_config: Path
    positive_label: str


def dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            key="sms_spam",
            display_name="SMS Spam",
            feature_config=Path("config/human20/sms_spam_human20_features.json"),
            positive_label="spam",
        ),
        DatasetSpec(
            key="sst2",
            display_name="SST-2",
            feature_config=Path("config/human20/sst2_human20_features.json"),
            positive_label="positive",
        ),
        DatasetSpec(
            key="ade",
            display_name="ADE",
            feature_config=Path("config/human20/ade_human20_features.json"),
            positive_label="ade",
        ),
        DatasetSpec(
            key="disaster_tweets",
            display_name="Disaster Tweets",
            feature_config=Path("config/human20/disaster_tweets_human20_features.json"),
            positive_label="real disaster",
        ),
    ]


def feature_file(features_dir: Path, spec: DatasetSpec, prompt: str) -> Path:
    return features_dir / f"{spec.key}_5000_human20_{prompt}.csv"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_feature_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [feature["id"] for feature in json.load(f)]


def sorted_by_id(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: row["id"])


def validate_rows(rows_by_prompt: dict[str, list[dict[str, str]]], feature_ids: list[str]) -> None:
    reference_prompt = next(iter(rows_by_prompt))
    reference_rows = sorted_by_id(rows_by_prompt[reference_prompt])
    reference_keys = [(row["id"], row["split"], row["label"]) for row in reference_rows]
    for prompt, rows in rows_by_prompt.items():
        if not rows:
            raise ValueError(f"{prompt}: no rows found")
        missing = {"id", "split", "label", *feature_ids}.difference(rows[0])
        if missing:
            raise ValueError(f"{prompt}: missing columns {sorted(missing)}")
        keys = [(row["id"], row["split"], row["label"]) for row in sorted_by_id(rows)]
        if keys != reference_keys:
            raise ValueError(f"{prompt}: rows do not match {reference_prompt}")
        for row in rows:
            for feature_id in feature_ids:
                if row[feature_id] not in {"0", "1"}:
                    raise ValueError(f"{prompt}: invalid value for {row['id']} {feature_id}: {row[feature_id]}")


def feature_matrix(rows: list[dict[str, str]], feature_ids: list[str]) -> np.ndarray:
    return np.array([[int(row[feature_id]) for feature_id in feature_ids] for row in rows], dtype=np.float32)


def y_values(rows: list[dict[str, str]], positive_label: str) -> list[int]:
    return [1 if row["label"] == positive_label else 0 for row in rows]


def train_lr_predictions_and_coefficients(
    rows: list[dict[str, str]],
    feature_ids: list[str],
    positive_label: str,
    random_state: int,
) -> tuple[list[str], list[int], dict[str, float]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = sorted_by_id([row for row in rows if row["split"] == "test"])
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    model.fit(feature_matrix(train_rows, feature_ids), y_values(train_rows, positive_label))
    predictions = model.predict(feature_matrix(test_rows, feature_ids)).astype(int).tolist()
    coefficients = {feature_id: float(coef) for feature_id, coef in zip(feature_ids, model.coef_[0])}
    return [row["id"] for row in test_rows], predictions, coefficients


def matrix_agreement(left_rows: list[dict[str, str]], right_rows: list[dict[str, str]], feature_ids: list[str]) -> float:
    left_matrix = feature_matrix(sorted_by_id(left_rows), feature_ids)
    right_matrix = feature_matrix(sorted_by_id(right_rows), feature_ids)
    return float(np.mean(left_matrix == right_matrix))


def per_feature_kappa_and_positive_agreement(
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
    feature_ids: list[str],
) -> tuple[float, float, int]:
    left_matrix = feature_matrix(sorted_by_id(left_rows), feature_ids).astype(int)
    right_matrix = feature_matrix(sorted_by_id(right_rows), feature_ids).astype(int)
    kappas: list[float] = []
    positive_agreements: list[float] = []
    for column in range(len(feature_ids)):
        left = left_matrix[:, column]
        right = right_matrix[:, column]
        if int(left.sum() + right.sum()) == 0:
            continue
        observed = float(np.mean(left == right))
        left_yes = float(np.mean(left == 1))
        right_yes = float(np.mean(right == 1))
        expected = left_yes * right_yes + (1.0 - left_yes) * (1.0 - right_yes)
        if expected < 1.0:
            kappas.append((observed - expected) / (1.0 - expected))
        tp = int(np.sum((left == 1) & (right == 1)))
        fp = int(np.sum((left == 0) & (right == 1)))
        fn = int(np.sum((left == 1) & (right == 0)))
        denominator = 2 * tp + fp + fn
        if denominator > 0:
            positive_agreements.append((2 * tp) / denominator)
    if not kappas or not positive_agreements:
        return 0.0, 0.0, 0
    return float(np.mean(kappas)), float(np.mean(positive_agreements)), len(positive_agreements)


def activation_shifts(
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
    feature_ids: list[str],
) -> tuple[float, float]:
    left_matrix = feature_matrix(sorted_by_id(left_rows), feature_ids)
    right_matrix = feature_matrix(sorted_by_id(right_rows), feature_ids)
    shifts = np.abs(left_matrix.mean(axis=0) - right_matrix.mean(axis=0))
    return float(np.mean(shifts)), float(np.max(shifts))


def prediction_agreement(left_predictions: list[int], right_predictions: list[int]) -> float:
    return float(np.mean(np.array(left_predictions) == np.array(right_predictions)))


def top_positive_features(coefficients: dict[str, float], k: int) -> list[str]:
    return [
        feature_id
        for feature_id, coefficient in sorted(coefficients.items(), key=lambda item: item[1], reverse=True)
        if coefficient > 0
    ][:k]


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="output/features/unified_human20")
    parser.add_argument("--output", default="paper_tables/table_prompt_stability_summary.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dataset", choices=["all", "sms_spam", "sst2", "ade", "disaster_tweets"], default="all")
    args = parser.parse_args()

    specs = dataset_specs()
    if args.dataset != "all":
        specs = [spec for spec in specs if spec.key == args.dataset]

    rows: list[dict[str, object]] = []
    for spec in specs:
        feature_ids = load_feature_ids(spec.feature_config)
        files = {prompt: feature_file(Path(args.features_dir), spec, prompt) for prompt in PROMPTS}
        missing = [str(path) for path in files.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{spec.display_name}: missing prompt feature files: {missing}")
        rows_by_prompt = {prompt: read_rows(path) for prompt, path in files.items()}
        validate_rows(rows_by_prompt, feature_ids)

        predictions_by_prompt: dict[str, list[int]] = {}
        coefficients_by_prompt: dict[str, dict[str, float]] = {}
        test_ids_by_prompt: dict[str, list[str]] = {}
        for prompt, prompt_rows in rows_by_prompt.items():
            test_ids, predictions, coefficients = train_lr_predictions_and_coefficients(
                prompt_rows,
                feature_ids,
                spec.positive_label,
                args.random_state,
            )
            test_ids_by_prompt[prompt] = test_ids
            predictions_by_prompt[prompt] = predictions
            coefficients_by_prompt[prompt] = coefficients

        for left_prompt, right_prompt in itertools.combinations(PROMPTS, 2):
            if test_ids_by_prompt[left_prompt] != test_ids_by_prompt[right_prompt]:
                raise ValueError(f"{spec.display_name}: test ids differ for {left_prompt} and {right_prompt}")
            mean_shift, max_shift = activation_shifts(
                rows_by_prompt[left_prompt],
                rows_by_prompt[right_prompt],
                feature_ids,
            )
            mean_kappa, mean_positive_agreement, active_features = per_feature_kappa_and_positive_agreement(
                rows_by_prompt[left_prompt],
                rows_by_prompt[right_prompt],
                feature_ids,
            )
            left_top = set(top_positive_features(coefficients_by_prompt[left_prompt], args.top_k))
            right_top = set(top_positive_features(coefficients_by_prompt[right_prompt], args.top_k))
            rows.append(
                {
                    "Dataset": spec.display_name,
                    "Prompt comparison": f"{PROMPTS[left_prompt]} vs {PROMPTS[right_prompt]}",
                    "Matrix agreement": fmt(matrix_agreement(rows_by_prompt[left_prompt], rows_by_prompt[right_prompt], feature_ids)),
                    "Mean per-feature kappa": fmt(mean_kappa),
                    "Mean positive agreement": fmt(mean_positive_agreement),
                    "Active features": active_features,
                    "Mean activation shift": fmt(mean_shift),
                    "Max activation shift": fmt(max_shift),
                    "Prediction agreement": fmt(
                        prediction_agreement(predictions_by_prompt[left_prompt], predictions_by_prompt[right_prompt])
                    ),
                    "Top-5 overlap": fmt(len(left_top.intersection(right_top)) / args.top_k),
                }
            )

    write_csv(
        Path(args.output),
        rows,
        [
            "Dataset",
            "Prompt comparison",
            "Matrix agreement",
            "Mean per-feature kappa",
            "Mean positive agreement",
            "Active features",
            "Mean activation shift",
            "Max activation shift",
            "Prediction agreement",
            "Top-5 overlap",
        ],
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
