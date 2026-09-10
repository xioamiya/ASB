#!/usr/bin/env python3
"""Train-only mutual-information feature selection sensitivity for ASB."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    feature_file: Path
    feature_config: Path
    positive_label: str


def dataset_specs(features_dir: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            key="sms_spam",
            display_name="SMS Spam",
            feature_file=features_dir / "sms_spam_5000_human20_strict.csv",
            feature_config=Path("config/human20/sms_spam_human20_features.json"),
            positive_label="spam",
        ),
        DatasetSpec(
            key="sst2",
            display_name="SST-2",
            feature_file=features_dir / "sst2_5000_human20_strict.csv",
            feature_config=Path("config/human20/sst2_human20_features.json"),
            positive_label="positive",
        ),
        DatasetSpec(
            key="ade",
            display_name="ADE",
            feature_file=features_dir / "ade_5000_human20_strict.csv",
            feature_config=Path("config/human20/ade_human20_features.json"),
            positive_label="ade",
        ),
        DatasetSpec(
            key="disaster_tweets",
            display_name="Disaster Tweets",
            feature_file=features_dir / "disaster_tweets_5000_human20_strict.csv",
            feature_config=Path("config/human20/disaster_tweets_human20_features.json"),
            positive_label="real disaster",
        ),
    ]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_feature_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [feature["id"] for feature in json.load(f)]


def validate(rows: list[dict[str, str]], feature_ids: list[str], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    missing = {"id", "split", "label", *feature_ids}.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    for row in rows:
        for feature_id in feature_ids:
            if row[feature_id] not in {"0", "1"}:
                raise ValueError(f"{path}: invalid value for {row['id']} {feature_id}: {row[feature_id]}")


def y_values(rows: list[dict[str, str]], positive_label: str) -> list[int]:
    return [1 if row["label"] == positive_label else 0 for row in rows]


def feature_matrix(rows: list[dict[str, str]], feature_ids: list[str]) -> list[list[int]]:
    return [[int(row[feature_id]) for feature_id in feature_ids] for row in rows]


def evaluate_feature_set(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_ids: list[str],
    positive_label: str,
    random_state: int,
) -> dict[str, float]:
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    model.fit(feature_matrix(train_rows, feature_ids), y_values(train_rows, positive_label))
    y_test = y_values(test_rows, positive_label)
    y_pred = model.predict(feature_matrix(test_rows, feature_ids)).tolist()
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro-F1": f1_score(y_test, y_pred, average="macro"),
        "Positive F1": f1_score(y_test, y_pred, pos_label=1),
    }


def mi_ranking(
    train_rows: list[dict[str, str]],
    feature_ids: list[str],
    positive_label: str,
    random_state: int,
) -> list[tuple[str, float]]:
    scores = mutual_info_classif(
        feature_matrix(train_rows, feature_ids),
        y_values(train_rows, positive_label),
        discrete_features=True,
        random_state=random_state,
    )
    return sorted(zip(feature_ids, scores.tolist()), key=lambda item: (-item[1], item[0]))


def metric_row(
    dataset: str,
    feature_set: str,
    feature_ids: list[str],
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "Dataset": dataset,
        "Feature set": feature_set,
        "#Features": len(feature_ids),
        "Accuracy": f"{metrics['Accuracy']:.4f}",
        "Macro-F1": f"{metrics['Macro-F1']:.4f}",
        "Positive F1": f"{metrics['Positive F1']:.4f}",
        "Selected features": "; ".join(feature_ids),
    }


def run_dataset(
    spec: DatasetSpec,
    top_ks: list[int],
    random_state: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    feature_ids = load_feature_ids(spec.feature_config)
    rows = read_rows(spec.feature_file)
    validate(rows, feature_ids, spec.feature_file)

    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    ranking = mi_ranking(train_rows, feature_ids, spec.positive_label, random_state)
    ranked_feature_ids = [feature_id for feature_id, _ in ranking]

    table_rows = [
        metric_row(
            spec.display_name,
            "All ASB features",
            feature_ids,
            evaluate_feature_set(train_rows, test_rows, feature_ids, spec.positive_label, random_state),
        )
    ]
    for top_k in top_ks:
        if top_k >= len(feature_ids):
            continue
        selected = ranked_feature_ids[:top_k]
        metrics = evaluate_feature_set(train_rows, test_rows, selected, spec.positive_label, random_state)
        table_rows.append(metric_row(spec.display_name, f"MI top-{top_k}", selected, metrics))

    ranking_rows = [
        {
            "Dataset": spec.display_name,
            "Rank": rank,
            "Feature": feature_id,
            "Mutual information": f"{score:.6f}",
        }
        for rank, (feature_id, score) in enumerate(ranking, start=1)
    ]
    return table_rows, ranking_rows


def parse_top_ks(raw_values: list[str]) -> list[int]:
    values: list[int] = []
    for raw_value in raw_values:
        for item in raw_value.split(","):
            item = item.strip()
            if item:
                values.append(int(item))
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="output/features/unified_human20")
    parser.add_argument("--output", default="paper_tables/table_mi_feature_selection_sensitivity.csv")
    parser.add_argument(
        "--details-output",
        default="output/results/unified_mi_feature_selection_sensitivity_details.csv",
    )
    parser.add_argument("--ranking-output", default="output/results/unified_mi_feature_selection_ranking.csv")
    parser.add_argument("--top-k", nargs="+", default=["5", "10", "15"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dataset", choices=["all", "sms_spam", "sst2", "ade", "disaster_tweets"], default="all")
    args = parser.parse_args()

    specs = dataset_specs(Path(args.features_dir))
    if args.dataset != "all":
        specs = [spec for spec in specs if spec.key == args.dataset]

    top_ks = parse_top_ks(args.top_k)
    table_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    for spec in specs:
        dataset_rows, dataset_ranking_rows = run_dataset(spec, top_ks, args.random_state)
        table_rows.extend(dataset_rows)
        ranking_rows.extend(dataset_ranking_rows)

    write_csv(
        Path(args.output),
        table_rows,
        ["Dataset", "Feature set", "#Features", "Accuracy", "Macro-F1", "Positive F1"],
    )
    write_csv(
        Path(args.details_output),
        table_rows,
        ["Dataset", "Feature set", "#Features", "Accuracy", "Macro-F1", "Positive F1", "Selected features"],
    )
    write_csv(
        Path(args.ranking_output),
        ranking_rows,
        ["Dataset", "Rank", "Feature", "Mutual information"],
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.details_output}")
    print(f"Wrote {args.ranking_output}")


if __name__ == "__main__":
    main()
