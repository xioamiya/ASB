#!/usr/bin/env python3
"""Train/evaluate unified method table rows and summarize over seeds."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    data_path: Path
    feature_path: Path
    positive_label: str
    labels: tuple[str, str]


def dataset_specs(features_dir: Path) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            key="sms_spam",
            display_name="SMS Spam",
            data_path=Path("data/sms_spam_5000.csv"),
            feature_path=features_dir / "sms_spam_5000_human20_strict.csv",
            positive_label="spam",
            labels=("ham", "spam"),
        ),
        DatasetSpec(
            key="sst2",
            display_name="SST-2",
            data_path=Path("data/sst2_5000.csv"),
            feature_path=features_dir / "sst2_5000_human20_strict.csv",
            positive_label="positive",
            labels=("negative", "positive"),
        ),
        DatasetSpec(
            key="ade",
            display_name="ADE",
            data_path=Path("data/ade_5000.csv"),
            feature_path=features_dir / "ade_5000_human20_strict.csv",
            positive_label="ade",
            labels=("non_ade", "ade"),
        ),
        DatasetSpec(
            key="disaster_tweets",
            display_name="Disaster Tweets",
            data_path=Path("data/disaster_5000.csv"),
            feature_path=features_dir / "disaster_tweets_5000_human20_strict.csv",
            positive_label="real disaster",
            labels=("not disaster", "real disaster"),
        ),
    ]


def feature_config_for_key(key: str) -> Path:
    return {
        "sms_spam": Path("config/human20/sms_spam_human20_features.json"),
        "sst2": Path("config/human20/sst2_human20_features.json"),
        "ade": Path("config/human20/ade_human20_features.json"),
        "disaster_tweets": Path("config/human20/disaster_tweets_human20_features.json"),
    }[key]


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def y_values(rows: list[dict[str, str]], positive_label: str) -> list[int]:
    return [1 if row["label"] == positive_label else 0 for row in rows]


def feature_matrix(rows: list[dict[str, str]], feature_ids: list[str]) -> np.ndarray:
    return np.array([[int(row[feature_id]) for feature_id in feature_ids] for row in rows], dtype=np.float32)


def metric_values(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=1)),
    }


def metric_row(
    *,
    dataset: str,
    model_input: str,
    method: str,
    seed: int,
    y_true: list[int],
    y_pred: list[int],
) -> dict[str, object]:
    metrics = metric_values(y_true, y_pred)
    return {
        "Dataset": dataset,
        "Input": model_input,
        "Method": method,
        "seed": seed,
        "Accuracy": metrics["accuracy"],
        "Macro-F1": metrics["macro_f1"],
        "Positive F1": metrics["positive_f1"],
    }


def validate_feature_rows(rows: list[dict[str, str]], feature_ids: list[str]) -> None:
    if not rows:
        raise ValueError("No feature rows found")
    required = {"id", "split", "label", "text", *feature_ids}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for row in rows:
        for feature_id in feature_ids:
            if row[feature_id] not in {"0", "1"}:
                raise ValueError(f"Invalid feature value for {row['id']} {feature_id}: {row[feature_id]}")


def train_bert(
    *,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    positive_label: str,
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    seed: int,
) -> list[int]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    class TextDataset(Dataset):
        def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            item = {key: value[index] for key, value in self.encodings.items()}
            item["labels"] = self.labels[index]
            return item

    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        local_files_only=True,
    ).to(device)

    train_dataset = TextDataset(
        [row["text"] for row in train_rows],
        y_values(train_rows, positive_label),
        tokenizer,
        max_length,
    )
    test_dataset = TextDataset(
        [row["text"] for row in test_rows],
        y_values(test_rows, positive_label),
        tokenizer,
        max_length,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += float(outputs.loss.detach().cpu())
        print(f"BERT seed={seed} epoch {epoch}/{epochs}: loss={total_loss / max(len(train_loader), 1):.4f}")

    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.detach().cpu()
            predictions.extend(torch.argmax(logits, dim=1).tolist())
    return predictions


def encode_transformer_embeddings(
    *,
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked_hidden = hidden * attention_mask
            summed = masked_hidden.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
            embeddings.append(pooled.detach().cpu().numpy())
    return np.vstack(embeddings).astype(np.float32)


def evaluate_llm_predictions(path: Path, positive_label: str) -> tuple[list[int], list[int]]:
    rows = read_rows(path)
    y_true = [1 if row["label"] == positive_label else 0 for row in rows]
    y_pred = [1 if row["prediction"] == positive_label else 0 for row in rows]
    return y_true, y_pred


def run_dataset(
    *,
    spec: DatasetSpec,
    seeds: list[int],
    llm_predictions_dir: Path,
    skip_bert: bool,
    require_llm_predictions: bool,
    bert_model: str,
    bert_epochs: int,
    bert_batch_size: int,
    bert_learning_rate: float,
    bert_max_length: int,
    include_embedding_lr: bool,
    embedding_model: str,
    embedding_batch_size: int,
    embedding_max_length: int,
) -> list[dict[str, object]]:
    raw_rows = read_rows(spec.data_path)
    feature_ids = load_feature_ids(feature_config_for_key(spec.key))
    feature_rows = read_rows(spec.feature_path)
    validate_feature_rows(feature_rows, feature_ids)

    train_rows = [row for row in raw_rows if row["split"] == "train"]
    test_rows = [row for row in raw_rows if row["split"] == "test"]
    feature_train_rows = [row for row in feature_rows if row["split"] == "train"]
    feature_test_rows = [row for row in feature_rows if row["split"] == "test"]
    y_train = y_values(train_rows, spec.positive_label)
    y_test = y_values(test_rows, spec.positive_label)
    y_feature_train = y_values(feature_train_rows, spec.positive_label)
    y_feature_test = y_values(feature_test_rows, spec.positive_label)
    train_texts = [row["text"] for row in train_rows]
    test_texts = [row["text"] for row in test_rows]
    x_train_features = feature_matrix(feature_train_rows, feature_ids)
    x_test_features = feature_matrix(feature_test_rows, feature_ids)
    x_train_embeddings: np.ndarray | None = None
    x_test_embeddings: np.ndarray | None = None
    if include_embedding_lr:
        print(f"Encoding frozen transformer embeddings for {spec.display_name}: {embedding_model}")
        all_embeddings = encode_transformer_embeddings(
            texts=train_texts + test_texts,
            model_name=embedding_model,
            batch_size=embedding_batch_size,
            max_length=embedding_max_length,
        )
        x_train_embeddings = all_embeddings[: len(train_texts)]
        x_test_embeddings = all_embeddings[len(train_texts) :]

    rows: list[dict[str, object]] = []
    for seed in seeds:
        print(f"Running {spec.display_name} seed={seed}")
        set_seed(seed)

        tfidf_lr = Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
                ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
            ]
        )
        tfidf_lr.fit(train_texts, y_train)
        rows.append(
            metric_row(
                dataset=spec.display_name,
                model_input="raw text",
                method="TF-IDF + LR",
                seed=seed,
                y_true=y_test,
                y_pred=tfidf_lr.predict(test_texts).tolist(),
            )
        )

        tfidf_svm = Pipeline(
            [
                ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)),
                ("svm", LinearSVC(class_weight="balanced", dual="auto", random_state=seed, max_iter=5000)),
            ]
        )
        tfidf_svm.fit(train_texts, y_train)
        rows.append(
            metric_row(
                dataset=spec.display_name,
                model_input="raw text",
                method="TF-IDF + SVM",
                seed=seed,
                y_true=y_test,
                y_pred=tfidf_svm.predict(test_texts).tolist(),
            )
        )

        if include_embedding_lr:
            if x_train_embeddings is None or x_test_embeddings is None:
                raise RuntimeError("Embedding matrices were not computed")
            embedding_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
            embedding_lr.fit(x_train_embeddings, y_train)
            rows.append(
                metric_row(
                    dataset=spec.display_name,
                    model_input="raw text",
                    method="Frozen DistilBERT embedding + LR",
                    seed=seed,
                    y_true=y_test,
                    y_pred=embedding_lr.predict(x_test_embeddings).tolist(),
                )
            )

        if not skip_bert:
            bert_pred = train_bert(
                train_rows=train_rows,
                test_rows=test_rows,
                positive_label=spec.positive_label,
                model_name=bert_model,
                epochs=bert_epochs,
                batch_size=bert_batch_size,
                learning_rate=bert_learning_rate,
                max_length=bert_max_length,
                seed=seed,
            )
            rows.append(
                metric_row(
                    dataset=spec.display_name,
                    model_input="raw text",
                    method="DistilBERT fine-tuning",
                    seed=seed,
                    y_true=y_test,
                    y_pred=bert_pred,
                )
            )

        zero_path = llm_predictions_dir / f"{spec.key}_zero_shot.csv"
        if zero_path.exists():
            zero_true, zero_pred = evaluate_llm_predictions(zero_path, spec.positive_label)
            rows.append(
                metric_row(
                    dataset=spec.display_name,
                    model_input="raw text",
                    method="LLM zero-shot",
                    seed=seed,
                    y_true=zero_true,
                    y_pred=zero_pred,
                )
            )
        elif require_llm_predictions:
            raise FileNotFoundError(f"Missing zero-shot predictions: {zero_path}")

        four_shot_path = llm_predictions_dir / f"{spec.key}_4shot_seed{seed}.csv"
        if four_shot_path.exists():
            four_true, four_pred = evaluate_llm_predictions(four_shot_path, spec.positive_label)
            rows.append(
                metric_row(
                    dataset=spec.display_name,
                    model_input="raw text",
                    method="LLM 4-shot",
                    seed=seed,
                    y_true=four_true,
                    y_pred=four_pred,
                )
            )
        elif require_llm_predictions:
            raise FileNotFoundError(f"Missing 4-shot predictions: {four_shot_path}")

        llm_lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        llm_lr.fit(x_train_features, y_feature_train)
        rows.append(
            metric_row(
                dataset=spec.display_name,
                model_input="ASB features",
                method="ASB-LR (ours)",
                seed=seed,
                y_true=y_feature_test,
                y_pred=llm_lr.predict(x_test_features).tolist(),
            )
        )

        negative_count = y_feature_train.count(0)
        positive_count = y_feature_train.count(1)
        xgb = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negative_count / positive_count,
            random_state=seed,
            n_jobs=2,
        )
        xgb.fit(x_train_features, y_feature_train)
        rows.append(
            metric_row(
                dataset=spec.display_name,
                model_input="ASB features",
                method="ASB-XGB (ours)",
                seed=seed,
                y_true=y_feature_test,
                y_pred=xgb.predict(x_test_features).astype(int).tolist(),
            )
        )
    return rows


def fmt_mean_std(values: list[float]) -> str:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    order = [
        ("raw text", "TF-IDF + LR"),
        ("raw text", "TF-IDF + SVM"),
        ("raw text", "Frozen DistilBERT embedding + LR"),
        ("raw text", "DistilBERT fine-tuning"),
        ("raw text", "LLM zero-shot"),
        ("raw text", "LLM 4-shot"),
        ("ASB features", "ASB-LR (ours)"),
        ("ASB features", "ASB-XGB (ours)"),
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["Dataset"]), str(row["Input"]), str(row["Method"])), []).append(row)

    summary_rows: list[dict[str, object]] = []
    dataset_order = ["SMS Spam", "SST-2", "ADE", "Disaster Tweets"]
    for dataset in dataset_order:
        for model_input, method in order:
            values = grouped.get((dataset, model_input, method))
            if not values:
                summary_rows.append(
                    {
                        "Dataset": dataset,
                        "Input": model_input,
                        "Method": method,
                        "Accuracy": "",
                        "Macro-F1": "",
                        "Positive F1": "",
                    }
                )
                continue
            summary_rows.append(
                {
                    "Dataset": dataset,
                    "Input": model_input,
                    "Method": method,
                    "Accuracy": fmt_mean_std([float(row["Accuracy"]) for row in values]),
                    "Macro-F1": fmt_mean_std([float(row["Macro-F1"]) for row in values]),
                    "Positive F1": fmt_mean_std([float(row["Positive F1"]) for row in values]),
                }
            )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="output/features/unified_human20")
    parser.add_argument("--llm-predictions-dir", default="output/llm_predictions/unified_human20")
    parser.add_argument("--output", default="paper_tables/table_unified_method_performance.csv")
    parser.add_argument("--per-seed-output", default="output/results/unified_method_performance_by_seed.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--dataset", choices=["all", "sms_spam", "sst2", "ade", "disaster_tweets"], default="all")
    parser.add_argument("--require-llm-predictions", action="store_true")
    parser.add_argument("--skip-bert", action="store_true")
    parser.add_argument("--bert-model", default="distilbert-base-uncased")
    parser.add_argument("--bert-epochs", type=int, default=3)
    parser.add_argument("--bert-batch-size", type=int, default=16)
    parser.add_argument("--bert-learning-rate", type=float, default=2e-5)
    parser.add_argument("--bert-max-length", type=int, default=128)
    parser.add_argument("--include-embedding-lr", action="store_true")
    parser.add_argument("--embedding-model", default="distilbert-base-uncased")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--embedding-max-length", type=int, default=128)
    args = parser.parse_args()

    specs = dataset_specs(Path(args.features_dir))
    if args.dataset != "all":
        specs = [spec for spec in specs if spec.key == args.dataset]

    per_seed_rows: list[dict[str, object]] = []
    for spec in specs:
        per_seed_rows.extend(
            run_dataset(
                spec=spec,
                seeds=args.seeds,
                llm_predictions_dir=Path(args.llm_predictions_dir),
                skip_bert=args.skip_bert,
                require_llm_predictions=args.require_llm_predictions,
                bert_model=args.bert_model,
                bert_epochs=args.bert_epochs,
                bert_batch_size=args.bert_batch_size,
                bert_learning_rate=args.bert_learning_rate,
                bert_max_length=args.bert_max_length,
                include_embedding_lr=args.include_embedding_lr,
                embedding_model=args.embedding_model,
                embedding_batch_size=args.embedding_batch_size,
                embedding_max_length=args.embedding_max_length,
            )
        )

    write_csv(
        Path(args.per_seed_output),
        per_seed_rows,
        ["Dataset", "Input", "Method", "seed", "Accuracy", "Macro-F1", "Positive F1"],
    )
    summary_rows = summarize(per_seed_rows)
    write_csv(
        Path(args.output),
        summary_rows,
        ["Dataset", "Input", "Method", "Accuracy", "Macro-F1", "Positive F1"],
    )
    print(f"Wrote {args.per_seed_output}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
