"""End-to-end recruitment fraud detection with TF-IDF, LSTM, and BERT.

The script downloads or reads the Real/Fake Job Posting dataset, preprocesses
job-post text, trains multiple classifiers, compares metrics, and exposes a
prediction function for new job descriptions.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import tf_keras
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import AutoTokenizer, BertConfig, TFAutoModelForSequenceClassification, TFBertForSequenceClassification


DATASET_URL = (
    "https://huggingface.co/datasets/victor/"
    "real-or-fake-fake-jobposting-prediction/resolve/main/fake_job_postings.csv"
)
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
TEXT_COLUMNS = [
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]
LABEL_COLUMN = "fraudulent"


@dataclass
class TrainingConfig:
    data_path: Optional[str] = None
    output_dir: str = "artifacts"
    test_size: float = 0.2
    validation_size: float = 0.1
    max_rows: Optional[int] = None
    seed: int = 42
    max_words: int = 20_000
    max_sequence_length: int = 220
    lstm_epochs: int = 3
    bert_epochs: int = 1
    batch_size: int = 32
    bert_batch_size: int = 8
    bert_model_name: str = "google/bert_uncased_L-2_H-128_A-2"
    bert_max_length: int = 192
    skip_bert: bool = False


@dataclass
class CliOptions:
    config: TrainingConfig
    predict: Optional[str] = None
    predict_model: str = "lstm"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def clean_text(text: object) -> str:
    """Normalize text with regex tokenization and stop-word removal."""
    text = "" if pd.isna(text) else str(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def combine_text_columns(df: pd.DataFrame) -> pd.Series:
    available = [col for col in TEXT_COLUMNS if col in df.columns]
    if not available:
        raise ValueError(f"Dataset must contain at least one of: {TEXT_COLUMNS}")
    return df[available].fillna("").astype(str).agg(" ".join, axis=1)


def load_job_postings(config: TrainingConfig) -> pd.DataFrame:
    df = read_dataset(config)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain a '{LABEL_COLUMN}' target column.")

    df = df.copy()
    df["raw_text"] = combine_text_columns(df)
    df["text"] = df["raw_text"].map(clean_text)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df = df[df["text"].str.len() > 0]

    if config.max_rows:
        real = df[df[LABEL_COLUMN] == 0]
        fake = df[df[LABEL_COLUMN] == 1]
        fake_n = min(len(fake), max(1, int(config.max_rows * 0.35)))
        real_n = min(len(real), config.max_rows - fake_n)
        df = pd.concat(
            [
                fake.sample(fake_n, random_state=config.seed),
                real.sample(real_n, random_state=config.seed),
            ]
        ).sample(frac=1, random_state=config.seed)

    return df.reset_index(drop=True)


def read_dataset(config: TrainingConfig) -> pd.DataFrame:
    if config.data_path:
        print(f"Loading dataset from {config.data_path}")
        return pd.read_csv(config.data_path)

    cache_path = Path("data") / "fake_job_postings.csv"
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Downloading dataset from {DATASET_URL}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            with requests.get(DATASET_URL, stream=True, timeout=60) as response:
                response.raise_for_status()
                with cache_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return pd.read_csv(cache_path)
        except Exception as exc:
            last_error = exc
            if cache_path.exists():
                cache_path.unlink()
            print(f"Download attempt {attempt} failed: {exc}")
            time.sleep(2 * attempt)

    if config.max_rows:
        print("Falling back to Hugging Face Dataset Viewer rows API for sampled training.")
        return read_dataset_viewer_rows(config.max_rows)

    raise RuntimeError(
        "Could not download the dataset. Pass --data-path with a local fake_job_postings.csv file."
    ) from last_error


def read_dataset_viewer_rows(max_rows: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    page_size = min(100, max_rows)
    for offset in range(0, max_rows, page_size):
        params = {
            "dataset": "victor/real-or-fake-fake-jobposting-prediction",
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": min(page_size, max_rows - offset),
        }
        response = requests.get(HF_ROWS_API, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        rows.extend(item["row"] for item in payload.get("rows", []))
        if len(rows) >= max_rows or len(payload.get("rows", [])) < page_size:
            break
    if not rows:
        raise RuntimeError("Dataset Viewer API did not return any rows.")
    return pd.DataFrame(rows)


def split_data(
    df: pd.DataFrame, config: TrainingConfig
) -> Tuple[pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    train_text, test_text, train_y, test_y = train_test_split(
        df["text"],
        df[LABEL_COLUMN].to_numpy(),
        test_size=config.test_size,
        stratify=df[LABEL_COLUMN],
        random_state=config.seed,
    )
    train_text, val_text, train_y, val_y = train_test_split(
        train_text,
        train_y,
        test_size=config.validation_size,
        stratify=train_y,
        random_state=config.seed,
    )
    return train_text, val_text, test_text, train_y, val_y, test_y


def class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    report = classification_report(
        y_true,
        y_pred,
        target_names=["real", "fake"],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "real_f1": report["real"]["f1-score"],
        "fake_f1": report["fake"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
    }
    print(f"\n{name} metrics")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    return metrics


def train_tfidf_baseline(
    train_text: Iterable[str], test_text: Iterable[str], train_y: np.ndarray, test_y: np.ndarray
) -> Tuple[TfidfVectorizer, LogisticRegression, Dict[str, object]]:
    vectorizer = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), min_df=2)
    x_train = vectorizer.fit_transform(train_text)
    x_test = vectorizer.transform(test_text)
    clf = LogisticRegression(max_iter=1_000, class_weight="balanced")
    clf.fit(x_train, train_y)
    pred = clf.predict(x_test)
    return vectorizer, clf, evaluate_model("TF-IDF + Logistic Regression", test_y, pred)


def build_lstm_model(vocab_size: int, max_sequence_length: int) -> tf.keras.Model:
    model = Sequential(
        [
            Embedding(vocab_size, 128),
            Bidirectional(LSTM(96, dropout=0.2, recurrent_dropout=0.2)),
            Dense(64, activation="relu"),
            Dropout(0.35),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_lstm(
    train_text: Iterable[str],
    val_text: Iterable[str],
    test_text: Iterable[str],
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    config: TrainingConfig,
) -> Tuple[Tokenizer, tf.keras.Model, Dict[str, object]]:
    tokenizer = Tokenizer(num_words=config.max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text)
    vocab_size = min(config.max_words, len(tokenizer.word_index) + 1)

    x_train = pad_sequences(
        tokenizer.texts_to_sequences(train_text), maxlen=config.max_sequence_length
    )
    x_val = pad_sequences(tokenizer.texts_to_sequences(val_text), maxlen=config.max_sequence_length)
    x_test = pad_sequences(
        tokenizer.texts_to_sequences(test_text), maxlen=config.max_sequence_length
    )

    model = build_lstm_model(vocab_size, config.max_sequence_length)
    model.fit(
        x_train,
        train_y,
        validation_data=(x_val, val_y),
        epochs=config.lstm_epochs,
        batch_size=config.batch_size,
        class_weight=class_weights(train_y),
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)],
        verbose=2,
    )
    pred = (model.predict(x_test, batch_size=config.batch_size).ravel() >= 0.5).astype(int)
    return tokenizer, model, evaluate_model("BiLSTM", test_y, pred)


def encode_for_bert(
    tokenizer: AutoTokenizer, texts: Iterable[str], max_length: int
) -> Dict[str, tf.Tensor]:
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="tf",
    )
    return dict(encodings)


def train_bert(
    train_text: Iterable[str],
    val_text: Iterable[str],
    test_text: Iterable[str],
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    config: TrainingConfig,
) -> Tuple[AutoTokenizer, tf.keras.Model, Dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
    try:
        model = TFAutoModelForSequenceClassification.from_pretrained(
            config.bert_model_name, num_labels=2, from_pt=False
        )
    except Exception as exc:
        print(
            "Pretrained TensorFlow BERT weights were not available for this checkpoint; "
            f"training a compact BERT model from scratch instead. Details: {exc}"
        )
        bert_config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
            num_labels=2,
        )
        model = TFBertForSequenceClassification(bert_config)
    optimizer = tf_keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[tf_keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    x_train = encode_for_bert(tokenizer, train_text, config.bert_max_length)
    x_val = encode_for_bert(tokenizer, val_text, config.bert_max_length)
    x_test = encode_for_bert(tokenizer, test_text, config.bert_max_length)

    model.fit(
        x_train,
        train_y,
        validation_data=(x_val, val_y),
        epochs=config.bert_epochs,
        batch_size=config.bert_batch_size,
        class_weight=class_weights(train_y),
        verbose=2,
    )
    logits = model.predict(x_test, batch_size=config.bert_batch_size).logits
    pred = np.argmax(logits, axis=1)
    return tokenizer, model, evaluate_model("BERT", test_y, pred)


def predict_job_posting(
    job_description: str,
    model_type: str = "lstm",
    artifacts_dir: str = "artifacts",
) -> Dict[str, object]:
    """Return fraud prediction for a new job description using saved artifacts."""
    artifacts = Path(artifacts_dir)
    cleaned = clean_text(job_description)

    if model_type == "lstm":
        with (artifacts / "lstm_tokenizer.pkl").open("rb") as f:
            tokenizer = pickle.load(f)
        model = tf.keras.models.load_model(artifacts / "lstm_model.keras")
        with (artifacts / "config.json").open("r", encoding="utf-8") as f:
            config = TrainingConfig(**json.load(f))
        seq = pad_sequences(
            tokenizer.texts_to_sequences([cleaned]), maxlen=config.max_sequence_length
        )
        fake_probability = float(model.predict(seq, verbose=0).ravel()[0])
    elif model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(artifacts / "bert_model")
        model = TFAutoModelForSequenceClassification.from_pretrained(artifacts / "bert_model")
        with (artifacts / "config.json").open("r", encoding="utf-8") as f:
            config = TrainingConfig(**json.load(f))
        encoded = encode_for_bert(tokenizer, [cleaned], config.bert_max_length)
        logits = model.predict(encoded, verbose=0).logits
        fake_probability = float(tf.nn.softmax(logits, axis=1).numpy()[0][1])
    else:
        raise ValueError("model_type must be 'lstm' or 'bert'.")

    return {
        "label": "fake" if fake_probability >= 0.5 else "real",
        "fake_probability": fake_probability,
        "real_probability": 1.0 - fake_probability,
    }


def save_artifacts(
    config: TrainingConfig,
    metrics: List[Dict[str, object]],
    tfidf_vectorizer: TfidfVectorizer,
    tfidf_model: LogisticRegression,
    lstm_tokenizer: Tokenizer,
    lstm_model: tf.keras.Model,
    bert_tokenizer: Optional[AutoTokenizer] = None,
    bert_model: Optional[tf.keras.Model] = None,
) -> None:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out / "tfidf_vectorizer.pkl").open("wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with (out / "tfidf_model.pkl").open("wb") as f:
        pickle.dump(tfidf_model, f)
    with (out / "lstm_tokenizer.pkl").open("wb") as f:
        pickle.dump(lstm_tokenizer, f)
    lstm_model.save(out / "lstm_model.keras")

    if bert_tokenizer is not None and bert_model is not None:
        bert_dir = out / "bert_model"
        bert_tokenizer.save_pretrained(bert_dir)
        bert_model.save_pretrained(bert_dir)


def run_training(config: TrainingConfig) -> List[Dict[str, object]]:
    set_seed(config.seed)
    df = load_job_postings(config)
    print(f"Dataset rows: {len(df):,}")
    print(df[LABEL_COLUMN].value_counts().rename(index={0: "real", 1: "fake"}))

    train_text, val_text, test_text, train_y, val_y, test_y = split_data(df, config)

    tfidf_vectorizer, tfidf_model, tfidf_metrics = train_tfidf_baseline(
        train_text, test_text, train_y, test_y
    )
    lstm_tokenizer, lstm_model, lstm_metrics = train_lstm(
        train_text, val_text, test_text, train_y, val_y, test_y, config
    )

    metrics = [tfidf_metrics, lstm_metrics]
    bert_tokenizer = None
    bert_model = None
    if not config.skip_bert:
        bert_tokenizer, bert_model, bert_metrics = train_bert(
            train_text, val_text, test_text, train_y, val_y, test_y, config
        )
        metrics.append(bert_metrics)

    save_artifacts(
        config,
        metrics,
        tfidf_vectorizer,
        tfidf_model,
        lstm_tokenizer,
        lstm_model,
        bert_tokenizer,
        bert_model,
    )

    print("\nModel comparison")
    comparison = pd.DataFrame(
        [
            {
                "model": m["model"],
                "accuracy": round(float(m["accuracy"]), 4),
                "fake_f1": round(float(m["fake_f1"]), 4),
                "macro_f1": round(float(m["macro_f1"]), 4),
            }
            for m in metrics
        ]
    )
    print(comparison.to_string(index=False))
    print(f"\nSaved artifacts to {Path(config.output_dir).resolve()}")
    return metrics


def parse_args() -> CliOptions:
    parser = argparse.ArgumentParser(description="Train job posting fraud detectors.")
    parser.add_argument("--data-path", default=None, help="Local CSV path. Defaults to HF download.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for trained artifacts.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smaller stratified sample.")
    parser.add_argument("--lstm-epochs", type=int, default=3)
    parser.add_argument("--bert-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bert-batch-size", type=int, default=8)
    parser.add_argument("--bert-model-name", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--skip-bert", action="store_true", help="Train TF-IDF and LSTM only.")
    parser.add_argument("--predict", default=None, help="Run prediction after training.")
    parser.add_argument("--predict-model", choices=["lstm", "bert"], default="lstm")
    args = parser.parse_args()
    values: Dict[str, Any] = vars(args)
    predict = values.pop("predict")
    predict_model = values.pop("predict_model")
    return CliOptions(config=TrainingConfig(**values), predict=predict, predict_model=predict_model)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    cli = parse_args()
    run_training(cli.config)
    if cli.predict:
        print("\nPrediction")
        print(
            json.dumps(
                predict_job_posting(cli.predict, cli.predict_model, cli.config.output_dir),
                indent=2,
            )
        )
