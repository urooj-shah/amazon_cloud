import argparse
import os
import time

import azureml.mlflow
import mlflow
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# Argument parsing
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_dataset(folder_path: str):
    return pd.read_parquet(folder_path)


# --------------------------------------------------
# Label creation
# --------------------------------------------------

def create_binary_labels(df):

    if "overall" not in df.columns:
        raise RuntimeError("Column 'overall' not found.")

    df = df[df["overall"].isin([1, 2, 4, 5])].copy()
    df["label"] = (df["overall"] >= 4).astype(int)

    return df


# --------------------------------------------------
# TF-IDF reconstruction
# --------------------------------------------------

def build_tfidf_matrix(series):
    """
    Accepts tfidf vectors stored either as:
      1) sparse list/array of (idx, val) pairs: [(12, 0.3), (98, 0.1)]
      2) dense vector: [0.0, 0.1, 0.0, ...] or np.ndarray([...])

    Any None/NaN/float values are treated as empty vectors.
    """
    def is_missing(x):
        if x is None:
            return True
        # pd.isna works for NaN and also for some pandas missing markers
        try:
            return bool(pd.isna(x))
        except Exception:
            return False

    # First pass: detect format and dimension
    max_index = -1
    dense_dim = None
    detected_dense = False
    detected_sparse = False

    for row in series:
        if is_missing(row):
            continue

        # If row is a single float, treat as missing/empty
        if isinstance(row, (float, np.floating)):
            continue

        arr = row
        # Convert to numpy array if it's a list-like
        if isinstance(row, (list, tuple, np.ndarray)):
            try:
                # If it looks like a dense numeric vector (1D numbers)
                a = np.asarray(row, dtype=np.float32)
                # If this succeeds and is 1D, it's probably dense
                if a.ndim == 1 and (len(a) > 0):
                    detected_dense = True
                    dense_dim = len(a) if dense_dim is None else max(dense_dim, len(a))
                    continue
            except Exception:
                pass

        # Otherwise try sparse: iterable of pairs
        try:
            for item in row:
                idx, val = item  # must be a 2-tuple
                detected_sparse = True
                if idx > max_index:
                    max_index = idx
        except Exception:
            # Unknown format: skip or raise
            # I'd raise to make data issues obvious:
            raise TypeError(f"Unsupported tfidf_vector row type: {type(row)} value={row}")

    # Decide which representation to use
    if detected_dense and not detected_sparse:
        dim = dense_dim or 0
        X = np.zeros((len(series), dim), dtype=np.float32)
        for i, row in enumerate(series):
            if is_missing(row) or isinstance(row, (float, np.floating)):
                continue
            v = np.asarray(row, dtype=np.float32).reshape(-1)
            X[i, :len(v)] = v
        return X

    # If sparse (or mixed), use sparse logic
    dim = max_index + 1
    if dim <= 0:
        return np.zeros((len(series), 0), dtype=np.float32)

    X = np.zeros((len(series), dim), dtype=np.float32)
    for i, row in enumerate(series):
        if is_missing(row) or isinstance(row, (float, np.floating)):
            continue
        try:
            for idx, val in row:
                X[i, idx] = val
        except Exception:
            # If mixed types exist, treat non-sparse rows as empty
            continue

    return X


# --------------------------------------------------
# Feature matrix
# --------------------------------------------------

def build_feature_matrix(df):

    # SBERT embeddings
    sbert_vectors = np.vstack(df["sbert_vector"].apply(np.array).values)

    # TF-IDF vectors
    tfidf_vectors = build_tfidf_matrix(df["tfidf_vector"])

    # Sentiment features
    sentiment_features = df[
        [
            "sentiment_pos",
            "sentiment_neg",
            "sentiment_neu",
            "sentiment_compound"
        ]
    ].values

    # Length features
    length_features = df[
        [
            "review_length_chars",
            "review_length_words"
        ]
    ].values

    # Combine all features
    X = np.hstack([
        sbert_vectors,
        tfidf_vectors,
        sentiment_features,
        length_features
    ])

    return X


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, X, y, split_name):

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    mlflow.log_metric(f"{split_name}_accuracy", accuracy)
    mlflow.log_metric(f"{split_name}_auc", auc)
    mlflow.log_metric(f"{split_name}_precision", precision)
    mlflow.log_metric(f"{split_name}_recall", recall)
    mlflow.log_metric(f"{split_name}_f1", f1)

    print("\n", split_name.upper(), "METRICS")
    print("Accuracy :", accuracy)
    print("AUC      :", auc)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1       :", f1)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    args = parse_args()

    start_time = time.time()

    print("Loading datasets...")

    train_df = load_dataset(args.train_data)
    val_df = load_dataset(args.val_data)
    test_df = load_dataset(args.test_data)

    print("Preparing labels...")

    train_df = create_binary_labels(train_df)
    val_df = create_binary_labels(val_df)
    test_df = create_binary_labels(test_df)

    print("Building feature matrices...")

    X_train = build_feature_matrix(train_df)
    y_train = train_df["label"].values

    X_val = build_feature_matrix(val_df)
    y_val = val_df["label"].values

    X_test = build_feature_matrix(test_df)
    y_test = test_df["label"].values

    print("\nDataset shapes")
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    # --------------------------------------------------
    # Log dataset info
    # --------------------------------------------------

    mlflow.log_param("model", "SGDClassifier_logistic")
    mlflow.log_param("feature_type", "sbert+tfidf+sentiment+length")
    mlflow.log_param("feature_dimension", int(X_train.shape[1]))
    mlflow.log_param("data_split", "60_15_15_10")
    mlflow.log_param("train_rows", int(X_train.shape[0]))
    mlflow.log_param("val_rows", int(X_val.shape[0]))
    mlflow.log_param("test_rows", int(X_test.shape[0]))

    # --------------------------------------------------
    # Model training
    # --------------------------------------------------

    print("\nTraining model...")

    model = SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        tol=1e-3,
        random_state=42
    )

    model.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    print("\nSaving model...")

    os.makedirs(args.output, exist_ok=True)

    model_path = os.path.join(args.output, "model.pkl")

    joblib.dump(model, model_path)

    mlflow.log_artifact(model_path)

    # --------------------------------------------------
    # Runtime
    # --------------------------------------------------

    runtime = time.time() - start_time

    mlflow.log_metric("training_runtime_seconds", runtime)

    print("\nTraining complete")
    print("Runtime:", runtime)


if __name__ == "__main__":
    main()