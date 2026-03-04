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

    # Determine TF-IDF dimension from the data
    max_index = 0
    for row in series:
        if row:
            max_index = max(max_index, max(idx for idx, _ in row))

    dim = max_index + 1

    X = np.zeros((len(series), dim), dtype=np.float32)

    for i, row in enumerate(series):
        for idx, val in row:
            X[i, idx] = val

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