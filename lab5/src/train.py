import argparse
import os
import time
import glob

import azureml.mlflow  #  REQUIRED: activates Azure ML MLflow integration
import mlflow
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Argument parsing
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_data", type=str, required=True)
    parser.add_argument("--label_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


# --------------------------------------------------
# Feature utilities
# --------------------------------------------------
def build_features(df):
    # SBERT embeddings are fixed-length
    return np.vstack(df["sbert_vector"].values)


# --------------------------------------------------
# Main training logic
# --------------------------------------------------
def main():
    args = parse_args()
    start_time = time.time()

    # --------------------------------------------------
    # Load merged feature data
    # --------------------------------------------------
    feature_df = pd.read_parquet(
        os.path.join(args.feature_data, "data.parquet")
    )

    # --------------------------------------------------
    # Load label data (multiple parquet shards)
    # --------------------------------------------------
    label_files = glob.glob(os.path.join(args.label_data, "*.parquet"))
    if not label_files:
        raise RuntimeError("No parquet files found in label_data")

    label_df = pd.concat(
        (
            pd.read_parquet(f, columns=["asin", "reviewerID", "overall"])
            for f in label_files
        ),
        ignore_index=True,
    )

    # --------------------------------------------------
    # Join features + labels
    # --------------------------------------------------
    df = feature_df.merge(
        label_df,
        on=["asin", "reviewerID"],
        how="inner",
    )

    # --------------------------------------------------
    # Sample to avoid OOM (critical)
    # --------------------------------------------------
    MAX_ROWS = 100_000
    df = df.sample(n=min(len(df), MAX_ROWS), random_state=42)

    # --------------------------------------------------
    # Binary labels
    # --------------------------------------------------
    df = df[df["overall"].isin([1, 2, 4, 5])].copy()
    df["label"] = (df["overall"] >= 4).astype(int)

    # --------------------------------------------------
    # Feature matrix
    # --------------------------------------------------
    X = build_features(df)
    y = df["label"].values

    # --------------------------------------------------
    # Train / test split
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # Model training
    # --------------------------------------------------
    model = SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        tol=1e-3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc", auc)

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    # --------------------------------------------------
    # Runtime metric
    # --------------------------------------------------
    runtime = time.time() - start_time
    mlflow.log_metric("training_runtime_seconds", runtime)

    print("Training complete.")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Runtime (s): {runtime:.2f}")


if __name__ == "__main__":
    main()