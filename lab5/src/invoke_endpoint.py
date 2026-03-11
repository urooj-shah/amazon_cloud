from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd
import json
import requests
import numpy as np
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# Azure ML workspace connection
# --------------------------------------------------

subscription_id = "a485bb50-61aa-4b2f-bc7f-b6b53539b9d3"
resource_group = "Project"
workspace_name = "60300832-test"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name
)

# --------------------------------------------------
# Load deploy dataset from Azure ML Data Asset
# --------------------------------------------------

data_asset = ml_client.data.get(
    name="amazon_review_merged_features_deploy",
    label="latest"
)

deploy_path = data_asset.path
df = pd.read_parquet(deploy_path)

# --------------------------------------------------
# Create binary labels (same rule as training)
# --------------------------------------------------

df = df[df["overall"].isin([1, 2, 4, 5])].copy()
df["label"] = (df["overall"] >= 4).astype(int)

y_true = df["label"].values

# --------------------------------------------------
# Feature matrix (same as train.py)
# --------------------------------------------------

def build_tfidf_matrix(series):
    return np.vstack(series.apply(np.array).values)

def build_feature_matrix(df):

    sbert_vectors = np.vstack(df["sbert_vector"].apply(np.array).values)

    tfidf_vectors = build_tfidf_matrix(df["tfidf_vector"])

    sentiment_features = df[
        ["sentiment_pos", "sentiment_neg", "sentiment_neu", "sentiment_compound"]
    ].values

    length_features = df[
        ["review_length_chars", "review_length_words"]
    ].values

    return np.hstack([
        sbert_vectors,
        tfidf_vectors,
        sentiment_features,
        length_features
    ])

X = build_feature_matrix(df)

# --------------------------------------------------
# Endpoint information
# --------------------------------------------------

ENDPOINT_URL = "https://amazon-review-endpoint.qatarcentral.inference.ml.azure.com/score"

API_KEY = "super secret key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# --------------------------------------------------
# Send requests in batches
# --------------------------------------------------

batch_size = 100
predictions = []

for i in range(0, len(X), batch_size):

    batch = X[i:i + batch_size]

    payload = {"data": batch.tolist()}

    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        data=json.dumps(payload)
    )

    preds = response.json()["predictions"]

    predictions.extend(preds)

# --------------------------------------------------
# Compute deployment accuracy
# --------------------------------------------------

y_pred = np.array(predictions[:len(y_true)])

deploy_accuracy = accuracy_score(y_true[:len(y_pred)], y_pred)

print("Deployment accuracy:", deploy_accuracy)