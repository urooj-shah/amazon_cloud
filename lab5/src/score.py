import json
import os
import joblib
import numpy as np

model = None


def init():
    global model

    model_dir = os.getenv("AZUREML_MODEL_DIR")

    model_path = os.path.join(model_dir, "model_output", "model.pkl")

    model = joblib.load(model_path)


def run(raw_data):

    try:

        data = json.loads(raw_data)

        X = np.array(data["data"])

        predictions = model.predict(X)

        return {
            "predictions": predictions.tolist()
        }

    except Exception as e:

        return {
            "error": str(e)
        }