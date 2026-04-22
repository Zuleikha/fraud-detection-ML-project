import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = "outputs/models/best_xgb.pkl"

app = FastAPI(title="Fraud Detection API")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


class Transaction(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        X = np.array(transaction.features).reshape(1, -1)
        prob = float(model.predict_proba(X)[0][1])
        label = int(prob >= 0.5)
        return {"fraud_probability": prob, "prediction": label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
