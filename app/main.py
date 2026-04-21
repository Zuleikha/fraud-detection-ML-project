from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app   = FastAPI(title="Fraud Detection API", version="1.0")
MODEL = joblib.load(Path("outputs/models/best_xgb.pkl"))
THRESHOLD = 0.28


class Transaction(BaseModel):
    # PCA components from the credit card dataset
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    log_amount: float
    hour_of_day: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction):
    df   = pd.DataFrame([txn.model_dump()])
    prob = float(MODEL.predict_proba(df)[:, 1][0])
    return {
        "fraud_probability": round(prob, 4),
        "is_fraud":          bool(prob >= THRESHOLD),
        "threshold_used":    THRESHOLD,
    }
