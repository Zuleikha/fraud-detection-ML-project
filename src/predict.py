"""
Prediction logic — loads the model once at startup, scores new transactions.

Separated from app/main.py so it can be unit-tested without starting a server.
"""
import joblib
import pandas as pd
import yaml
from pathlib import Path


_model = None
_config = None


def load_artifacts(config_path: str = "config/config.yaml"):
    global _model, _config
    with open(config_path) as f:
        _config = yaml.safe_load(f)
    _model = joblib.load(_config["model"]["path"])


def predict(transaction: dict) -> dict:
    """
    Score a single transaction dict.

    Returns:
        {
            "fraud_probability": float,   # raw model score
            "is_fraud": bool,             # True if prob >= threshold
            "threshold_used": float,
        }
    """
    if _model is None or _config is None:
        raise RuntimeError("Call load_artifacts() before predict()")

    df = pd.DataFrame([transaction])
    prob = float(_model.predict_proba(df)[:, 1][0])
    threshold = _config["prediction"]["threshold"]

    return {
        "fraud_probability": round(prob, 4),
        "is_fraud": prob >= threshold,
        "threshold_used": threshold,
    }
