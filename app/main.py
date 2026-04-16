"""
FastAPI serving layer for the fraud detection model.

Lifecycle:
  1. App starts → load_artifacts() loads model + config into memory once.
  2. POST /predict → extract features from request, call predict(), return result.
  3. GET /health   → load balancer / Render health check.

Run locally:
    uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import load_artifacts, predict


# --- Request / response schemas -----------------------------------------

class Transaction(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in EUR")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    transactions_last_1h: int = Field(0, ge=0)
    transactions_last_24h: int = Field(0, ge=0)
    amount_mean_last_24h: float = Field(0.0, ge=0)
    amount_std_last_24h: float = Field(0.0, ge=0)
    merchant_category: str
    card_type: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 249.99,
                "hour_of_day": 2,
                "day_of_week": 5,
                "transactions_last_1h": 3,
                "transactions_last_24h": 7,
                "amount_mean_last_24h": 55.0,
                "amount_std_last_24h": 30.0,
                "merchant_category": "electronics",
                "card_type": "credit",
            }
        }
    }


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold_used: float


# --- App lifecycle -------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model once so every request reuses the same object.
    # In production this avoids a ~200ms disk read on every call.
    load_artifacts()
    yield
    # Shutdown: nothing to clean up for a simple joblib model.


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description="Scores transactions and flags likely fraud based on a configurable threshold.",
    lifespan=lifespan,
)


# --- Endpoints -----------------------------------------------------------

@app.get("/health")
def health():
    """
    Load balancer / Render health check.

    Returns 200 when the model is loaded and the service is ready.
    If this returns non-200, Render will restart the instance.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(transaction: Transaction):
    """
    Score a single transaction.

    The threshold comes from config/config.yaml — change it there to shift
    the precision/recall tradeoff without redeploying code.
    """
    try:
        result = predict(transaction.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
