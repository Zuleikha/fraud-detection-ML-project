# 07 — API and Deployment

## What a production ML API looks like vs a notebook

A Jupyter notebook is a linear script: load data, train, evaluate, done.
A production API is a long-running process that waits for requests and scores them one at a time.

```
Notebook                          Production API
─────────────────────────────────────────────────────
Runs top to bottom, once          Runs indefinitely, handles many requests
Loads model each cell run         Loads model once at startup (saves 200ms/request)
You call it                       Load balancer calls it
No validation                     Pydantic validates every request shape
No versioning                     /health tells the balancer "I'm ready"
```

## How FastAPI serves the model

Request lifecycle:

```
Client (another service, browser, curl)
  │
  ▼
POST /predict  { "amount": 249.99, "merchant_category": "electronics", ... }
  │
  ▼
Pydantic validates request shape — rejects malformed input with 422 before
the model ever sees it
  │
  ▼
transaction.model_dump() → dict
  │
  ▼
predict(transaction_dict)
  ├── wraps in DataFrame
  ├── pipeline.predict_proba() → [0.65, 0.35]   # [P(legit), P(fraud)]
  ├── prob = 0.35
  ├── threshold = config["prediction"]["threshold"]  # 0.30
  └── is_fraud = prob >= threshold  # True
  │
  ▼
Response: { "fraud_probability": 0.35, "is_fraud": true, "threshold_used": 0.3 }
```

The model is loaded **once** in the `lifespan` context manager when the app starts.
Every request reuses the same in-memory object. If you loaded the model inside the
endpoint function, you'd pay a ~200ms disk read on every single call.

## What the /health endpoint is for

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

This exists for the **load balancer**, not for users.

When Render (or AWS ALB, or Kubernetes) deploys your service it continuously polls `/health`.
If it returns a non-200 response, the balancer stops routing traffic to that instance and
may restart it. This means:

- New deployments only receive traffic after they pass the health check
- Crashed instances are replaced automatically
- Rolling deploys work: old version stays up until new version is healthy

In a more complete implementation, the health check would also verify the model is loaded:

```python
@app.get("/health")
def health():
    from predict import _model
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}
```

## Deploying to Render

You've already deployed `rentpulse-automation-system` to Render. The differences for an ML API:

| Aspect | rentpulse | fraud-detection-ml |
|--------|-----------|-------------------|
| Start command | `python app.py` | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Build command | `pip install -r requirements.txt` | same |
| Model artefact | N/A | `models/fraud_model.joblib` must exist at startup |
| Memory | Low | Medium — scikit-learn models load into RAM |
| Health check path | none configured | `/health` |

**The model artefact problem:** Render rebuilds from git on every deploy. Your `.gitignore`
excludes `models/*.joblib`. Three solutions:
1. Train the model as part of the build command (`python src/train.py`)
2. Store the model in Render's persistent disk and mount it
3. Store the model in cloud storage (S3/GCS) and download on startup

Option 1 is easiest to start with. Add to your Render build command:
```
pip install -r requirements.txt && python src/train.py
```

## What monitoring you would add next in a real job

Once the service is running, these are the first three things a senior ML engineer adds:

### 1. Prediction distribution monitoring

Log every `fraud_probability` value to a database or monitoring tool (Datadog, Grafana).
Watch the daily mean. If it shifts — say, from 0.12 to 0.05 — something changed.
Either fraud patterns changed, or your input features changed, or there's a bug.

```python
# In predict()
logger.info("prediction", extra={
    "fraud_probability": prob,
    "is_fraud": is_fraud,
    "threshold": threshold,
    "merchant_category": transaction["merchant_category"],
})
```

### 2. Data drift detection

Compare the distribution of incoming features to the training distribution.
If `amount_mean_last_24h` starts looking different from training, the model is
predicting on data it wasn't built for.

Tools: Evidently AI, WhyLabs, or a simple hourly Kolmogorov-Smirnov test.

### 3. Label feedback loop

In fraud detection, labels arrive late — a transaction confirmed fraudulent
today was made three weeks ago. Build a pipeline to:
1. Collect confirmed fraud labels from the chargebacks/investigations team
2. Join them back to your logged predictions
3. Compute actual precision and recall over the last 30 days
4. Alert if recall drops below a threshold (e.g. < 85%)

Without this you're flying blind. The model might be degrading for weeks before anyone notices.
