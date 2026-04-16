# Fraud Detection ML Project

A production-ready fraud detection system built with scikit-learn and served via FastAPI.

## Project Structure

```
fraud-detection-ml/
├── data/
│   ├── raw/          # Original, immutable data
│   └── processed/    # Cleaned and feature-engineered data
├── notebooks/        # Exploratory analysis and experiments
├── src/              # Core ML logic (features, training, prediction)
├── app/              # FastAPI serving layer
├── models/           # Serialised model artefacts
├── docs/             # Learning notes — read in order (01 → 07)
├── tests/            # Unit tests
└── config/           # Threshold and feature config
```

## Quickstart

```bash
pip install -r requirements.txt

# Train
python src/train.py

# Serve
uvicorn app.main:app --reload

# Test
pytest tests/
```

## Docs

| File | Topic |
|------|-------|
| [01 Project Overview](docs/01_project_overview.md) | What we're building and why |
| [02 Data Understanding](docs/02_data_understanding.md) | EDA and class imbalance |
| [03 Feature Engineering](docs/03_feature_engineering.md) | Turning raw data into model inputs |
| [04 Model Training](docs/04_model_training.md) | Choosing and fitting the model |
| [05 Model Evaluation](docs/05_model_evaluation.md) | Metrics that matter for fraud |
| [06 Threshold Optimisation](docs/06_threshold_optimisation.md) | Why 0.5 is almost always wrong |
| [07 API and Deployment](docs/07_api_and_deployment.md) | Serving the model in production |
