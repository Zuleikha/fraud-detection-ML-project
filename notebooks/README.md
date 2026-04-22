# Notebooks

Run in order — each notebook builds on outputs saved by the previous one.

---

## Execution order

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_eda.ipynb` | Explores the raw dataset: class imbalance, amount distributions, fraud-by-hour patterns, and top PCA feature separability. |
| 02 | `02_baseline_model.ipynb` | Trains a default logistic regression on the raw data to establish a PR-AUC baseline of 0.73 to beat. |
| 03 | `03_imbalance_handling.ipynb` | Compares SMOTE, random undersampling, class weights, and SMOTE+Tomek; saves the best-scoring strategy as the new baseline. |
| 04 | `04_xgboost_tuning.ipynb` | Replaces logistic regression with XGBoost, tunes hyperparameters across 150 Optuna trials with every run logged to MLflow, reaching PR-AUC 0.88. |
| 05 | `05_shap_explainability.ipynb` | Loads the tuned XGBoost model and generates SHAP summary, beeswarm, waterfall, force, and dependence plots. |
| 06 | `06_threshold_tuning.ipynb` | Sweeps thresholds against a business cost matrix and finds 0.28 as the optimal cut-off to minimise total fraud cost. |
| 07 | `07_api_and_deployment.ipynb` | Smoke-tests the FastAPI `/predict` endpoint locally and walks through the Render deployment steps. |

---

## Outputs saved per notebook

### 01 — EDA
```
outputs/figures/01_class_balance.png
outputs/figures/01_amount_distribution.png
outputs/figures/01_fraud_by_hour.png
outputs/figures/01_feature_correlations.png
outputs/figures/01_top3_features.png
```

### 02 — Baseline model
```
outputs/figures/02_confusion_matrix.png
outputs/figures/02_pr_roc_curves.png
outputs/models/baseline_lr.joblib
outputs/models/baseline_scaler.joblib
```

### 03 — Imbalance handling
```
outputs/figures/03_strategy_comparison.png
outputs/figures/03_pr_curves.png
outputs/figures/03_confusion_matrix.png
outputs/models/best_imbalance_model.pkl
outputs/models/imbalance_results.pkl
```

### 04 — XGBoost tuning
```
outputs/figures/04_confusion_matrix.png
outputs/figures/04_shap_importance.png
outputs/models/best_xgb.pkl          ← used by notebooks 05, 06, and 07
```

### 05 — SHAP explainability
```
outputs/figures/05_shap_importance_bar.png
outputs/figures/05_shap_beeswarm.png
outputs/figures/05_shap_waterfall.png
outputs/figures/05_shap_force_plot.html
outputs/figures/05_shap_ranking.png
outputs/figures/05_shap_dependence_v14.png
```

### 06 — Threshold tuning
```
outputs/figures/06_cost_curve.png
outputs/figures/06_precision_recall_threshold.png
outputs/figures/06_confusion_matrix_comparison.png
```

### 07 — API and deployment
No new files saved. Loads `outputs/models/best_xgb.pkl` to run live prediction smoke-tests.
