# 00 — Roadmap: 5-Day Fraud Detection Interview Prep Plan

Build a production-grade fraud detection system from raw data to deployed API in 5 days.
Each day maps to one or two notebooks and leaves you with concrete artefacts and talking points for interviews.

---

## Ground rules (apply across all notebooks for fair comparison)

- **Same split every time:** `train_test_split(stratify=y, test_size=0.2, random_state=42)`
  Lock this once in Day 1 and never change it — otherwise PR-AUC improvements may just be lucky splits.
- **Primary metric:** PR-AUC (Average Precision). Accuracy is meaningless at 0.17% fraud rate.
- **Secondary metrics:** ROC-AUC, Precision, Recall, F1 at chosen threshold.
- **Features engineered from EDA:** `Amount_log` (log1p of Amount), `Hour` (Time // 3600 % 24).
  Both are derived once and reused identically in every notebook.
- **No data leakage:** scalers and resamplers are always fit on training folds only,
  never on the full training set before splitting.

---

## Day 1 — EDA (`01_eda.ipynb`)

**Goal:** understand the data before touching a model.

| Task | Outcome |
|------|---------|
| Load `creditcard.csv`, check shape and dtypes | Confirm 284,807 rows, 31 columns, zero nulls |
| Plot class distribution | Fraud = 0.172% → accuracy is useless |
| Examine Amount distribution | Right-skewed → use `log1p` scaling |
| Derive `hour_of_day` from `Time` | Fraud rate varies by hour → useful feature |
| Rank PCA features by correlation with `Class` | V17, V14, V12 are strongest signals |
| Plot top-3 feature distributions by class | Clear separation → model has real signal to learn |

**Interview talking point:** "The first thing I did was check class balance. At 0.17% fraud rate,
a model that predicts all-legit gets 99.83% accuracy but catches zero fraud. So I switched to PR-AUC immediately."

---

## Day 2 — Baseline Model (`02_baseline_model.ipynb`)

**Goal:** establish a floor to beat. Logistic Regression only.

| Task | Outcome |
|------|---------|
| `StandardScaler` → `LogisticRegression(class_weight='balanced')` | Simple, fast, interpretable |
| Evaluate: PR-AUC, ROC-AUC, confusion matrix, PR curve | Baseline score to beat in Day 3 |
| Save `models/baseline_lr.joblib` | Reusable artefact |

**Interview talking point:** "I always start with the simplest model that addresses the known problem.
`class_weight='balanced'` is one line and immediately beats the naive classifier on PR-AUC."

---

## Day 3 — Imbalance Handling (`03_imbalance_handling.ipynb`)

**Goal:** find the best resampling strategy. All strategies use the same LR classifier so the
comparison isolates the effect of resampling alone.

| Strategy | What it does |
|----------|-------------|
| `class_weight='balanced'` | Up-weights minority in the loss — no data added |
| `RandomUnderSampler` | Randomly drops majority rows until 1:1 ratio |
| `SMOTE` | Synthesises new minority rows by interpolating nearest neighbours |
| `SMOTETomek` | SMOTE + removes borderline Tomek pairs from both classes |

All four are wrapped in `imblearn.Pipeline` so resampling never touches test data.
Winner (highest PR-AUC on held-out test) is saved to `models/best_imbalance_model.pkl`.

**Interview talking point:** "SMOTE looks attractive but it can create unrealistic synthetic samples.
I compare it against the simpler alternatives on PR-AUC and let the data decide."

---

## Day 4 — XGBoost + MLflow + SHAP (`04_xgboost_tuning.ipynb`)

**Goal:** swap LR for XGBoost, tune with Optuna, explain predictions with SHAP.

| Task | Outcome |
|------|---------|
| Out-of-the-box XGBoost with `scale_pos_weight` | Immediate PR-AUC jump over LR |
| Optuna: 30 trials, 3-fold stratified CV | Each trial logged as nested MLflow run |
| Retrain best params on full train set | Final tuned model |
| SHAP `TreeExplainer`: bar, beeswarm, waterfall | Explains WHY specific transactions are flagged |
| Save `models/xgboost_tuned.pkl` + MLflow run | Reproducible experiment record |

**Hyperparams tuned:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`,
`colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`.

**Interview talking point:** "SHAP lets me tell a stakeholder exactly which features pushed a
specific transaction over the fraud threshold — that's essential for regulatory explainability."

---

## Day 5 — Evaluation, Threshold Tuning & Deployment (`05` → `06` → `07`)

**Goal:** pick the right threshold using business costs, then serve the model.

### `05_model_evaluation.ipynb`
Full evaluation suite on the tuned XGBoost: confusion matrix, ROC, PR curve, calibration plot.
Compare all models built across Days 2–4 on the same held-out test set.

### `06_threshold_tuning.ipynb`
Default threshold of 0.5 is almost never optimal for imbalanced problems.

```
Cost matrix:
  False Negative (missed fraud) = €200 average loss
  False Positive (blocked legit) = €5 customer friction cost
  → optimal threshold minimises: FN×200 + FP×5
```

Sweep thresholds 0.01 → 0.99, compute total cost at each point, pick the minimum.
Write the winning threshold back to `config.yaml`.

### `07_api_and_deployment.ipynb`
Smoke-test the FastAPI `/predict` endpoint locally, walk through Render deployment,
verify the `/health` check.

**Interview talking point:** "Threshold tuning is where ML meets business. I build a cost matrix
with the fraud team, sweep the threshold, and show them the precision-recall tradeoff in euros —
not in abstract metric numbers."

---

## Artefacts produced by the end of Day 5

| File | Created in |
|------|-----------|
| `models/baseline_lr.joblib` | Day 2 |
| `models/baseline_scaler.joblib` | Day 2 |
| `models/best_imbalance_model.pkl` | Day 3 |
| `models/imbalance_results.pkl` | Day 3 |
| `models/xgboost_tuned.pkl` | Day 4 |
| `reports/figures/01_*.png` → `07_*.png` | Days 1–5 |
| MLflow experiment `fraud-xgboost-tuning` | Day 4 |

---

## Key interview questions this project answers

| Question | Where the answer lives |
|----------|----------------------|
| How do you handle class imbalance? | Day 3 — four strategies compared empirically |
| Why not use accuracy? | Day 1 EDA + Day 2 baseline |
| How do you pick a threshold? | Day 5 — cost matrix sweep |
| How do you explain a black-box model? | Day 4 — SHAP waterfall per transaction |
| How do you track experiments? | Day 4 — MLflow nested runs |
| How do you deploy an ML model? | Day 5 — FastAPI + Render |
