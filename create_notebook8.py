"""
Build notebooks/08_monitoring_and_drift.ipynb programmatically.
Run from the repo root: python create_notebook8.py
"""
from pathlib import Path
import nbformat as nbf

OUTPUT = Path("notebooks/08_monitoring_and_drift.ipynb")

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11.0"},
}

cells = []
md   = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ── Cell 1: Title (exact as specified) ───────────────────────────────────────
cells.append(md(
    "# Notebook 8: Model Monitoring and Drift Detection\n"
    "## Goal\n"
    "In production a fraud model degrades over time as fraud patterns change. "
    "Fraudsters adapt. This notebook simulates what happens when the data "
    "distribution shifts and shows how to detect it before model performance drops.\n\n"
    "This directly answers the interview question: how would you monitor a fraud "
    "model in production?\n\n"
    "Three things to monitor:\n"
    "- Data drift: input feature distributions changing over time\n"
    "- Prediction drift: model output distribution changing\n"
    "- Performance drift: actual PR-AUC dropping on new labelled data"
))

# ── Cell 2: Setup markdown ────────────────────────────────────────────────────
cells.append(md("## Setup"))

# ── Cell 3: Imports ───────────────────────────────────────────────────────────
cells.append(code(
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import joblib\n"
    "from pathlib import Path\n"
    "from scipy import stats\n"
    "from sklearn.metrics import average_precision_score\n"
    "\n"
    "MODELS_PATH  = Path('../outputs/models')\n"
    "FIGURES_PATH = Path('../outputs/figures')\n"
    "FIGURES_PATH.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "model     = joblib.load(MODELS_PATH / 'best_xgb.pkl')\n"
    "THRESHOLD = 0.28\n"
    "FEATURES  = [f'V{i}' for i in range(1, 29)] + ['log_amount', 'hour_of_day']\n"
    "\n"
    "print('Model loaded :', type(model).__name__)\n"
    "print('Features     :', len(FEATURES))\n"
    "print('Threshold    :', THRESHOLD)"
))

# ── Cell 4: Load data markdown ────────────────────────────────────────────────
cells.append(md(
    "## 1. Load Data and Create Reference / Production Windows\n\n"
    "We treat the first 80 % of transactions (sorted by `Time`) as the **reference** "
    "window — the distribution the model was trained on. "
    "The remaining 20 % become the **production** window.\n\n"
    "In a real system these would be separate date-partitioned tables in your data warehouse."
))

# ── Cell 5: Load data code ────────────────────────────────────────────────────
cells.append(code(
    "df = pd.read_csv('../data/raw/creditcard.csv')\n"
    "\n"
    "df['log_amount']  = np.log1p(df['Amount'])\n"
    "df['hour_of_day'] = (df['Time'] % 86400) / 3600\n"
    "\n"
    "X = df[FEATURES]\n"
    "y = df['Class']\n"
    "\n"
    "split   = int(len(df) * 0.8)\n"
    "X_ref,  y_ref  = X.iloc[:split],  y.iloc[:split]\n"
    "X_prod, y_prod = X.iloc[split:],  y.iloc[split:]\n"
    "\n"
    "print(f'Reference  : {len(X_ref):>7,} rows | fraud rate: {y_ref.mean():.4%}')\n"
    "print(f'Production : {len(X_prod):>7,} rows | fraud rate: {y_prod.mean():.4%}')"
))

# ── Cell 6: Simulate drift markdown ──────────────────────────────────────────
cells.append(md(
    "## 2. Simulate Concept Drift in the Production Window\n\n"
    "We inject drift into the production window by shifting the three most important "
    "features (V14, V4, V12 — top SHAP features from notebook 05) and scaling up "
    "transaction amounts. This mimics fraudsters changing behaviour over time."
))

# ── Cell 7: Simulate drift code ───────────────────────────────────────────────
cells.append(code(
    "rng = np.random.default_rng(42)\n"
    "\n"
    "X_prod_drifted = X_prod.copy()\n"
    "\n"
    "X_prod_drifted['V14'] = X_prod_drifted['V14'] + rng.normal(0.8,  0.3, len(X_prod))\n"
    "X_prod_drifted['V4']  = X_prod_drifted['V4']  + rng.normal(0.5,  0.2, len(X_prod))\n"
    "X_prod_drifted['V12'] = X_prod_drifted['V12'] + rng.normal(-0.6, 0.3, len(X_prod))\n"
    "X_prod_drifted['log_amount'] = X_prod_drifted['log_amount'] * 1.2\n"
    "\n"
    "print('Drift injected into V14, V4, V12, log_amount')"
))

# ── Cell 8: KS test markdown ─────────────────────────────────────────────────
cells.append(md(
    "## 3. Data Drift — Kolmogorov-Smirnov Test\n\n"
    "The KS test compares two distributions without assuming normality. "
    "A low p-value (< 0.05) means the two samples are unlikely to come from the "
    "same distribution. We run it on every input feature and flag those that drift."
))

# ── Cell 9: KS test code ──────────────────────────────────────────────────────
cells.append(code(
    "ks_rows = []\n"
    "for col in FEATURES:\n"
    "    stat, p = stats.ks_2samp(X_ref[col], X_prod_drifted[col])\n"
    "    ks_rows.append({'feature': col, 'ks_stat': round(stat, 4), 'p_value': round(p, 6)})\n"
    "\n"
    "ks_df = pd.DataFrame(ks_rows).sort_values('ks_stat', ascending=False)\n"
    "ks_df['drifted'] = ks_df['p_value'] < 0.05\n"
    "\n"
    "print(f\"Features flagged as drifted: {ks_df['drifted'].sum()} / {len(FEATURES)}\")\n"
    "ks_df.head(10)"
))

# ── Cell 10: KS visualisation ────────────────────────────────────────────────
cells.append(code(
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
    "\n"
    "for ax, feat in zip(axes, ['V14', 'V4', 'V12']):\n"
    "    ax.hist(X_ref[feat],          bins=60, alpha=0.5, label='Reference',  density=True)\n"
    "    ax.hist(X_prod_drifted[feat], bins=60, alpha=0.5, label='Production', density=True)\n"
    "    row = ks_df[ks_df['feature'] == feat].iloc[0]\n"
    "    ax.set_title(f'{feat}  |  KS={row.ks_stat:.3f}  p={row.p_value:.4f}')\n"
    "    ax.legend(fontsize=8)\n"
    "    ax.set_xlabel('Value')\n"
    "    ax.set_ylabel('Density')\n"
    "\n"
    "plt.suptitle('Feature Distribution Shift — Reference vs Production', fontsize=12, y=1.02)\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGURES_PATH / '08_data_drift_ks.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))

# ── Cell 11: PSI markdown ─────────────────────────────────────────────────────
cells.append(md(
    "## 4. Data Drift — Population Stability Index (PSI)\n\n"
    "PSI is the standard drift metric used by fraud and risk teams in financial services. "
    "It measures how much a feature's distribution has shifted relative to a baseline.\n\n"
    "| PSI | Interpretation |\n"
    "|-----|----------------|\n"
    "| < 0.10 | No significant change |\n"
    "| 0.10 – 0.20 | Moderate change — investigate |\n"
    "| > 0.20 | Significant change — schedule retrain |"
))

# ── Cell 12: PSI calculation ─────────────────────────────────────────────────
cells.append(code(
    "def psi(reference: np.ndarray, production: np.ndarray, buckets: int = 10) -> float:\n"
    "    breakpoints  = np.unique(np.percentile(reference, np.linspace(0, 100, buckets + 1)))\n"
    "    ref_counts   = np.histogram(reference,  bins=breakpoints)[0] + 1e-6\n"
    "    prod_counts  = np.histogram(production, bins=breakpoints)[0] + 1e-6\n"
    "    ref_pct  = ref_counts  / ref_counts.sum()\n"
    "    prod_pct = prod_counts / prod_counts.sum()\n"
    "    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))\n"
    "\n"
    "\n"
    "psi_rows = []\n"
    "for col in FEATURES:\n"
    "    score = psi(X_ref[col].values, X_prod_drifted[col].values)\n"
    "    psi_rows.append({'feature': col, 'psi': round(score, 4)})\n"
    "\n"
    "psi_df = pd.DataFrame(psi_rows).sort_values('psi', ascending=False)\n"
    "psi_df['alert'] = pd.cut(\n"
    "    psi_df['psi'],\n"
    "    bins=[-np.inf, 0.10, 0.20, np.inf],\n"
    "    labels=['OK', 'Investigate', 'Retrain'],\n"
    ")\n"
    "print(psi_df.head(10).to_string(index=False))"
))

# ── Cell 13: PSI bar chart ────────────────────────────────────────────────────
cells.append(code(
    "colours = {'OK': '#4caf50', 'Investigate': '#ff9800', 'Retrain': '#f44336'}\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 5))\n"
    "ax.bar(psi_df['feature'], psi_df['psi'],\n"
    "       color=[colours[str(a)] for a in psi_df['alert']])\n"
    "ax.axhline(0.10, color='orange', linestyle='--', linewidth=1, label='Investigate (0.10)')\n"
    "ax.axhline(0.20, color='red',    linestyle='--', linewidth=1, label='Retrain (0.20)')\n"
    "ax.set_xlabel('Feature')\n"
    "ax.set_ylabel('PSI')\n"
    "ax.set_title('Population Stability Index per Feature')\n"
    "ax.tick_params(axis='x', rotation=45)\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGURES_PATH / '08_psi_bar.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))

# ── Cell 14: Prediction drift markdown ───────────────────────────────────────
cells.append(md(
    "## 5. Prediction Drift\n\n"
    "Even if individual features look stable, the model's *output distribution* can shift. "
    "Comparing score distributions is a label-free early-warning signal — "
    "it fires before you have enough labelled production data to measure PR-AUC."
))

# ── Cell 15: Prediction drift code ───────────────────────────────────────────
cells.append(code(
    "probs_ref  = model.predict_proba(X_ref)[:, 1]\n"
    "probs_prod = model.predict_proba(X_prod_drifted)[:, 1]\n"
    "\n"
    "ks_stat, ks_p = stats.ks_2samp(probs_ref, probs_prod)\n"
    "print(f'Prediction KS stat : {ks_stat:.4f}')\n"
    "print(f'Prediction KS p    : {ks_p:.2e}')\n"
    "print(f'Score distribution drifted: {ks_p < 0.05}')\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
    "\n"
    "axes[0].hist(probs_ref,  bins=80, alpha=0.55, label='Reference',  density=True, color='steelblue')\n"
    "axes[0].hist(probs_prod, bins=80, alpha=0.55, label='Production', density=True, color='tomato')\n"
    "axes[0].set_xlabel('Fraud Probability')\n"
    "axes[0].set_ylabel('Density')\n"
    "axes[0].set_title(f'Full Score Distribution  |  KS={ks_stat:.4f}')\n"
    "axes[0].legend()\n"
    "\n"
    "axes[1].hist(probs_ref[probs_ref   >= THRESHOLD], bins=50, alpha=0.55,\n"
    "             label='Reference',  density=True, color='steelblue')\n"
    "axes[1].hist(probs_prod[probs_prod >= THRESHOLD], bins=50, alpha=0.55,\n"
    "             label='Production', density=True, color='tomato')\n"
    "axes[1].set_xlabel('Fraud Probability')\n"
    "axes[1].set_title(f'High-Risk Scores Only (>= {THRESHOLD})')\n"
    "axes[1].legend()\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGURES_PATH / '08_prediction_drift.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))

# ── Cell 16: Performance drift markdown ──────────────────────────────────────
cells.append(md(
    "## 6. Performance Drift — PR-AUC Over Time\n\n"
    "The most reliable signal, but the slowest to arrive: you need ground-truth labels. "
    "We split the production window into 5 equal time buckets and compute PR-AUC for each. "
    "A declining trend means the model is actively degrading and retraining is overdue."
))

# ── Cell 17: Performance drift code ──────────────────────────────────────────
cells.append(code(
    "N_BUCKETS   = 5\n"
    "bucket_size = len(X_prod) // N_BUCKETS\n"
    "\n"
    "perf_rows = []\n"
    "for i in range(N_BUCKETS):\n"
    "    start, end = i * bucket_size, (i + 1) * bucket_size\n"
    "    X_b, y_b   = X_prod_drifted.iloc[start:end], y_prod.iloc[start:end]\n"
    "    if y_b.sum() == 0:\n"
    "        continue\n"
    "    pr_auc = average_precision_score(y_b, model.predict_proba(X_b)[:, 1])\n"
    "    perf_rows.append({'bucket': i + 1, 'pr_auc': round(pr_auc, 4), 'n_fraud': int(y_b.sum())})\n"
    "\n"
    "perf_df = pd.DataFrame(perf_rows)\n"
    "print(perf_df.to_string(index=False))\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "ax.plot(perf_df['bucket'], perf_df['pr_auc'], marker='o', linewidth=2, color='steelblue')\n"
    "ax.axhline(0.88, color='green', linestyle='--', linewidth=1, label='Baseline PR-AUC (0.88)')\n"
    "ax.fill_between(perf_df['bucket'], perf_df['pr_auc'], 0.88,\n"
    "                where=(perf_df['pr_auc'] < 0.88),\n"
    "                alpha=0.15, color='red', label='Degradation')\n"
    "ax.set_xlabel('Production Time Bucket')\n"
    "ax.set_ylabel('PR-AUC')\n"
    "ax.set_title('PR-AUC Degradation Across Production Time Windows')\n"
    "ax.set_ylim(0, 1)\n"
    "ax.legend()\n"
    "ax.grid(axis='y', alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIGURES_PATH / '08_performance_drift.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))

# ── Cell 18: Summary markdown ─────────────────────────────────────────────────
cells.append(md(
    "## 7. Summary — What to Monitor in Production\n\n"
    "| Signal | Method | Frequency | Action if triggered |\n"
    "|--------|--------|-----------|---------------------|\n"
    "| Data drift | KS test per feature | Daily | Investigate top features, alert ML team |\n"
    "| Data drift | PSI per feature | Weekly | PSI > 0.20 → schedule retrain |\n"
    "| Prediction drift | KS on score distribution | Daily | Alert if p < 0.05 |\n"
    "| Performance drift | PR-AUC per time window | Weekly (needs labels) | Drop > 5 % → retrain |\n\n"
    "### Retraining strategy\n\n"
    "1. **Trigger**: PSI > 0.20 on any top-5 SHAP feature, or PR-AUC drops > 5 % from baseline.\n"
    "2. **Data window**: retrain on a rolling 6-month window — old patterns are noise.\n"
    "3. **Validation**: new model must beat the current model on a held-out recent slice before promotion.\n"
    "4. **Threshold**: re-run notebook 06 on the new model — the optimal cut-off will shift.\n\n"
    "In a mature production system this pipeline runs as a scheduled job "
    "(Airflow DAG or GitHub Actions cron) and posts alerts to Slack or PagerDuty "
    "when any threshold is breached."
))

nb.cells = cells

OUTPUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Saved {OUTPUT}  ({len(cells)} cells)")
