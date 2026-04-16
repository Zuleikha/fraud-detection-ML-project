"""
Train the fraud detection model and serialise it to models/.

Run: python src/train.py
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from features import load_config, build_time_features, build_velocity_features, build_preprocessor


def train(config_path: str = "config/config.yaml"):
    cfg = load_config(config_path)

    print("Loading data...")
    df = pd.read_csv(cfg["data"]["raw_path"])
    df = build_time_features(df)
    df = build_velocity_features(df)

    feature_cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    X = df[feature_cols]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )

    preprocessor = build_preprocessor(cfg)

    # SMOTE addresses class imbalance — fraud is typically <1% of transactions.
    # Applied inside the pipeline so it only runs on training folds,
    # never leaking into validation/test.
    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=cfg["data"]["random_state"])),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=cfg["data"]["random_state"],
            n_jobs=-1,
        )),
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC: {roc:.4f}")

    # Evaluate at config threshold, not 0.5
    threshold = cfg["prediction"]["threshold"]
    y_pred = (y_prob >= threshold).astype(int)
    print(f"\nClassification report at threshold={threshold}:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    model_path = cfg["model"]["path"]
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    train()
