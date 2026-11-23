"""Retrain the final churn model from the command line.

Usage:  python train.py

Rebuilds models/churn_model.pkl with the settings chosen in
notebooks/telco_churn_analysis.ipynb. The app also calls this automatically
if the saved model can't be loaded (for example, on a different scikit-learn version).
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 11 brand-new customers (tenure 0) have a blank TotalCharges: they haven't been billed yet
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return df


def build_model(categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])
    classifier = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_model(save: bool = True) -> Pipeline:
    df = load_data()
    X = df.drop(columns=["customerID", "Churn"])
    y = (df["Churn"] == "Yes").astype(int)
    categorical = [c for c in X.columns if c not in NUMERIC_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    model = build_model(categorical).fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Churn precision": precision_score(y_test, pred),
        "Churn recall": recall_score(y_test, pred),
        "Churn F1": f1_score(y_test, pred),
        "ROC AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
    }
    if save:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_DIR / "churn_model.pkl")
        (MODEL_DIR / "metrics.json").write_text(json.dumps({
            "model_name": "Random Forest",
            "test_metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": list(X.columns),
        }, indent=2))
    return model


if __name__ == "__main__":
    train_model()
    print(json.loads((MODEL_DIR / "metrics.json").read_text())["test_metrics"])
