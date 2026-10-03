import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Exited"          # 1 = customer churned, 0 = customer retained
DATA_FILE = "Churn_Modelling.csv"
DROP_COLS = ["RowNumber", "CustomerId", "Surname"]   # identifiers, no predictive value

# Raw input columns the model expects (this is what the dashboard collects)
RAW_FEATURES = ["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

CATEGORICAL = ["Geography", "Gender"]
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
           "IsActiveMember", "EstimatedSalary",
           # engineered features
           "BalanceSalaryRatio", "TenureByAge", "HasBalance", "EngagementScore"]


def make_synthetic_data(n=10000, seed=42):
    """Fallback data with the SAME columns as the real Kaggle 'Churn_Modelling.csv'.
    Only used if the real file is not found, so the project always runs."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "RowNumber": np.arange(1, n + 1),
        "CustomerId": rng.integers(15_500_000, 15_900_000, n),
        "Surname": "Sample",
        "CreditScore": np.clip(rng.normal(650, 96, n), 350, 850).round().astype(int),
        "Geography": rng.choice(["France", "Germany", "Spain"], n, p=[0.5, 0.25, 0.25]),
        "Gender": rng.choice(["Male", "Female"], n, p=[0.55, 0.45]),
        "Age": np.clip(rng.normal(39, 10, n), 18, 92).round().astype(int),
        "Tenure": rng.integers(0, 11, n),
        "Balance": np.where(rng.random(n) < 0.36, 0, rng.normal(120000, 30000, n)).clip(0).round(2),
        "NumOfProducts": rng.choice([1, 2, 3, 4], n, p=[0.5, 0.46, 0.03, 0.01]),
        "HasCrCard": rng.choice([0, 1], n, p=[0.3, 0.7]),
        "IsActiveMember": rng.choice([0, 1], n, p=[0.48, 0.52]),
        "EstimatedSalary": rng.uniform(11, 200000, n).round(2),
    })
    z = (-3.2 + 0.07 * (df["Age"] - 38).clip(lower=0) - 1.1 * df["IsActiveMember"]
         + 0.75 * (df["Geography"] == "Germany") + 0.35 * (df["Gender"] == "Female")
         + 1.6 * (df["NumOfProducts"] >= 3) - 0.6 * (df["NumOfProducts"] == 2)
         + 0.000004 * df["Balance"] - 0.002 * (df["CreditScore"] - 650) / 10 + 0.9)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def load_data(path=DATA_FILE):
    """Load the real dataset if present, otherwise synthetic data."""
    if os.path.exists(path):
        print(f"[data] Loaded real dataset: {path}")
        df = pd.read_csv(path)
    else:
        print(f"[data] '{path}' not found -> using SYNTHETIC sample data (for testing only).")
        df = make_synthetic_data()
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def add_features(X):
    """Feature engineering based on engagement and product utilisation.
    Works on a DataFrame that has the RAW_FEATURES columns."""
    X = X.copy()
    X["BalanceSalaryRatio"] = X["Balance"] / (X["EstimatedSalary"] + 1)
    X["TenureByAge"] = X["Tenure"] / (X["Age"] + 1)
    X["HasBalance"] = (X["Balance"] > 0).astype(int)
    # simple engagement indicator: active member + holds product(s) + has card
    X["EngagementScore"] = (X["IsActiveMember"] + X["HasCrCard"]
                            + (X["NumOfProducts"] >= 2).astype(int))
    return X


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])


def risk_band(p):
    """Translate a churn probability into a business-friendly risk label."""
    if p < 0.25:
        return "Low"
    if p < 0.50:
        return "Medium"
    if p < 0.75:
        return "High"
    return "Very High"


def predict_one(model, customer: dict) -> float:
    """Churn probability (0-1) for a single customer given as a dict of RAW_FEATURES."""
    row = pd.DataFrame([customer])[RAW_FEATURES]
    return float(model.predict_proba(row)[0, 1])


def sweep(model, customer: dict, feature: str, values) -> pd.Series:
    """Churn probability as ONE feature changes and everything else stays the same."""
    rows = pd.DataFrame([{**customer, feature: v} for v in values])[RAW_FEATURES]
    return pd.Series(model.predict_proba(rows)[:, 1], index=list(values), name="churn_probability")
