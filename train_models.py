import json
import os
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier

from churn_utils import (RAW_FEATURES, TARGET, add_features, build_preprocessor,
                         load_data)

warnings.filterwarnings("ignore")
SHOW = False        # True = also pop up every chart in a window
ART = "artifacts"
os.makedirs(ART, exist_ok=True)
if not SHOW:
    matplotlib.use("Agg")
RANDOM_STATE = 42


def finish(name):
    plt.tight_layout()
    plt.savefig(f"{ART}/{name}", dpi=130)
    if SHOW:
        plt.show()
    plt.close()


# ------------------------------------------------------------------ 1. EDA
def eda(df):
    print("\n=== EDA ===")
    print("Shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print(f"Churn rate: {df[TARGET].mean() * 100:.1f}%")

    # churn distribution
    df[TARGET].value_counts().sort_index().plot.bar(color=["#4C72B0", "#C44E52"])
    plt.xticks([0, 1], ["Retained", "Churned"], rotation=0)
    plt.title("Class distribution"); plt.ylabel("Customers")
    finish("eda_class_balance.png")

    # churn rate by category / key drivers
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.ravel(), ["Geography", "Gender", "NumOfProducts",
                                      "IsActiveMember", "HasCrCard", "Tenure"]):
        (df.groupby(col)[TARGET].mean() * 100).plot.bar(ax=ax, color="#4C72B0")
        ax.set_title(f"Churn rate (%) by {col}"); ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=0)
    finish("eda_churn_by_category.png")

    # numeric distributions split by churn
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["Age", "CreditScore", "Balance"]):
        for val, lab, c in [(0, "Retained", "#4C72B0"), (1, "Churned", "#C44E52")]:
            ax.hist(df.loc[df[TARGET] == val, col], bins=30, alpha=0.6, label=lab, color=c)
        ax.set_title(col); ax.legend()
    finish("eda_numeric_distributions.png")

    # correlation heatmap
    corr = df.select_dtypes("number").corr()
    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    plt.yticks(range(len(corr)), corr.columns)
    plt.colorbar(); plt.title("Correlation matrix")
    finish("eda_correlation.png")


# ------------------------------------------------------------------ 2. Models
def make_pipeline(clf):
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor()),
        ("model", clf),
    ])


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10,
                                                min_samples_leaf=10, class_weight="balanced",
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                        max_depth=3, random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                          subsample=0.9, colsample_bytree=0.9,
                                          eval_metric="logloss", random_state=RANDOM_STATE)
    except ImportError:
        print("[info] xgboost not installed -> skipping the optional XGBoost model "
              "(pip install xgboost to enable it)")
    return models


# ------------------------------------------------------------------ main
def main():
    df = load_data()
    eda(df)

    X, y = df[RAW_FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # ---- train + evaluate every model
    print("\n=== Model comparison (20% hold-out test set) ===")
    results, fitted, probas = [], {}, {}
    for name, clf in get_models().items():
        pipe = make_pipeline(clf).fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred, zero_division=0),
            "Recall": recall_score(y_te, pred),
            "F1": f1_score(y_te, pred),
            "ROC_AUC": roc_auc_score(y_te, proba),
        })
        fitted[name], probas[name] = pipe, proba
    res = pd.DataFrame(results).set_index("Model").round(4).sort_values("ROC_AUC", ascending=False)
    print(res)
    res.to_csv(f"{ART}/model_comparison.csv")

    best_name = res.index[0]
    best = fitted[best_name]
    best_proba = probas[best_name]
    print(f"\nBest model (by ROC-AUC): {best_name}")

    # ---- ROC curves
    plt.figure(figsize=(7, 6))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curves"); plt.legend()
    finish("roc_curves.png")

    # ---- best threshold (max F1) and confusion matrix
    prec, rec, thr = precision_recall_curve(y_te, best_proba)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    best_thr = float(thr[np.argmax(f1s)])
    print(f"Threshold that maximises F1: {best_thr:.2f}")
    ConfusionMatrixDisplay.from_predictions(
        y_te, (best_proba >= best_thr).astype(int), display_labels=["Retained", "Churned"],
        cmap="Blues")
    plt.title(f"{best_name} - confusion matrix (threshold={best_thr:.2f})")
    finish("confusion_matrix.png")

    # ---- explainability 1: permutation feature importance (model-agnostic)
    imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=5,
                                 random_state=RANDOM_STATE, n_jobs=-1)
    imp_df = (pd.DataFrame({"feature": RAW_FEATURES, "importance": imp.importances_mean,
                            "std": imp.importances_std})
              .sort_values("importance", ascending=True))
    imp_df.to_csv(f"{ART}/feature_importance.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.barh(imp_df["feature"], imp_df["importance"], xerr=imp_df["std"], color="#4C72B0")
    plt.xlabel("Drop in ROC-AUC when feature is shuffled")
    plt.title(f"Feature importance - {best_name}")
    finish("feature_importance.png")
    print("\nTop drivers of churn:\n", imp_df.sort_values("importance", ascending=False).head(5))

    # ---- explainability 2: partial dependence plots
    sample = X_te.sample(min(1500, len(X_te)), random_state=RANDOM_STATE)
    # PDP uses fractional grid values, so integer columns must be floats
    sample_f = sample.astype({c: float for c in sample.select_dtypes("number").columns})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    PartialDependenceDisplay.from_estimator(
        best, sample_f, ["Age", "Balance", "NumOfProducts", "CreditScore"],
        grid_resolution=20, ax=ax.ravel())
    fig.suptitle("Partial dependence (average effect on churn probability)")
    finish("partial_dependence.png")

    # ---- explainability 3: SHAP (optional)
    try:
        import shap
        if best_name in ("Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"):
            Xt = best.named_steps["features"].transform(sample)
            Xt = best.named_steps["prep"].transform(Xt)
            names = best.named_steps["prep"].get_feature_names_out()
            sv = shap.TreeExplainer(best.named_steps["model"]).shap_values(Xt)
            if isinstance(sv, list):
                sv = sv[1]
            elif getattr(sv, "ndim", 2) == 3:
                sv = sv[:, :, 1]
            shap.summary_plot(sv, Xt, feature_names=names, show=False)
            finish("shap_summary.png")
            print("[shap] summary plot saved")
    except ImportError:
        print("[info] shap not installed -> skipping SHAP plot (pip install shap to enable it)")
    except Exception as e:  # SHAP version quirks should never break the project
        print("[info] SHAP plot skipped:", e)

    # ---- save everything the dashboard needs
    joblib.dump(best, f"{ART}/best_model.joblib")
    pd.DataFrame({"churn_probability": best_proba, "actual": y_te.values}).to_csv(
        f"{ART}/test_predictions.csv", index=False)
    X_te.assign(**{TARGET: y_te.values}).head(500).to_csv(f"{ART}/sample_customers.csv", index=False)
    with open(f"{ART}/metadata.json", "w") as f:
        json.dump({"best_model": best_name, "threshold": best_thr,
                   "data_rows": int(len(df)), "churn_rate": float(y.mean()),
                   "metrics": res.loc[best_name].to_dict()}, f, indent=2)
    print(f"\nDone. Everything saved in ./{ART}/  ->  now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
