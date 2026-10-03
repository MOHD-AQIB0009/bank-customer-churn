import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from churn_utils import predict_one, risk_band, sweep

ART = "artifacts"
st.set_page_config(page_title="Bank Churn Risk Scoring", page_icon="🏦", layout="wide")

# ----------------------------------------------------------------- load artifacts
if not os.path.exists(f"{ART}/best_model.joblib"):
    st.error("Model not found. Please run `python train_models.py` first, then restart the app.")
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(f"{ART}/best_model.joblib")


@st.cache_data
def load_tables():
    with open(f"{ART}/metadata.json") as f:
        meta = json.load(f)
    return (meta,
            pd.read_csv(f"{ART}/test_predictions.csv"),
            pd.read_csv(f"{ART}/feature_importance.csv"),
            pd.read_csv(f"{ART}/model_comparison.csv"))


model = load_model()
meta, test_pred, imp, comparison = load_tables()

# ----------------------------------------------------------------- sidebar
st.sidebar.title("🏦 Churn Risk Scoring")
st.sidebar.write(f"**Model in use:** {meta['best_model']}")
st.sidebar.write(f"**ROC-AUC:** {meta['metrics']['ROC_AUC']:.3f}")
threshold = st.sidebar.slider(
    "Decision threshold", 0.05, 0.95, round(float(meta["threshold"]), 2), 0.01,
    help="A customer is flagged as 'will churn' when probability >= threshold. "
         "Lower threshold = catch more churners but more false alarms.")
st.sidebar.caption("Default threshold = the value that maximises F1 on the test set.")

st.title("Predictive Modeling and Risk Scoring for Bank Customer Churn")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Risk Calculator", "🔀 What-if Simulator", "📊 Probability Distribution",
    "🧠 Feature Importance", "📈 Model Performance"])

# ================================================================= 1. calculator
with tab1:
    st.subheader("Customer churn risk calculator")
    st.write("Enter the customer's details to get a churn probability.")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit = st.slider("Credit score", 350, 850, 650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 92, 40)
    with c2:
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5)
        balance = st.number_input("Account balance", 0.0, 300000.0, 75000.0, 1000.0)
        salary = st.number_input("Estimated salary", 0.0, 250000.0, 100000.0, 1000.0)
    with c3:
        products = st.slider("Number of products", 1, 4, 2)
        has_card = st.selectbox("Has credit card?", [1, 0], format_func=lambda v: "Yes" if v else "No")
        active = st.selectbox("Active member?", [1, 0], format_func=lambda v: "Yes" if v else "No")

    customer = dict(CreditScore=credit, Geography=geography, Gender=gender, Age=age,
                    Tenure=tenure, Balance=float(balance), NumOfProducts=products,
                    HasCrCard=has_card, IsActiveMember=active, EstimatedSalary=float(salary))
    prob = predict_one(model, customer)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn probability", f"{prob * 100:.1f}%")
    m2.metric("Risk band", risk_band(prob))
    m3.metric("Prediction at threshold", "WILL CHURN" if prob >= threshold else "Will stay")
    st.progress(min(max(prob, 0.0), 1.0))
    if prob >= threshold:
        st.warning("High churn risk - consider a proactive retention action "
                   "(personalised offer, relationship-manager call, product bundle).")
    else:
        st.success("Low churn risk - standard engagement is sufficient.")

# ================================================================= 2. what-if
with tab2:
    st.subheader("What-if scenario simulator")
    st.write("Start from the customer entered in the calculator, change engagement / product "
             "values, and see how the churn probability moves.")
    s1, s2, s3 = st.columns(3)
    with s1:
        n_products = st.slider("Scenario: number of products", 1, 4, products)
        n_active = st.selectbox("Scenario: active member?", [1, 0], index=[1, 0].index(active),
                                format_func=lambda v: "Yes" if v else "No")
    with s2:
        n_balance = st.number_input("Scenario: balance", 0.0, 300000.0, float(balance), 1000.0)
        n_tenure = st.slider("Scenario: tenure (years)", 0, 10, tenure)
    with s3:
        n_card = st.selectbox("Scenario: has credit card?", [1, 0], index=[1, 0].index(has_card),
                              format_func=lambda v: "Yes" if v else "No")
        n_credit = st.slider("Scenario: credit score", 350, 850, credit)

    scenario = {**customer, "NumOfProducts": n_products, "IsActiveMember": n_active,
                "Balance": float(n_balance), "Tenure": n_tenure, "HasCrCard": n_card,
                "CreditScore": n_credit}
    new_prob = predict_one(model, scenario)

    a, b, c = st.columns(3)
    a.metric("Baseline probability", f"{prob * 100:.1f}%")
    b.metric("Scenario probability", f"{new_prob * 100:.1f}%",
             delta=f"{(new_prob - prob) * 100:+.1f} pts", delta_color="inverse")
    c.metric("Scenario risk band", risk_band(new_prob))

    st.markdown("**Sensitivity curve** - how churn probability changes as one feature varies "
                "(everything else stays as in the scenario).")
    options = {
        "NumOfProducts": [1, 2, 3, 4],
        "IsActiveMember": [0, 1],
        "Age": list(range(18, 81, 2)),
        "Balance": list(range(0, 250001, 10000)),
        "CreditScore": list(range(350, 851, 25)),
        "Tenure": list(range(0, 11)),
    }
    feat = st.selectbox("Feature to vary", list(options))
    curve = sweep(model, scenario, feat, options[feat])
    st.line_chart(curve.rename("Churn probability"))

# ================================================================= 3. distribution
with tab3:
    st.subheader("Probability distribution")
    st.write("Predicted churn probabilities for the hold-out test customers.")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(test_pred.loc[test_pred.actual == 0, "churn_probability"], bins=40, alpha=0.65,
            label="Actually retained", color="#4C72B0")
    ax.hist(test_pred.loc[test_pred.actual == 1, "churn_probability"], bins=40, alpha=0.65,
            label="Actually churned", color="#C44E52")
    ax.axvline(threshold, color="k", linestyle="--", label=f"Threshold {threshold:.2f}")
    ax.axvline(prob, color="green", linestyle="-", linewidth=2, label="Current customer")
    ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Customers"); ax.legend()
    st.pyplot(fig)

    bands = test_pred["churn_probability"].apply(risk_band)
    order = ["Low", "Medium", "High", "Very High"]
    band_df = bands.value_counts().reindex(order).fillna(0).astype(int)
    churn_by_band = (test_pred.assign(band=bands).groupby("band")["actual"].mean()
                     .reindex(order).fillna(0) * 100).round(1)
    col_a, col_b = st.columns(2)
    col_a.markdown("**Customers per risk band**")
    col_a.bar_chart(band_df)
    col_b.markdown("**Actual churn rate (%) per risk band**")
    col_b.bar_chart(churn_by_band)
    flagged = (test_pred["churn_probability"] >= threshold).mean() * 100
    st.info(f"At threshold {threshold:.2f}, **{flagged:.1f}%** of customers would be flagged "
            "for a retention action.")

# ================================================================= 4. importance
with tab4:
    st.subheader("Feature importance dashboard")
    st.write("Which customer attributes drive churn predictions "
             "(permutation importance: drop in ROC-AUC when a feature is shuffled).")
    imp_sorted = imp.sort_values("importance", ascending=False)
    st.bar_chart(imp_sorted.set_index("feature")["importance"])
    st.dataframe(imp_sorted.round(4), hide_index=True)

    st.markdown("**Partial dependence** - average effect of a feature on churn probability")
    if os.path.exists(f"{ART}/partial_dependence.png"):
        st.image(f"{ART}/partial_dependence.png")
    if os.path.exists(f"{ART}/shap_summary.png"):
        st.markdown("**SHAP summary**")
        st.image(f"{ART}/shap_summary.png")

# ================================================================= 5. performance
with tab5:
    st.subheader("Model comparison")
    st.dataframe(comparison.round(4), hide_index=True)
    p1, p2 = st.columns(2)
    if os.path.exists(f"{ART}/roc_curves.png"):
        p1.image(f"{ART}/roc_curves.png", caption="ROC curves")
    if os.path.exists(f"{ART}/confusion_matrix.png"):
        p2.image(f"{ART}/confusion_matrix.png", caption="Confusion matrix (best model)")
    st.caption("Metrics in the table use a 0.5 threshold; the dashboard threshold can be changed "
               "in the sidebar.")
