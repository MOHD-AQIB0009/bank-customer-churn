# Predictive Modeling and Risk Scoring for Bank Customer Churn

Unified Mentor internship project (Machine Learning). Predicts which bank customers are likely
to leave, gives each customer a churn probability, and explains the drivers.

## Files
| File | Purpose |
|---|---|
| `churn_utils.py` | Shared code: data loading, feature engineering, preprocessing, helper functions |
| `train_models.py` | EDA, trains 5 models, evaluates them, creates explainability plots, saves the best model |
| `app.py` | Streamlit dashboard (risk calculator, what-if simulator, probability distribution, feature importance) |
| `requirements.txt` | Python libraries needed |

## How to run
1. Download the dataset with the **Access Dataset** button on the project page and save it in this
   folder as **`Churn_Modelling.csv`**. (If the file is missing, the code falls back to
   synthetic sample data so you can test, but results from it must NOT be reported.)
2. Install libraries:  `pip install -r requirements.txt`
3. Train and evaluate:  `python train_models.py`   (creates the `artifacts/` folder with charts, tables, model)
4. Launch the dashboard:  `streamlit run app.py`

## Expected dataset columns
`CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember,
EstimatedSalary, Exited` (target: 1 = churned, 0 = retained). `RowNumber`, `CustomerId` and `Surname`
are dropped automatically. If your dataset uses different names, edit `RAW_FEATURES`, `CATEGORICAL`
and `NUMERIC` at the top of `churn_utils.py`.

## What the project covers (matches the brief)
- **Models:** Logistic Regression (baseline), Decision Tree, Random Forest, Gradient Boosting, XGBoost (optional)
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC (`artifacts/model_comparison.csv`, ROC curves, confusion matrix)
- **Explainability:** permutation feature importance, SHAP summary (if `shap` installed), partial dependence plots
- **Prediction output:** churn probability (0-1) and a binary flag using an adjustable threshold
- **Engineered features:** BalanceSalaryRatio, TenureByAge, HasBalance, EngagementScore (engagement / product utilisation focus)
- **Dashboard modules:** churn risk calculator, probability distribution, feature importance, what-if simulator

## Remaining deliverables for submission
- Research paper (EDA, insights, recommendations) - use the charts in `artifacts/` and the real results
- Executive summary for stakeholders
