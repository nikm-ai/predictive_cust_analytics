import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_curve, auc, confusion_matrix, mean_squared_error, r2_score)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="SaaS Customer Churn & Revenue Prediction", layout="wide")
st.title("SaaS Customer Churn & Revenue Prediction Dashboard")
st.markdown("""
### Business Context  
This dashboard simulates how a payment platform might analyze merchant customers to:  
- Predict **churn risk** for proactive retention  
- Predict **monthly revenue** to guide financial planning
""")

# --- SYNTHETIC DATA ---
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 2000
    data = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "subscription_age_days": np.random.randint(30, 900, n),
        "monthly_txn_volume": np.random.poisson(50, n),
        "avg_txn_value": np.round(np.random.normal(75, 25, n), 2),
        "payment_method": np.random.choice(
            ["Credit Card", "ACH", "Wire Transfer", "Paypal"], n, p=[0.6, 0.25, 0.1, 0.05]),
        "support_tickets": np.random.poisson(1.5, n),
        "is_active": np.random.choice([0, 1], n, p=[0.4, 0.6])
    })
    data["churn"] = np.where(
        (data["is_active"] == 0) | ((data["support_tickets"] > 1) & (data["monthly_txn_volume"] < 35)),
        1, 0)
    data["monthly_revenue"] = data["monthly_txn_volume"] * data["avg_txn_value"] * np.random.uniform(0.85, 1.15, n)
    data["monthly_revenue"] = data["monthly_revenue"].round(2)
    return data

df = load_sample_data()

uploaded_file = st.sidebar.file_uploader("Upload your customer dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded")
else:
    st.sidebar.info("Using synthetic sample dataset")

df = df.dropna()

# --- FEATURE ENGINEERING ---
@st.cache_data
def feature_engineering(df):
    df = df.copy()

    # Original feature
    df["engagement_score"] = df["monthly_txn_volume"] * (df["subscription_age_days"] / 365)

    # New churn features:
    avg_txn = df["monthly_txn_volume"].mean()
    df["txn_volume_change"] = (df["monthly_txn_volume"] - avg_txn) / avg_txn
    df["days_since_last_txn"] = np.random.randint(1, 60, len(df))

    # 5 additional industry-standard churn features:
    # 1. avg_txn_value_change: simulate change in avg txn value
    df["avg_txn_value_change"] = (df["avg_txn_value"] - df["avg_txn_value"].mean()) / df["avg_txn_value"].mean()

    # 2. high_support_flag: binary flag for many support tickets (e.g. >3)
    df["high_support_flag"] = (df["support_tickets"] > 3).astype(int)

    # 3. low_volume_flag: binary flag for low monthly transaction volume (< 30)
    df["low_volume_flag"] = (df["monthly_txn_volume"] < 30).astype(int)

    # 4. recent_subscription_flag: new customers < 90 days
    df["recent_subscription_flag"] = (df["subscription_age_days"] < 90).astype(int)

    # 5. payment_method_risk: assign risk score to payment method (example)
    risk_map = {"Credit Card": 1, "ACH": 2, "Wire Transfer": 3, "Paypal": 4}
    df["payment_method_risk"] = df["payment_method"].map(risk_map)

    # One-hot encode payment_method (still useful)
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    payment_ohe = encoder.fit_transform(df[["payment_method"]])
    payment_df = pd.DataFrame(payment_ohe, columns=[f"pay_{cat}" for cat in encoder.categories_[0][1:]])
    df = pd.concat([df.reset_index(drop=True), payment_df], axis=1)

    return df, list(payment_df.columns)

df, payment_ohe_cols = feature_engineering(df)

# --- SIDEBAR ---
st.sidebar.header("Select Model & Features")
task = st.sidebar.selectbox("Prediction Task", ["Churn Classification", "Revenue Regression"])

base_features = [
    "subscription_age_days",
    "monthly_txn_volume",
    "avg_txn_value",
    "support_tickets",
    "engagement_score",
    "txn_volume_change",
    "days_since_last_txn",
    "avg_txn_value_change",
    "high_support_flag",
    "low_volume_flag",
    "recent_subscription_flag",
    "payment_method_risk"
] + payment_ohe_cols

selected_features = st.sidebar.multiselect("Features", base_features, default=base_features)

if task == "Churn Classification":
    threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

target = "churn" if task == "Churn Classification" else "monthly_revenue"

# --- DATA PREVIEW ---
st.subheader("1️⃣ Dataset Overview")
st.dataframe(df.head())
st.markdown(f"**Total rows:** `{len(df)}` &nbsp;|&nbsp; **Churn rate:** `{df['churn'].mean():.2%}`")

# --- DISTRIBUTIONS ---
st.subheader("2️⃣ Feature Distributions")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(df, x="subscription_age_days", nbins=40)
    st.plotly_chart(fig1)
    st.markdown("- **Most merchants are within 1–2 years of tenure**, showing a typical SaaS customer lifecycle.")

with col2:
    fig2 = px.histogram(df, x="monthly_txn_volume", nbins=40)
    st.plotly_chart(fig2)
    st.markdown("- **Transaction volume distribution** shows usage spread — heavy users are stable contributors.")

# --- MODELING ---
st.subheader("3️⃣ Model Training & Evaluation")

if len(selected_features) == 0:
    st.warning("Please select at least one feature.")
else:
    X = df[selected_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task == "Churn Classification":
        # Calculate scale_pos_weight for imbalance
        neg, pos = np.bincount(y_train)
        spw = neg / pos

        # Hyperparameter tuning with RandomizedSearchCV
        param_dist = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'scale_pos_weight': [spw]  # keep imbalance tuned
        }
        base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        search = RandomizedSearchCV(base_model, param_distributions=param_dist,
                                    n_iter=20, scoring='f1', cv=3, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred_adjusted = (y_prob >= threshold).astype(int)

        acc = (y_pred_adjusted == y_test).mean()
        prec = precision_score(y_test, y_pred_adjusted, zero_division=0)
        rec = recall_score(y_test, y_pred_adjusted, zero_division=0)
        f1 = f1_score(y_test, y_pred_adjusted, zero_division=0)

        st.metric("Accuracy", f"{acc:.2%}")
        st.metric("Precision", f"{prec:.2%}")
        st.metric("Recall", f"{rec:.2%}")
        st.metric("F1 Score", f"{f1:.2%}")
        st.markdown(f"**Decision Threshold:** `{threshold}` — lower it to increase recall, higher it to increase precision.")

        cm = confusion_matrix(y_test, y_pred_adjusted)
        st.write("**Confusion Matrix:**")
        st.write(pd.DataFrame(cm, index=["Actual No Churn", "Actual Churn"],
                              columns=["Predicted No Churn", "Predicted Churn"]))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.3f})",
                          labels=dict(x="False Positive Rate", y="True Positive Rate"))
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, y0=0, x1=1, y1=1)
        st.plotly_chart(fig_roc)

        # Feature importance (gain)
        importance = model.feature_importances_
        coef_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(coef_df)

        # --- SHAP explanations ---
        st.subheader("4️⃣ SHAP Feature Interpretability")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Summary plot (force matplotlib figure to show)
        fig_shap, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        st.pyplot(fig_shap)

        st.markdown("""
        SHAP values explain the impact of each feature on the model's predictions.  
        Higher SHAP value means a feature pushes prediction towards churn, lower means reduces churn risk.
        """)

    else:
        # Revenue Regression (no tuning here, but can be added similarly)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("R²", f"{r2:.3f}")

        fig_scatter = px.scatter(x=y_test, y=y_pred,
                                 labels={"x": "Actual Revenue", "y": "Predicted Revenue"},
                                 title="Predicted vs. Actual Revenue")
        st.plotly_chart(fig_scatter)

        coef_df = pd.DataFrame({
            "Feature": selected_features,
            "Coefficient": model.coef_
        }).sort_values(by="Coefficient", key=abs, ascending=False)
        st.dataframe(coef_df)

        st.markdown("""
        ✅ **Interpretation:**  
        - **Positive coefficients** increase revenue  
        - **Negative coefficients** decrease revenue  
        - Look for biggest drivers to inform strategy.
        """)
