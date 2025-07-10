import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, mean_squared_error, precision_score,
                             recall_score, f1_score, r2_score, roc_curve, auc, confusion_matrix)
from sklearn.preprocessing import OneHotEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="SaaS Customer Churn & Revenue Prediction", layout="wide")
st.title("SaaS Customer Churn & Revenue Prediction Dashboard")
st.markdown("""
### Business Context  
This dashboard simulates how a payment platform (like Stripe) analyzes merchant customers to:  
- Predict **churn risk** for proactive retention  
- Predict **monthly revenue** to guide financial planning

This version fixes class imbalance issues & lets you tune your decision threshold to see real tradeoffs.
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
        "is_active": np.random.choice([0, 1], n, p=[0.4, 0.6])  # More inactive now!
    })
    # More generous churn condition — boost churn % to ~30–40%
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
    df["engagement_score"] = df["monthly_txn_volume"] * (df["subscription_age_days"] / 365)
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
    "engagement_score"
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
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred_adjusted = (y_prob >= threshold).astype(int)

        acc = model.score(X_test, y_test)
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

        coef_df = pd.DataFrame({
            "Feature": selected_features,
            "Coefficient": model.coef_[0]
        }).sort_values(by="Coefficient", key=abs, ascending=False)
        st.dataframe(coef_df)

        st.markdown("""
        ✅ **Interpretation:**  
        - **Positive coefficients** increase churn risk  
        - **Negative coefficients** reduce churn risk  
        - **Adjust threshold** to trade precision vs recall
        """)

    else:
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

st.markdown("### 🔍 Feature-by-Feature Interpretation")

# Create a list of (feature, coef, abs(coef))
coef_info = [(feat, coef, abs(coef)) for feat, coef in zip(selected_features, model.coef_[0])]

# Sort by absolute value, descending
coef_info.sort(key=lambda x: x[2], reverse=True)

feature_explanations = []

# Adjust these thresholds as needed for your scale:
tiny_threshold = 0.01
moderate_threshold = 0.05

for feat, coef, abs_coef in coef_info:
    # Plain-English meaning
    if feat == "subscription_age_days":
        meaning = "the number of days the merchant has been subscribed"
    elif feat == "monthly_txn_volume":
        meaning = "the number of transactions processed per month"
    elif feat == "avg_txn_value":
        meaning = "the average value of each transaction"
    elif feat == "support_tickets":
        meaning = "the number of support tickets opened"
    elif feat == "engagement_score":
        meaning = "a composite score: transaction volume weighted by subscription age"
    elif feat.startswith("pay_"):
        payment_type = feat.replace("pay_", "")
        meaning = f"whether the merchant primarily uses {payment_type} as their payment method"
    else:
        meaning = "a custom feature"

    # Sign-based impact direction
    if coef > 0:
        direction = "increases churn risk"
    elif coef < 0:
        direction = "reduces churn risk"
    else:
        direction = "has no impact on churn risk"

    # Magnitude-based nuance
    if abs_coef < tiny_threshold:
        magnitude = "has **no meaningful impact**"
    elif abs_coef < moderate_threshold:
        magnitude = f"has a **small but noticeable effect** and generally {direction}"
    else:
        magnitude = f"has a **strong impact** and clearly {direction}"

    feature_explanations.append(
        f"- **{feat}** represents *{meaning}*. Its coefficient is `{coef:.3f}`, so it {magnitude}."
    )

st.markdown("\n".join(feature_explanations))


st.markdown("---")
st.markdown("🔍 **Balanced churn, adjustable threshold, confusion matrix — realistic and practical, just like you’d do at Stripe!**")
