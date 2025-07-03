import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

st.set_page_config(page_title="🧠 Predictive Customer Analytics", layout="wide")
st.title("🧠 Predictive Customer Analytics Dashboard")

st.markdown("""
This dashboard helps you explore customer behavior, identify patterns, and predict outcomes such as **churn** or **spend**.  
You can upload your own dataset or use the sample data provided.
""")

# ---- Load or Upload Data ----
def load_sample_data():
    np.random.seed(42)
    n = 1000
    data = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": np.random.randint(18, 70, n),
        "gender": np.random.choice(["Male", "Female"], n),
        "tenure_months": np.random.randint(1, 60, n),
        "monthly_spend": np.round(np.random.normal(80, 30, n), 2),
        "support_calls": np.random.poisson(2, n),
        "active": np.random.choice([0, 1], n, p=[0.3, 0.7])
    })
    data["churn"] = np.where((data["active"] == 0) & (data["support_calls"] > 3), 1, 0)
    return data

st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your customer data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom dataset loaded")
else:
    df = load_sample_data()
    st.sidebar.info("Using built-in sample dataset")

df = df.dropna()
categorical = df.select_dtypes(include="object").columns.tolist()
numerical = df.select_dtypes(include=np.number).columns.tolist()

# ---- Data Preview ----
st.subheader("📄 Data Preview")
st.dataframe(df.head())
st.markdown("""
Sample customer data includes:
- **Demographics**: age, gender
- **Behavior**: spend, support calls, tenure
- **Target**: churn (0/1) or monthly spend (regression)
""")

# ---- Feature Distributions ----
st.subheader("📊 Feature Distributions")
col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="age", nbins=30, title="Customer Age Distribution")
    st.plotly_chart(fig, use_container_width=True)
with col2:
    fig = px.histogram(df, x="monthly_spend", nbins=30, title="Monthly Spend Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ---- Segmentation ----
st.subheader("🧩 Customer Segmentation")
fig = px.box(df, x="gender", y="monthly_spend", color="gender", title="Spend by Gender")
st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(df, x="tenure_months", y="monthly_spend", color=df["churn"].map({0: "No Churn", 1: "Churn"}),
                 title="Spend vs. Tenure with Churn Highlight")
st.plotly_chart(fig, use_container_width=True)

# ---- Prediction Section ----
st.subheader("🔮 Predictive Modeling")
task = st.selectbox("Choose Prediction Task", ["Churn Classification", "Spend Regression"])

target = "churn" if task == "Churn Classification" else "monthly_spend"
features = st.multiselect("Select Features for Model", [col for col in numerical if col != target])

if len(features) > 0:
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task == "Churn Classification":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.metric("Accuracy", f"{report['accuracy']:.2%}")
        st.json(report)
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.metric("RMSE", f"{np.sqrt(mse):.2f}")
        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                         title="Predicted vs. Actual Spend")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("Model coefficients:")
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_ if task == "Spend Regression" else model.coef_[0]
    })
    st.dataframe(coef_df)

else:
    st.info("👈 Select features in the sidebar to train the model.")
