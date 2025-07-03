import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

# --- PAGE CONFIG ---
st.set_page_config(page_title="🧠 Predictive Customer Analytics", layout="wide")
st.title("🧠 Predictive Customer Analytics Dashboard")

st.markdown("""
This dashboard helps you analyze customer behavior, segment users, and **predict churn or spend** using logistic and linear regression.  
You can upload your own dataset or use the built-in sample.
""")

# --- LOAD DATA ---
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

# --- FILE UPLOAD ---
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

# --- DATA PREVIEW ---
st.subheader("📄 1. Data Preview")
st.dataframe(df.head())
st.markdown("""
Each row represents a customer. Columns include:

- `age`: Customer's age
- `gender`: Gender identity
- `tenure_months`: How long the customer has been with the company
- `monthly_spend`: Average spend per month
- `support_calls`: Number of support interactions
- `active`: Whether the customer is active
- `churn`: Whether the customer churned (1 = yes, 0 = no)

Churn is a binary classification target; monthly spend can be used for regression.
""")

# --- DISTRIBUTIONS ---
st.subheader("📊 2. Feature Distributions")

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(df, x="age", nbins=30, title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Shows how customer ages are distributed. Affects spending, churn risk, or product needs.")

with col2:
    fig = px.histogram(df, x="monthly_spend", nbins=30, title="Monthly Spend Distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    Distribution of customer spend. This is the **target variable** in regression tasks.

    Regression predicts a continuous output $y$ given features $X$:
    $$
    \\hat{y} = w_1 x_1 + w_2 x_2 + \\dots + w_n x_n + b
    $$
    """)

# --- SEGMENTATION ---
st.subheader("🧩 3. Customer Segmentation")

fig = px.box(df, x="gender", y="monthly_spend", color="gender", title="Spend by Gender")
st.plotly_chart(fig, use_container_width=True)
st.markdown("""
Box plot shows how monthly spending differs between genders.
Outliers and variability are visualized by box height and whiskers.
""")

fig = px.scatter(df, x="tenure_months", y="monthly_spend", color=df["churn"].map({0: "No Churn", 1: "Churn"}),
                 title="Spend vs. Tenure with Churn Highlight")
st.plotly_chart(fig, use_container_width=True)
st.markdown("""
Longer-tenured users tend to spend more. Customers who churn often cluster with **shorter tenure** and **lower spend**.
""")

# --- PREDICTIVE MODELING ---
st.subheader("🔮 4. Predictive Modeling")
st.markdown("""
We use basic supervised learning models:
- **Logistic Regression** for churn (classification)
- **Linear Regression** for spend (regression)

You can select features and see model accuracy or RMSE (root mean squared error).
""")

task = st.selectbox("Choose Prediction Task", ["Churn Classification", "Spend Regression"])
target = "churn" if task == "Churn Classification" else "monthly_spend"
features = st.multiselect("Select Features", [col for col in numerical if col != target])

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
        st.markdown("""
        Logistic Regression models the probability of churn:

        $$
        P(y = 1 | X) = \\frac{1}{1 + e^{-z}}, \\text{ where } z = w^T X + b
        $$

        If $P > 0.5$, we classify as churn ($y = 1$).
        """)

    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        st.metric("RMSE", f"{rmse:.2f}")
        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual", "y": "Predicted"},
                         title="Predicted vs. Actual Spend")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        Linear Regression fits a line to predict customer spend:
        $$
        \\hat{y} = w_1 x_1 + w_2 x_2 + \\dots + w_n x_n + b
        $$
        RMSE (Root Mean Squared Error) is a common error metric:
        $$
        RMSE = \\sqrt{\\frac{1}{n} \\sum_{i=1}^n (y_i - \\hat{y}_i)^2}
        $$
        """)

    st.markdown("### 🔍 Model Coefficients")
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_ if task == "Spend Regression" else model.coef_[0]
    })
    st.dataframe(coef_df)
    st.markdown("""
    A positive coefficient means the feature **increases** the prediction (churn probability or spend).
    A negative one **reduces** it.
    """)

else:
    st.info("👈 Select features in the sidebar to train a model.")
