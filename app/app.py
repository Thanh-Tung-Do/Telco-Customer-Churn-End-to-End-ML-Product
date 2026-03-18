"""
Customer Churn Predictor — Streamlit App

Run from project root:
    streamlit run app/app.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import ALL_FEATURE_COLS

# ── Constants (must match src/model_utils.py) ────────────────────────────────
COST_FN = 500    # Cost of missing a churner ($) — lost CLV
COST_FP = 50     # Cost of a wasted retention offer ($)
VALUE_TP = 300   # Net value of successfully retaining a churner ($)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "models", "xgboost_pipeline.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

st.title("📉 Customer Churn Predictor")
st.caption(
    "Telco customer retention tool — enter a customer's profile to get their churn "
    "probability and a recommended retention action."
)

# ── Sidebar: Customer Profile Inputs ─────────────────────────────────────────
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
total_charges = float(monthly_charges * tenure)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No"
)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

st.sidebar.subheader("Services")
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["No", "Yes", "No phone service"]
)
internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)
online_security = st.sidebar.selectbox(
    "Online Security", ["No", "Yes", "No internet service"]
)
online_backup = st.sidebar.selectbox(
    "Online Backup", ["No", "Yes", "No internet service"]
)
device_protection = st.sidebar.selectbox(
    "Device Protection", ["No", "Yes", "No internet service"]
)
tech_support = st.sidebar.selectbox(
    "Tech Support", ["No", "Yes", "No internet service"]
)
streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["No", "Yes", "No internet service"]
)
streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["No", "Yes", "No internet service"]
)

st.sidebar.subheader("Billing")
contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

# ── Build input row ───────────────────────────────────────────────────────────
input_data = pd.DataFrame(
    [
        {
            "SeniorCitizen": senior_citizen,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "PaperlessBilling": paperless_billing,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
    ]
)[ALL_FEATURE_COLS]

# ── Predict ───────────────────────────────────────────────────────────────────
try:
    model = load_model()
    churn_prob = float(model.predict_proba(input_data)[0, 1])

    # Risk tier
    if churn_prob >= 0.5:
        risk_level, risk_icon, action = (
            "High",
            "🔴",
            "Escalate — offer retention package immediately",
        )
    elif churn_prob >= 0.25:
        risk_level, risk_icon, action = (
            "Medium",
            "🟡",
            "Monitor — send a targeted loyalty discount",
        )
    else:
        risk_level, risk_icon, action = "Low", "🟢", "No action required"

    # ── Top metrics row ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{churn_prob:.1%}")
    col2.metric("Risk Level", f"{risk_icon} {risk_level}")
    col3.metric("Recommended Action", action)

    # ── Probability bar ───────────────────────────────────────────────────────
    st.progress(churn_prob, text=f"{churn_prob:.1%} likelihood of churning")

    if churn_prob >= 0.5:
        st.error(
            f"⚠️ **High churn risk ({churn_prob:.1%}).** This customer shows multiple "
            f"risk factors. Proactive outreach is recommended."
        )
    elif churn_prob >= 0.25:
        st.warning(
            f"⚡ **Moderate churn risk ({churn_prob:.1%}).** Consider a loyalty offer "
            f"or check-in call."
        )
    else:
        st.success(f"✅ **Low churn risk ({churn_prob:.1%}).** No immediate action needed.")

    # ── Business cost panel ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Business Impact of This Prediction")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        "If we miss this churner (FN)",
        f"−${COST_FN:,}",
        help="Estimated lost CLV if this customer churns and we do nothing",
    )
    col_b.metric(
        "Cost of retention offer (FP)",
        f"−${COST_FP:,}",
        help="Discount + agent time if customer was not actually going to churn",
    )
    col_c.metric(
        "Value if successfully retained (TP)",
        f"+${VALUE_TP:,}",
        help="Net saving after retention spend",
    )

    # ── Profile summary ───────────────────────────────────────────────────────
    with st.expander("Full customer profile"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

except FileNotFoundError:
    st.error(
        "**Model file not found.** Please run `notebooks/02_modeling.ipynb` first to "
        "train and save the XGBoost pipeline, then relaunch this app."
    )
    st.code("jupyter nbconvert --to notebook --execute notebooks/02_modeling.ipynb --output notebooks/02_modeling.ipynb")
