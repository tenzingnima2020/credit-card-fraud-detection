# Streamlit Fraud Detection App

# importing libraries

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered"
)

# ── Load the saved model and feature columns ─────────────────
# We load these once when the app starts (Streamlit caches them).
# @st.cache_resource prevents the model from reloading on every interaction.

@st.cache_resource
def load_model():
    model = joblib.load('models/fraud_model.pkl')
    columns = joblib.load('models/model_columns.pkl')
    return model, columns

model, model_columns = load_model()

# ── App Header ────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detector")
st.markdown("Enter transaction details below to check if it is likely fraudulent.")
st.markdown("---")

# ── Input Form ────────────────────────────────────────────────
# We use Streamlit widgets to collect transaction details from the user.
# These map to the features our model was trained on.

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "💵 Transaction Amount ($)",
        min_value=0.01, max_value=50000.0, value=100.0, step=1.0,
        help="The dollar amount of the transaction"
    )
    trans_hour = st.slider(
        "🕐 Transaction Hour (24hr)",
        min_value=0, max_value=23, value=14,
        help="Hour of the day when the transaction occurred (0 = midnight)"
    )
    age = st.number_input(
        "👤 Cardholder Age",
        min_value=18, max_value=100, value=35,
        help="Age of the credit card holder"
    )

with col2:
    distance = st.number_input(
        "📍 Distance from Merchant",
        min_value=0.0, max_value=500.0, value=5.0, step=0.5,
        help="Euclidean distance between cardholder location and merchant"
    )
    gender = st.selectbox(
        "⚧ Gender",
        options=["Male", "Female"],
        help="Gender of the cardholder"
    )
    city_pop_log = st.number_input(
        "🏙️ City Population (log scale)",
        min_value=0.0, max_value=15.0, value=10.0, step=0.5,
        help="Log of the city population (e.g. log(50000) ≈ 10.8)"
    )

# Derived feature — is_night is calculated automatically from the hour
is_night = 1 if (trans_hour >= 22 or trans_hour <= 4) else 0

if is_night:
    st.info("🌙 This transaction occurs during **late-night hours** — a higher-risk window for fraud.")

st.markdown("---")

# ── Prediction ────────────────────────────────────────────────
# When the user clicks Predict, we:
# 1. Build an input row with all required features (zeros for one-hot columns)
# 2. Fill in the values from the form
# 3. Pass it to the model and show the result

if st.button("🔍 Predict", use_container_width=True):

    # Create a row of zeros with all the columns the model expects
    input_dict = {col: 0 for col in model_columns}

    # Fill in the known numeric features
    input_dict['amount']       = amount
    input_dict['trans_hour']   = trans_hour
    input_dict['age']          = age
    input_dict['distance']     = distance
    input_dict['gender']       = 1 if gender == "Male" else 0
    input_dict['city_pop_log'] = city_pop_log
    input_dict['is_night']     = is_night

    input_df = pd.DataFrame([input_dict])

    # Get prediction and fraud probability
    prediction  = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("### 📊 Result")

    if prediction == 1:
        st.error(f"🚨 **FRAUD DETECTED**")
        st.metric(label="Fraud Probability", value=f"{probability:.1%}")
        st.markdown(
            "> This transaction has been flagged as likely fraudulent. "
            "A real system would block this and notify the cardholder."
        )
    else:
        st.success(f"✅ **Transaction Looks Legitimate**")
        st.metric(label="Fraud Probability", value=f"{probability:.1%}")
        st.markdown(
            "> The model does not consider this transaction suspicious. "
            "It would be approved in a real system."
        )

    # Show a probability gauge bar
    st.markdown("**Fraud Risk Level:**")
    st.progress(float(probability))

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Built with Random Forest · Trained on Credit Card Transaction Data · "
    "Co-op Portfolio Project</small>",
    unsafe_allow_html=True
)
