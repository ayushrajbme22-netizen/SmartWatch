
import streamlit as st
import pandas as pd
import joblib

# Load trained ML model
@st.cache_resource
def load_model():
    return joblib.load("heart_status_model.joblib")

model = load_model()

# Page configuration
st.set_page_config(
    page_title="Smartwatch Health Monitor",
    page_icon="❤️",
    layout="centered"
)

# Application title
st.title("❤️ Smartwatch Health Monitor")
st.write("Enter your health readings to predict your status using a Machine Learning model.")

st.divider()

# User inputs
col1, col2 = st.columns(2)

with col1:
    spo2 = st.number_input(
        "SpO₂ (%)",
        min_value=50.0,
        max_value=100.0,
        value=98.0,
        step=1.0
    )

with col2:
    pulse = st.number_input(
        "Heart Rate (BPM)",
        min_value=20,
        max_value=250,
        value=75,
        step=1
    )

# Prediction
if st.button("Predict Health Status", use_container_width=True):

    # IMPORTANT: Keep the same feature names and order
    # used during model training.
    input_data = pd.DataFrame(
        [[pulse, spo2]],
        columns=["pulse_rate_ppg_bpm", "spo2_percent"]
    )

    prediction = model.predict(input_data)[0]

    st.divider()
    st.subheader("Prediction Result")

    st.success(f"Predicted Status: {prediction}")

    # Optional prediction confidence
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = max(probabilities) * 100

        st.metric("Model Confidence", f"{confidence:.1f}%")

st.divider()

st.caption(
    "Educational prototype only. This ML model is not a medical "
    "diagnostic tool. Consult a healthcare professional for medical concerns."
)
