import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Simple Tabular Predictor", layout="centered")
st.title("Tabular Data Prediction App")

# =========================
# Load Models
# =========================
# Make sure you've uploaded these to Streamlit Cloud
scaler = joblib.load("scaler.joblib")
tabular_model = joblib.load("tabular_model.joblib")

# =========================
# User Input
# =========================
st.header("Enter Input Values")

# Replace with your actual feature names
feature_names = ["feature1", "feature2", "feature3"]  
user_input = {}

for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter {feature}:")

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Scale input
input_scaled = scaler.transform(input_df)

# =========================
# Make Prediction
# =========================
prediction = tabular_model.predict(input_scaled)

st.subheader("Prediction Result")
st.write(prediction[0])
