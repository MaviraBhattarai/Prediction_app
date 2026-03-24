import streamlit as st
import pandas as pd
import joblib

st.title("Simple Prediction App")

# Load scaler and model
scaler = joblib.load("scaler.joblib")
tabular_model = joblib.load("tabular_model.joblib")

# Features to ask from the user
feature_names = ["Age", "IUD (years)", "STDs", "Num of pregnancies"]

# Collect user input
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter {feature}:", value=0, min_value=0)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = tabular_model.predict(input_scaled)

st.subheader("Prediction Result")
st.write(prediction[0])
