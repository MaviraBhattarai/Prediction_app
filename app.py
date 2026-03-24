import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.title("Health Prediction App")

# Load model and scaler
with open("tabular_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# All feature names from training
feature_names = ['Age', 'Number of sexual partners', 'First sexual intercourse', 
                 'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
                 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 
                 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
                 'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
                 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                 'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
                 'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
                 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
                 'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis']

# Features we want user to input
user_features = ["Age", "IUD", "STDs", "Num of pregnancies"]

# Get user input
data = {}
st.subheader("Enter your information")
for feature in user_features:
    data[feature] = st.text_input(feature, "0")  # default 0

# Convert to numeric
user_input = pd.DataFrame([data]).apply(pd.to_numeric, errors='coerce').fillna(0)

# Fill missing features with 0
for feature in feature_names:
    if feature not in user_input.columns:
        user_input[feature] = 0

# Reorder columns to match training
user_input = user_input[feature_names]

# Scale input
input_scaled = scaler.transform(user_input)

# Make prediction
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader("Prediction")
st.write(prediction[0])

st.subheader("Prediction Probability")
st.write(prediction_proba[0])
