import streamlit as st
import pickle
import pandas as pd
from joblib import load

st.title("Health Prediction App")

# Load model and scaler
with open("tabular_model.joblib", "rb") as f:
    model = load(f)

with open("scaler.joblib", "rb") as f:
    scaler = load(f)

# All features used during training
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

# Features we ask the user for
user_features = {
    "Age": "Age",
    "Number of sexual partners": "Number of sexual partners",
    "First sexual intercourse": "First sexual intercourse",
    "Num of pregnancies": "Num of pregnancies",
    "Hormonal Contraceptives": "Hormonal Contraceptives",
    "STDs:HIV": "STDs:HIV"
}

# Collect user input
st.subheader("Enter your information")
user_input_data = {}
for label, col_name in user_features.items():
    user_input_data[col_name] = st.number_input(label, min_value=0, value=0)

# Convert input to DataFrame
user_input_df = pd.DataFrame([user_input_data])

# Fill missing features with 0 so all features are present
for feature in feature_names:
    if feature not in user_input_df.columns:
        user_input_df[feature] = 0

# Reorder columns to match training
user_input_df = user_input_df[feature_names]

# Scale and predict
input_scaled = scaler.transform(user_input_df)
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Show results as High/Low Risk
st.subheader("Prediction")
# Determine risk based on probability threshold
risk = "High Risk" if prediction_proba[0][1] > 0.45 else "Low Risk"

# Show the risk
st.subheader("Risk Score")
st.write(risk)
