import streamlit as st
import pickle
import pandas as pd

st.title("Health Prediction App")

# Load model and scaler
with open("tabular_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

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

# Features we actually ask the user for (friendly label -> model column name)
user_features = {
    "Age": "Age",
    "Num of Pregnancies": "Num of pregnancies"
    "IUD (years)": "IUD (years)",
    "STDs": "STDs"
}

# Collect user input
st.subheader("Enter your information")
user_input_data = {}
for label, col_name in user_features.items():
    user_input_data[col_name] = st.text_input(label, "0")  # default 0

# Convert input to numeric
user_input_df = pd.DataFrame([user_input_data]).apply(pd.to_numeric, errors='coerce').fillna(0)

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

# Show results
st.subheader("Prediction")
st.write(prediction[0])

st.subheader("Prediction Probability")
st.write(prediction_proba[0])
