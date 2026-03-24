import streamlit as st
import pickle
import pandas as pd

# Load your trained model and scaler
with open("tabular_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Prediction App")

# Features to ask from the user
user_features = {
    "Age": st.number_input("Age", min_value=0, max_value=120, value=25),
    "IUD (years)": st.number_input("IUD (years)", min_value=0, max_value=50, value=0),
    "STDs": st.selectbox("STDs", [0, 1]),
    "Num of pregnancies": st.number_input("Num of pregnancies", min_value=0, max_value=20, value=0)
}

# All features that the model expects
model_features = [
    "Age",
    "First sexual intercourse",
    "Hormonal Contraceptives",
    "Hormonal Contraceptives (years)",
    "IUD (years)",
    "Num of pregnancies",
    "STDs",
    "Other_feature_1",
    "Other_feature_2"
]

# Create input DataFrame with defaults
input_data = {}
for feat in model_features:
    if feat in user_features:
        input_data[feat] = [user_features[feat]]
    else:
        # Fill missing features with 0 or default value
        input_data[feat] = [0]

df = pd.DataFrame(input_data)

# Scale features if scaler is used
df_scaled = scaler.transform(df)

# Prediction
prediction = model.predict(df_scaled)

st.write(f"Prediction: {prediction[0]}")
