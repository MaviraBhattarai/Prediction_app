import streamlit as st
import pickle
import joblib
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

st.title("Health Prediction App with Tabular + CNN Models")

# -------------------------------
# Load Tabular Model and Scaler
# -------------------------------
with open("tabular_model.joblib", "rb") as f:
    tabular_model = joblib.load(f)

with open("scaler.joblib", "rb") as f:
    scaler = joblib.load(f)

# -------------------------------
# Define Features
# -------------------------------
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
    "STDs: HIV": "STDs:HIV",
    "IUD (years)": "IUD (years)"
}

# -------------------------------
# Collect Tabular User Input
# -------------------------------
st.subheader("Enter your information")
user_input_data = {}
for label, col_name in user_features.items():
    user_input_data[col_name] = st.text_input(label, "0")

# Convert to numeric and fill missing features
user_input_df = pd.DataFrame([user_input_data]).apply(pd.to_numeric, errors='coerce').fillna(0)
for feature in feature_names:
    if feature not in user_input_df.columns:
        user_input_df[feature] = 0

user_input_df = user_input_df[feature_names]

# -------------------------------
# Tabular Prediction
# -------------------------------
input_scaled = scaler.transform(user_input_df)
tabular_pred = tabular_model.predict(input_scaled)
tabular_proba = tabular_model.predict_proba(input_scaled)[0][1]  # probability of high risk

st.subheader("Tabular Model Prediction")
st.write(f"Tabular Risk: {'High Risk' if tabular_proba > 0.45 else 'Low Risk'}")
st.write(f"Probability: {tabular_proba:.2f}")

# -------------------------------
# CNN Image Prediction
# -------------------------------
st.subheader("Upload Image for CNN Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Load CNN model
    cnn_model = torch.load("cnn_model.pth", map_location=torch.device('cpu'), weights_only=False)
    cnn_model.eval()

    with torch.no_grad():
        cnn_output = cnn_model(image_tensor)
        cnn_proba = torch.sigmoid(cnn_output).item()  # probability of high risk

    st.subheader("CNN Model Prediction")
    st.write(f"CNN Risk: {'High Risk' if cnn_proba > 0.45 else 'Low Risk'}")
    st.write(f"Probability: {cnn_proba:.2f}")

    # -------------------------------
    # Combined Prediction
    # -------------------------------
    final_risk_score = 0.6 * tabular_proba + 0.4 * cnn_proba
    combined_risk = "High Risk" if final_risk_score > 0.45 else "Low Risk"

    st.subheader("Combined Prediction")
    st.write(f"Overall Risk: {combined_risk}")
    st.write(f"Combined Risk Score: {final_risk_score:.2f}")
