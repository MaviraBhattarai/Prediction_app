import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import joblib

st.title("Cervical Cancer Risk Prediction App")

# =========================
# Load Tabular Model
# =========================
@st.cache_resource
def load_tabular():
    model = joblib.load("tabular_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

tabular_model, scaler = load_tabular()

# =========================
# Load CNN Model (EfficientNet)
# =========================
@st.cache_resource
def load_cnn():
    model = models.efficientnet_b0(pretrained=False)

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 4)
    )

    model.load_state_dict(torch.load("cnn_weights.pth", map_location="cpu"))
    model.eval()
    return model

cnn_model = load_cnn()

# =========================
# Feature List (IMPORTANT)
# =========================
feature_names = [
    'Age', 'Number of sexual partners', 'First sexual intercourse', 
    'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
    'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD', 
    'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
    'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
    'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
    'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
    'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
    'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'
]

# =========================
# User Input (only important ones)
# =========================
st.subheader("Enter Patient Information")

user_features = {
    "Age": "Age",
    "Number of sexual partners": "Number of sexual partners",
    "First sexual intercourse": "First sexual intercourse",
    "Num of pregnancies": "Num of pregnancies",
    "Hormonal Contraceptives": "Hormonal Contraceptives",
    "STDs: HIV": "STDs:HIV"
}

user_data = {}
for label, col in user_features.items():
    user_data[col] = st.number_input(label, min_value=0, value=0)

# Convert to DataFrame
input_df = pd.DataFrame([user_data])

# Fill missing features with 0
for feature in feature_names:
    if feature not in input_df.columns:
        input_df[feature] = 0

# Reorder columns
input_df = input_df[feature_names]

# =========================
# Tabular Prediction
# =========================
scaled_input = scaler.transform(input_df)
tabular_prob = tabular_model.predict_proba(scaled_input)[0][1]

st.subheader("Tabular Model Result")
st.write(f"Probability: {tabular_prob:.2f}")

# =========================
# CNN Prediction (Image)
# =========================
st.subheader("Upload Pap Smear Image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

cnn_prob = 0.0  # default

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(image_tensor)
        probs = F.softmax(output, dim=1)

        # Cancer-related probability (same logic as your notebook)
        cnn_prob = (probs[:, 0] + probs[:, 1]).item()

    st.subheader("CNN Model Result")
    st.write(f"Probability: {cnn_prob:.2f}")

# =========================
# FINAL MULTIMODAL PREDICTION
# =========================
final_score = 0.6 * tabular_prob + 0.4 * cnn_prob

risk = "High Risk" if final_score > 0.45 else "Low Risk"

st.subheader("Final Risk Prediction")
st.write(risk)
st.write(f"Final Risk Score: {final_score:.2f}")
