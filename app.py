import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Cancer Risk Predictor", layout="wide")

st.title("🧬 Cervical Cancer Risk Prediction System")
st.markdown("This app predicts cancer risk using **clinical data + Pap smear images**.")

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_tabular():
    model = joblib.load("tabular_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

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

tabular_model, scaler = load_tabular()
cnn_model = load_cnn()

# =========================
# FEATURE LIST
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
# SIDEBAR INPUTS
# =========================
st.sidebar.header("📋 Patient Information")

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
    user_data[col] = st.sidebar.number_input(label, min_value=0, value=0)

# =========================
# IMAGE UPLOAD
# =========================
st.sidebar.header("🖼️ Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload Pap Smear Image", type=["jpg", "jpeg", "png"])

# =========================
# PREDICT BUTTON
# =========================
if st.sidebar.button("🔍 Predict Risk"):

    with st.spinner("Analyzing patient data and image..."):

        # -------------------------
        # Prepare Tabular Data
        # -------------------------
        input_df = pd.DataFrame([user_data])

        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[feature_names]

        scaled_input = scaler.transform(input_df)
        tabular_prob = tabular_model.predict_proba(scaled_input)[0][1]

        # -------------------------
        # CNN Prediction
        # -------------------------
        cnn_prob = 0.0

        col1, col2 = st.columns(2)

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

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
                cnn_prob = (probs[:, 0] + probs[:, 1]).item()

            # Show image
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

        # -------------------------
        # FINAL SCORE
        # -------------------------
        Final Score = 0.5 × Tabular + 0.5 × CNN

        # -------------------------
        # DISPLAY RESULTS
        # -------------------------
        with col2:
            st.subheader("📊 Results")

            st.metric("Tabular Probability", f"{tabular_prob:.2f}")
            st.metric("CNN Probability", f"{cnn_prob:.2f}")
            st.metric("Final Risk Score", f"{final_score:.2f}")

            st.progress(int(final_score * 100))

            if final_score > 0.45:
                st.error("🔴 High Risk")
            else:
                st.success("🟢 Low Risk")

        st.info("Prediction is based on combined clinical and image analysis.")
