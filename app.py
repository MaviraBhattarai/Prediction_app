import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pickle

# =========================
# Load Tabular Model
# =========================
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("tabular_model.pkl", "rb") as f:
    tabular_model = pickle.load(f)

# =========================
# Load CNN Model (EfficientNet B0)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["cancer", "lesion", "normal", "others"]

cnn_model = models.efficientnet_b0(pretrained=False)
cnn_model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(cnn_model.classifier[1].in_features, len(CLASS_NAMES))
)
cnn_model.load_state_dict(torch.load("cnn_model_state.pth", map_location=device))
cnn_model.to(device)
cnn_model.eval()

# =========================
# Image Transform
# =========================
IMG_SIZE = 224
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# =========================
# Streamlit UI
# =========================
st.title("🩺 Cervical Cancer Risk Prediction")
st.write("Upload patient data and Pap smear image to predict risk.")

# Tabular inputs
st.header("1️⃣ Tabular Data Input")
uploaded_csv = st.file_uploader("Upload tabular data CSV", type=["csv"])
if uploaded_csv is not None:
    df_tab = pd.read_csv(uploaded_csv)
    st.dataframe(df_tab.head())

    # Scale and predict
    X_scaled = scaler.transform(df_tab)
    tabular_probs = tabular_model.predict_proba(X_scaled)[:,1]

# Image input
st.header("2️⃣ Pap Smear Image Upload")
uploaded_img = st.file_uploader("Upload Pap smear image", type=["jpg","jpeg","png"])
if uploaded_img is not None:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_tensor = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        # Combine "cancer" + "lesion" classes as risk
        cnn_prob = (probs[0,0] + probs[0,1]).item()

# =========================
# Multimodal Fusion
# =========================
if uploaded_csv is not None and uploaded_img is not None:
    n = min(len(tabular_probs), 1)  # Only 1 image
    final_risk_score = 0.6 * tabular_probs[:n] + 0.4 * cnn_prob
    fusion_pred = (final_risk_score >= 0.45).astype(int)
    
    st.header("3️⃣ Risk Prediction")
    st.write(f"**Predicted Risk Score:** {final_risk_score[0]:.3f}")
    st.write("**High Risk" if fusion_pred[0]==1 else "**Low Risk**")
