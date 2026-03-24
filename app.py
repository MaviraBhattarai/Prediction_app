import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cervical Cancer Risk Predictor", layout="wide")

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    tabular_model = joblib.load("tabular_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    cnn_model = models.efficientnet_b0(pretrained=False)
    cnn_model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(cnn_model.classifier[1].in_features, 4)  # 4 classes
    )
    cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location='cpu'))
    cnn_model.eval()
    
    return tabular_model, scaler, cnn_model

tabular_model, scaler, cnn_model = load_models()

# -----------------------------
# CNN Transforms
# -----------------------------
IMG_SIZE = 224
cnn_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -----------------------------
# Helper Functions
# -----------------------------
def predict_tabular(input_df):
    X_scaled = scaler.transform(input_df)
    prob = tabular_model.predict_proba(X_scaled)[:,1]
    return prob

def predict_cnn(img):
    img_tensor = cnn_transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        # cancer + lesion probability
        prob = probs[0,0].item() + probs[0,1].item()
    return prob

def generate_gradcam(img):
    # Simplified Grad-CAM using PIL + matplotlib
    img_tensor = cnn_transform(img).unsqueeze(0)
    cnn_model.eval()
    gradients = None
    activations = None
    
    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()
    
    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()
    
    target_layer = cnn_model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    
    output = cnn_model(img_tensor)
    pred_class = output.argmax()
    cnn_model.zero_grad()
    output[0, pred_class].backward()
    
    pooled_gradients = torch.mean(gradients, dim=[0,2,3])
    activ = activations[0]
    for i in range(len(pooled_gradients)):
        activ[i] *= pooled_gradients[i]
    
    heatmap = torch.mean(activ, dim=0).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-6
    
    # Resize heatmap using PIL
    heatmap_img = Image.fromarray(np.uint8(255*heatmap))
    heatmap_img = heatmap_img.resize(img.size)
    heatmap = np.array(heatmap_img)/255.0
    heatmap = plt.cm.jet(heatmap)[:,:,:3]  # RGB
    return heatmap

# -----------------------------
# App UI
# -----------------------------
st.title("🩺 Cervical Cancer Risk Predictor")
st.write("Enter patient data and/or upload Pap smear image to get risk score.")

# Tabular Input
st.header("Patient Risk Factors")
uploaded_csv = st.file_uploader("Upload CSV of risk factors (optional)", type=["csv"])
manual_input = {}

if uploaded_csv:
    df_input = pd.read_csv(uploaded_csv)
else:
    st.subheader("Manual Entry")
    # Example features; adapt to your dataset
    feature_names = ['Age', 'Number of sexual partners', 'First sexual intercourse age', 'Num of pregnancies']
    for f in feature_names:
        manual_input[f] = st.number_input(f, min_value=0.0, value=0.0)
    df_input = pd.DataFrame([manual_input])

# Image Input
st.header("Pap Smear Image (optional)")
uploaded_img = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
img = None
if uploaded_img:
    img = Image.open(uploaded_img).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Risk"):
    tabular_prob = predict_tabular(df_input) if not df_input.empty else np.array([0.0])
    cnn_prob = predict_cnn(img) if img is not None else np.array([0.0])
    
    # Fusion
    final_score = 0.6*tabular_prob + 0.4*cnn_prob
    fusion_threshold = 0.45
    prediction = "High Risk" if final_score >= fusion_threshold else "Low Risk"
    
    st.subheader("🔹 Results")
    st.write(f"Tabular Model Probability: {tabular_prob[0]:.3f}")
    st.write(f"CNN Model Probability: {cnn_prob:.3f}")
    st.write(f"**Final Risk Score: {final_score[0]:.3f} → {prediction}**")
    
    # Grad-CAM visualization
    if img is not None:
        st.subheader("📈 Model Attention (Grad-CAM)")
        heatmap = generate_gradcam(img)
        superimposed = 0.4*heatmap + np.array(img)/255.0
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(superimposed)
        ax.axis("off")
        st.pyplot(fig)
