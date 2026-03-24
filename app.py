import streamlit as st
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("Health Prediction App with Tabular + Image Input")

# -----------------------------
# Load models
# -----------------------------
with open("tabular_model.joblib", "rb") as f:
    tab_model = joblib.load(f)
with open("scaler.joblib", "rb") as f:
    scaler = joblib.load(f)

cnn_model = load_model("cnn_model.h5")

# -----------------------------
# Tabular input
# -----------------------------
st.subheader("Enter your information (Tabular Data)")

user_features = ["Age", "Number of sexual partners", "First sexual intercourse", 
                 "Num of pregnancies", "Hormonal Contraceptives", "STDs:HIV"]

user_input_data = {}
for feat in user_features:
    user_input_data[feat] = st.text_input(feat, "")

# Process tabular input
feature_names = tab_model.feature_names_in_
tab_input_df = pd.DataFrame([user_input_data]).apply(pd.to_numeric, errors='coerce').fillna(0)
for feature in feature_names:
    if feature not in tab_input_df.columns:
        tab_input_df[feature] = 0
tab_input_df = tab_input_df[feature_names]

# Scale and predict
tab_scaled = scaler.transform(tab_input_df)
tab_pred_proba = tab_model.predict_proba(tab_scaled)[0][1]  # probability of class 1

# -----------------------------
# Image input
# -----------------------------
st.subheader("Upload your image for CNN prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
cnn_pred_proba = None
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224,224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    cnn_pred_proba = cnn_model.predict(img_array)[0][0]
    st.image(img, caption="Uploaded Image", use_column_width=True)

# -----------------------------
# Combine predictions
# -----------------------------
if cnn_pred_proba is not None:
    final_prob = (tab_pred_proba + cnn_pred_proba) / 2
else:
    final_prob = tab_pred_proba

final_risk = "High Risk" if final_prob > 0.45 else "Low Risk"

# -----------------------------
# Show results
# -----------------------------
st.subheader("Final Risk Prediction")
st.write(f"Overall Risk: {final_risk} ({final_prob:.2f})")

# Optional: show individual model probabilities
st.subheader("Individual Model Probabilities")
st.write(f"Tabular Model Probability: {tab_pred_proba:.2f}")
if cnn_pred_proba is not None:
    st.write(f"CNN Model Probability: {cnn_pred_proba:.2f}")
