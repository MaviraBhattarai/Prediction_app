# Scale and predict
input_scaled = scaler.transform(user_input_df)
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

# Convert prediction to friendly label
risk_label = "High Risk" if prediction[0] == 1 else "Low Risk"

# Show results
st.subheader("Prediction")
st.write(risk_label)

st.subheader("Prediction Probability")
st.write({
    "Low Risk": round(prediction_proba[0][0], 2),
    "High Risk": round(prediction_proba[0][1], 2)
})
