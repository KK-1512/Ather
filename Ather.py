import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="EV Range Predictor", page_icon="⚡")

st.title("⚡ EV Scooter Range Predictor")
st.caption("Mini Machine Learning demo for interview presentation")

# Load model
model = joblib.load("model/ev_range_model.pkl")

battery = st.slider("Battery Level (%)", 10, 100, 60)
speed = st.slider("Average Speed (km/h)", 20, 80, 40)
temp = st.slider("Temperature (°C)", 15, 45, 30)
mode = st.selectbox("Riding Mode", ["Eco", "Normal", "Sport"])

if st.button("Predict Range"):
    sample = np.array([[battery, speed, temp, mode]])
    prediction = model.predict(sample)[0]
    st.success(f"Estimated Range: {prediction:.2f} km")
