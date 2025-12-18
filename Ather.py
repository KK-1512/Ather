import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="EV Range Predictor", page_icon="⚡")

st.title("⚡ EV Scooter Range Predictor (Linear Regression Model)")
st.write("Enter parameters to estimate riding range")

# Load model
model = joblib.load("ev_model.pkl")

# Inputs
battery = st.slider("Battery Level (%)", min_value=10, max_value=100, value=60)
speed = st.slider("Average Speed (km/h)", min_value=20, max_value=80, value=40)
temp = st.slider("Temperature (°C)", min_value=15, max_value=45, value=30)
mode = st.selectbox("Riding Mode", ["Eco", "Normal", "Sport"])

# Predict button
if st.button("Predict Range (km)"):
    user_df = pd.DataFrame({
        "battery_level": [battery],
        "avg_speed": [speed],
        "temperature": [temp],
        "riding_mode": [mode]
    })
    result = model.predict(user_df)[0]
    st.success(f"Estimated Riding Range: {result:.2f} km")
