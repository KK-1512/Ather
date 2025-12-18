import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(page_title="EV Range Predictor", page_icon="⚡")

# Title
st.title("⚡ EV Scooter Range Predictor")
st.write("Mini ML demo showcasing Linear Regression model")

# Load Model
model = joblib.load("Ather.pkl")

# User Inputs
battery = st.slider("Battery Level (%)", 10, 100, 60)
speed = st.slider("Average Speed (km/h)", 20, 80, 40)
temp = st.slider("Temperature (°C)", 15, 45, 30)
mode = st.selectbox("Riding Mode", ["Eco", "Normal", "Sport"])

# Run Prediction
if st.button("Predict Range (km)"):
    user_df = pd.DataFrame({
        "battery_level": [battery],
        "avg_speed": [speed],
        "temperature": [temp],
        "riding_mode": [mode]
    })
    
    prediction = model.predict(user_df)[0]
    st.success(f"Estimated Range: {prediction:.2f} km")
