import streamlit as st
import joblib
import pandas as pd

st.title("Ather Ride Range Estimator")
st.markdown("### **Done by Krishnakumar**")
st.image("Ather.jpg")


# Load model
model = joblib.load("model.pkl")

battery = st.number_input("Battery Level (%)", 10, 100, 60)
speed = st.number_input("Avg Speed (km/h)", 20, 80, 40)
temp = st.number_input("Temperature (°C)", 15, 45, 30)

if st.button("Predict Range"):
    df = pd.DataFrame([[battery, speed, temp]],
                      columns=['battery_level', 'avg_speed', 'temperature'])
    result = model.predict(df)[0]
    st.success(f"Estimated Range: {result:.2f} km")
