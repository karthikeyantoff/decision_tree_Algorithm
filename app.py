import streamlit as st
import joblib
import numpy as np

# Page configuration (MUST be first Streamlit command)
st.set_page_config(
    page_title="Smart Home Energy Usage Prediction",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Smart Home Energy Usage Prediction")
st.write("Predict household energy usage using a Decision Tree model")

# Load trained model and encoders
model = joblib.load("smart_home_energy_dt.pkl")
ac_encoder = joblib.load("ac_encoder.pkl")
energy_encoder = joblib.load("energy_encoder.pkl")

st.subheader("🔧 Enter Home Details")

# User Inputs
hour = st.slider("Hour of the Day", 0, 23, 12)

temperature = st.slider(
    "Indoor Temperature (°C)",
    min_value=15,
    max_value=45,
    value=30
)

ac_status = st.selectbox("Air Conditioner", ["No", "Yes"])

occupants = st.slider(
    "Number of Occupants",
    min_value=0,
    max_value=10,
    value=3
)

appliance_load = st.slider(
    "Appliance Load (Active devices)",
    min_value=0,
    max_value=15,
    value=5
)

# Encode categorical input
ac_encoded = ac_encoder.transform([ac_status])[0]

# Prediction
if st.button("Predict Energy Usage"):
    input_data = np.array([[hour, temperature, ac_encoded, occupants, appliance_load]])
    prediction = model.predict(input_data)[0]

    energy_label = energy_encoder.inverse_transform([prediction])[0]

    if energy_label == "High":
        st.error("🔴 High Energy Usage")
    elif energy_label == "Medium":
        st.warning("🟠 Medium Energy Usage")
    else:
        st.success("🟢 Low Energy Usage")