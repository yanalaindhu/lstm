import streamlit as st
import numpy as np
import joblib

from tensorflow.keras.models import load_model

# Load Files
model = load_model("models/lstm_model.keras")

scaler = joblib.load("models/scaler.pkl")

lookback = joblib.load("models/lookback.pkl")

st.title("✈ Airline Passenger Forecasting")

st.write(
    f"Enter last {lookback} passenger values"
)

values = []

for i in range(lookback):
    val = st.number_input(
        f"Month {i+1}",
        min_value=0.0,
        value=100.0
    )
    values.append(val)

if st.button("Predict"):

    arr = np.array(values).reshape(-1,1)

    scaled = scaler.transform(arr)

    X = scaled.reshape(
        1,
        lookback,
        1
    )

    pred = model.predict(X)

    pred = scaler.inverse_transform(pred)

    st.success(
        f"Predicted Passengers: {int(pred[0][0])}"
    )