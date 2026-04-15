import streamlit as st
import pandas as pd
import joblib

model = joblib.load("house_price_model.pkl")
columns = joblib.load("model_columns.pkl")

st.title("House Price Predictor")

st.write("Enter property details:")

user_input = {}

# Dynamically create inputs based on training columns
for col in columns:
    if "bed" in col.lower():
        user_input[col] = st.number_input(col, 0, 10, 3)
    elif "bath" in col.lower():
        user_input[col] = st.number_input(col, 0.0, 10.0, 2.0)
    elif "zip" in col.lower():
        user_input[col] = st.number_input(col, 10000, 99999, 10001)
    elif "type" in col.lower() or "style" in col.lower():
        user_input[col] = st.selectbox(col, ["house", "condo", "apartment"])
    else:
        user_input[col] = st.number_input(col, 0.0, 100000.0, 1000.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Price: ${prediction:,.0f}")
