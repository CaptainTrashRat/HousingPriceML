import streamlit as st
import pandas as pd
import joblib

model = joblib.load("house_price_model.pkl")

st.title("House Price Predictor")

# Inputs
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1.0, 10.0)
sqft = st.number_input("Square Feet", 300, 10000)
lot_size = st.number_input("Lot Size (acre)", 0.01, 10.0)
year_built = st.number_input("Year Built", 1900, 2025)
zip_code = st.number_input("Zip Code", 10000, 99999)

# Convert to dataframe
input_data = pd.DataFrame([{
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft": sqft,
    "lot_size": lot_size,
    "year_built": year_built,
    "zip": zip_code
}])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ${prediction:,.0f}")