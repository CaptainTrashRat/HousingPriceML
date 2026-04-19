import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load model assets ---
model           = joblib.load('housing_model.joblib')
scaler          = joblib.load('scaler.joblib')
feature_columns = joblib.load('feature_columns.joblib')
mean_zip_region = joblib.load('mean_zip_region.joblib')

st.title("🏠 Housing Price Predictor")
st.markdown("Enter the property details below to get a predicted price.")

# --- Inputs ---
col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Property Type", ["Single Family", "Condo", "Townhouse"])
    bedrooms      = st.number_input("Bedrooms", min_value=1, max_value=20, value=3)
    bathrooms     = st.number_input("Bathrooms", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
    sqft          = st.number_input("Living Area (sqft)", min_value=100, max_value=20000, value=1500)

with col2:
    zip_code   = st.text_input("Zip Code (optional)", value="",
                               placeholder="Leave blank to use average location")
    lot_acres  = st.number_input("Lot Size (acres)", min_value=0.01, max_value=100.0, value=0.25, step=0.01)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=1990)

# --- Predict ---
if st.button("Predict Price", type="primary"):

    # Resolve zip region
    zip_provided = zip_code.strip() != ""
    if zip_provided:
        if not zip_code.strip().isdigit() or len(zip_code.strip()) < 3:
            st.error("Please enter a valid zip code (at least 3 digits) or leave it blank.")
            st.stop()
        zip_region = int(zip_code.strip()[:3])
    else:
        zip_region = mean_zip_region  # fallback: average location

    # Build feature dict
    house_age = 2024 - year_built

    input_dict = {
        'bedrooms':           bedrooms,
        'bathrooms':          bathrooms,
        'sqft':               sqft,
        'lot_acres':          lot_acres,
        'house_age':          house_age,
        'type_Condo':         property_type == 'Condo',
        'type_Single Family': property_type == 'Single Family',
        'type_Townhouse':     property_type == 'Townhouse',
        'zip_region':         zip_region,
    }

    input_df = pd.DataFrame([input_dict])[feature_columns]

    # Scale numeric columns
    num_cols = ['bedrooms', 'bathrooms', 'sqft', 'lot_acres', 'house_age', 'zip_region']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict (model trained on log price)
    log_pred        = model.predict(input_df)[0]
    predicted_price = np.exp(log_pred)

    st.success(f"### Predicted Price: ${predicted_price:,.0f}")
    st.caption(f"Model accuracy: MAE ≈ $126,500 | R² = 0.716 | MAPE ≈ 21%")

    if not zip_provided:
        st.warning("No zip code provided — prediction uses average location pricing. "
                   "Adding a zip code may improve accuracy.")

    # Confidence range based on MAPE
    low  = predicted_price * 0.79
    high = predicted_price * 1.21
    st.info(f"Estimated range (±21%): **${low:,.0f}** – **${high:,.0f}**")
