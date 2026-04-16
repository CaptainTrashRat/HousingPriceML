"""
train_model.py
--------------
Trains a Gradient Boosting model to predict housing prices.
Run this script whenever you want to retrain on new data.

Usage:
    python train_model.py
    python train_model.py --data path/to/your_data.csv

Outputs (saved to the same directory as this script):
    housing_model.joblib    — trained model
    scaler.joblib           — fitted StandardScaler
    feature_columns.joblib  — ordered list of feature column names
    mean_zip_region.joblib  — fallback zip region for the Streamlit app
"""

import argparse
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = "housing_data.csv"
TEST_SIZE         = 0.2
RANDOM_STATE      = 42

MODEL_PARAMS = dict(
    n_estimators   = 500,
    learning_rate  = 0.05,
    max_depth      = 5,
    min_samples_leaf = 10,
    subsample      = 0.8,
    random_state   = RANDOM_STATE,
)

NUM_COLS = ['bedrooms', 'bathrooms', 'sqft', 'lot_acres', 'house_age', 'zip_region']


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_prompt(text: str) -> dict:
    """Extract structured features from a natural-language property description."""
    prop_type = re.search(r'Property: (\w[\w\s]*) in zip', text)
    zip_code  = re.search(r'zip (\d+)', text)
    bedrooms  = re.search(r'(\d+) bedrooms', text)
    bathrooms = re.search(r'([\d.]+) bathrooms', text)
    sqft      = re.search(r'(\d+) sqft', text)
    lot       = re.search(r'([\d.]+) acre lot', text)
    year      = re.search(r'built in (\d+)', text)
    return {
        'property_type': prop_type.group(1).strip() if prop_type else None,
        'zip_code':      zip_code.group(1)          if zip_code  else None,
        'bedrooms':      int(bedrooms.group(1))      if bedrooms  else None,
        'bathrooms':     float(bathrooms.group(1))   if bathrooms else None,
        'sqft':          int(sqft.group(1))          if sqft      else None,
        'lot_acres':     float(lot.group(1))         if lot       else None,
        'year_built':    int(year.group(1))          if year      else None,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Parse, clean, and engineer features from raw dataframe."""
    features = df['prompt'].apply(parse_prompt).apply(pd.Series)
    features['price'] = df['price']

    # Fill missing property type with most common value
    features['property_type'] = features['property_type'].fillna(
        features['property_type'].mode()[0]
    )

    # Convert year built -> house age
    features['house_age'] = 2024 - features['year_built']
    features.drop(columns=['year_built'], inplace=True)

    # One-hot encode property type
    features = pd.get_dummies(features, columns=['property_type'], prefix='type')

    # Reduce sparse zip codes to 3-digit regional prefix
    features['zip_region'] = features['zip_code'].str[:3].astype(int)
    features.drop(columns=['zip_code'], inplace=True)

    return features


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(data_path: str):
    print(f"\n📂 Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"   {len(df):,} rows loaded")

    print("\n🔧 Preprocessing...")
    features = preprocess(df)

    # Train / test split
    train_df, test_df = train_test_split(features, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train = train_df.drop(columns=['price'])
    y_train = train_df['price']
    X_test  = test_df.drop(columns=['price'])
    y_test  = test_df['price']

    # Scale numeric features
    scaler = StandardScaler()
    X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test[NUM_COLS]  = scaler.transform(X_test[NUM_COLS])

    # Mean zip region fallback (used by Streamlit when zip is not provided)
    mean_zip_region = train_df['zip_region'].mean()

    print(f"\n🤖 Training model with params: {MODEL_PARAMS}")
    model = GradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X_train, np.log(y_train))   # train on log(price)

    # Evaluate
    preds = np.exp(model.predict(X_test))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    mape  = np.mean(np.abs((y_test - preds) / y_test)) * 100

    print("\n📊 Evaluation on test set:")
    print(f"   MAE:  ${mae:,.0f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   MAPE: {mape:.2f}%")

    print("\n💾 Saving artifacts...")
    joblib.dump(model,                  'housing_model.joblib')
    joblib.dump(scaler,                 'scaler.joblib')
    joblib.dump(list(X_train.columns),  'feature_columns.joblib')
    joblib.dump(mean_zip_region,        'mean_zip_region.joblib')

    print("   ✅ housing_model.joblib")
    print("   ✅ scaler.joblib")
    print("   ✅ feature_columns.joblib")
    print("   ✅ mean_zip_region.joblib")
    print("\n✅ Done! All files saved. Run `streamlit run app.py` to launch the app.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the housing price model.")
    parser.add_argument(
        '--data',
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to the CSV data file (default: {DEFAULT_DATA_PATH})"
    )
    args = parser.parse_args()
    train(args.data)
