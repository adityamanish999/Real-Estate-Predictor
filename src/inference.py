"""Turns raw property details into the feature vector the trained models expect.

This mirrors src/preprocessing.py. Keep the two in sync: tests/test_inference.py
checks that both produce identical features for real rows of the dataset.
"""
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = 'models'
DATASET_YEAR = 2025  # Age_of_Property in the dataset is 2025 - Year_Built

NUM_COLS = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 'Year_Built',
            'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools',
            'Nearby_Hospitals', 'Amenity_Score']

# Categorical columns that preprocessing.py one-hot encodes (drop_first=True).
CATEGORICAL_COLS = ['Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility',
                    'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status']


def load_artifacts(models_dir=MODELS_DIR):
    return {
        'clf': joblib.load(f'{models_dir}/best_classifier.pkl'),
        'reg': joblib.load(f'{models_dir}/best_regressor.pkl'),
        'scaler': joblib.load(f'{models_dir}/scaler.pkl'),
        'features': joblib.load(f'{models_dir}/feature_columns.pkl'),
        'encodings': {col: joblib.load(f'{models_dir}/{col}_encoding.pkl')
                      for col in ['State', 'City', 'Locality']},
    }


def build_input_row(raw, features, encodings, scaler):
    """raw: dict with the original dataset columns (State, City, Locality, BHK,
    Size_in_SqFt, Price_in_Lakhs, Year_Built, Amenities, and the categorical columns).
    Returns a one-row DataFrame, scaled and ordered like the training data."""
    row = {
        'BHK': raw['BHK'],
        'Size_in_SqFt': raw['Size_in_SqFt'],
        'Price_in_Lakhs': raw['Price_in_Lakhs'],
        # Training data stores price per sq ft in lakhs, rounded to 2 decimals.
        'Price_per_SqFt': round(raw['Price_in_Lakhs'] / raw['Size_in_SqFt'], 2),
        'Year_Built': raw['Year_Built'],
        'Floor_No': raw['Floor_No'],
        'Total_Floors': raw['Total_Floors'],
        'Age_of_Property': DATASET_YEAR - raw['Year_Built'],
        'Nearby_Schools': raw['Nearby_Schools'],
        'Nearby_Hospitals': raw['Nearby_Hospitals'],
        'Amenity_Score': len(str(raw['Amenities']).split(',')),
    }
    for col, enc in encodings.items():
        # Unseen values fall back to the average encoding.
        row[f'{col}_Encoded'] = enc.get(raw[col], float(np.mean(list(enc.values()))))

    # One-hot columns are named <Col>_<value>; the dropped baseline category has no column.
    for col in CATEGORICAL_COLS:
        row[f'{col}_{raw[col]}'] = 1

    df = pd.DataFrame([row])
    for col in features:
        if col not in df.columns:
            df[col] = 0
    df = df[features].astype(float)
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])
    return df
