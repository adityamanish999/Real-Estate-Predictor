import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.chdir(ROOT)

from inference import build_input_row, load_artifacts  # noqa: E402

N_ROWS = 2000


@pytest.fixture(scope='module')
def parity_data():
    raw = pd.read_csv('india_housing_prices.csv').iloc[:N_ROWS]
    ready = pd.read_csv('data/model_ready_data.csv').iloc[:N_ROWS]
    return raw, ready


def test_app_features_match_training_features(parity_data):
    """The feature vector built for one property must equal the one preprocessing.py produced."""
    raw, ready = parity_data
    art = load_artifacts()
    mismatched = 0
    for i in range(N_ROWS):
        row = build_input_row(raw.iloc[i].to_dict(), art['features'], art['encodings'], art['scaler'])
        expected = ready.iloc[i][art['features']].astype(float).to_numpy()
        # Price_per_SqFt in the raw data is rounded to 2 decimals, so a rare 0.01 difference is allowed.
        if not np.allclose(row.to_numpy()[0], expected, atol=0.05):
            mismatched += 1
    assert mismatched / N_ROWS < 0.02, f'{mismatched}/{N_ROWS} rows differ from training features'


def test_price_per_sqft_is_in_lakhs():
    art = load_artifacts()
    raw = pd.read_csv('india_housing_prices.csv').iloc[0].to_dict()
    raw['Price_in_Lakhs'], raw['Size_in_SqFt'] = 100.0, 1000
    row = build_input_row(raw, art['features'], art['encodings'], art['scaler'])
    mean = art['scaler'].mean_[art['features'].index('Price_per_SqFt')]
    scale = art['scaler'].scale_[art['features'].index('Price_per_SqFt')]
    assert row['Price_per_SqFt'].iloc[0] == pytest.approx((0.10 - mean) / scale)
