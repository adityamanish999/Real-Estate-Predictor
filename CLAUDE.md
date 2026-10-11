# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Real estate investment advisor for Indian housing data (`india_housing_prices.csv`, about 250k rows). It trains two models: a classifier that labels a property as a "Good Investment" and a regressor that predicts its price in 5 years. A Streamlit app serves both models.

## Commands

All scripts use paths relative to the repo root, so run them from there.

```bash
pip install -r requirements.txt

python src/preprocessing.py   # raw CSV -> data/*.csv + models/{scaler,feature_columns,*_encoding}.pkl
python src/eda.py             # data/clean_data_for_eda.csv -> plots/*.png
python src/train.py           # data/model_ready_data.csv -> models/best_{classifier,regressor}.pkl, models/metrics.json + MLflow runs and registry
streamlit run app.py          # serves the saved models and plots
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiment "Real_Estate_Investment_Advisor"
pytest tests                  # one test: tests/test_inference.py::test_price_per_sqft_is_in_lakhs
```

There is no linter or build step. A virtualenv in `.venv/` is git-ignored.

## Pipeline architecture

The stages run in order and share files on disk. No stage imports another:

1. **`src/preprocessing.py`**
   - `feature_engineering()` creates `Amenity_Score` (a count of comma-separated amenities) and both targets, then writes `data/clean_data_for_eda.csv`.
   - `encode_and_scale()` target-encodes State, City and Locality with the mean `Price_in_Lakhs` and saves each mapping as a dict in `models/<col>_encoding.pkl`. It one-hot encodes the low-cardinality categoricals with `drop_first=True`, fits a `StandardScaler` on 11 numeric columns, and writes `data/model_ready_data.csv` and `models/feature_columns.pkl`.
2. **`src/eda.py`** reads the clean (unencoded) CSV and writes the PNGs that the app's dashboard tab displays.
3. **`src/train.py`** samples 50k rows. For each target it fits 5 models (sklearn and XGBoost) as MLflow runs (`sqlite:///mlflow.db`, artifacts in `mlruns/`). It keeps the best classifier by F1 and the best regressor by R² and saves them with joblib. It also registers them in the MLflow model registry and writes `models/metrics.json`, which the app reads.
4. **`app.py`** has four tabs (prediction form, dashboard, property explorer, model performance). It loads every `models/*.pkl` file through `src/inference.py`, which rebuilds the feature vector for each prediction.

**The targets are synthetic.** They don't come from real outcomes:
- `Good_Investment` = `Price_per_SqFt` < the locality's median `Price_per_SqFt` AND `BHK >= 2`.
- `Future_Price_5Y` = `Price_in_Lakhs * (1 + rate)^5`. The rate is drawn per city from U(0.06, 0.12) with seed 42, plus 0.01 for `Ready_to_Move`.

Because of this, the regressor mostly learns a deterministic formula, and high metrics are expected.

## Keeping train and serve in sync

`src/inference.py` (`build_input_row`) rebuilds the feature vector for the app by hand instead of reusing `preprocessing.py`, so the two must be changed together. `tests/test_inference.py` compares its output with `data/model_ready_data.csv` for 2,000 real rows and fails if they diverge. Things that must match `preprocessing.py`:
- `NUM_COLS`, the columns the scaler was fitted on, in that order.
- `Price_per_SqFt` is in **lakhs per sq ft, rounded to 2 decimals** (the dataset's convention), not rupees. `Age_of_Property` is `2025 - Year_Built`.
- One-hot columns are named `<Col>_<value>`. Names that are not in `feature_columns.pkl` are silently ignored, so a mismatched dropdown value falls back to the baseline category without an error. The app takes its dropdown options from `data/clean_data_for_eda.csv` for this reason.

After retraining, the app picks up the new `.pkl` files only after a restart, because `load_assets` is wrapped in `@st.cache_resource`.

## MLflow quirks

- `mlflow.sklearn.log_model` must use `serialization_format="cloudpickle"`; MLflow 3's default (skops) refuses to save tree models.
- The experiment `Real_Estate_Investment_Advisor_legacy` is the original 10 runs, whose artifact path was recorded on a Windows machine (`C:/Users/...`). Logging to an experiment with such a path creates a stray `C:/` folder in the repo, so new runs go to a fresh experiment.
- Registered models: `RealEstate_Classifier` and `RealEstate_Regressor`, re-registered (new version) on every `train.py` run.

## Committed artifacts

The repo commits the generated data CSVs, `models/*.pkl`, `plots/`, `mlflow.db`, and `mlruns/`. Running the pipeline overwrites them, and each `train.py` run adds 10 more MLflow runs and two registry versions.
