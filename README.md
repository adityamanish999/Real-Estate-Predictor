# Real Estate Investment Advisor

Predicts whether an Indian property is a **good investment** (classification) and what it will be **worth in 5 years** (regression). The models are tracked with MLflow and served through a Streamlit app.

## Quick start

```bash
pip install -r requirements.txt

python src/preprocessing.py   # clean, engineer features, encode, scale
python src/eda.py             # generate the EDA plots in plots/
python src/train.py           # train 10 models, log to MLflow, save the best of each
streamlit run app.py          # open the app

mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse experiments and the model registry
pytest tests                  # check the app builds the same features as training
```

Run everything from the repository root. The committed `data/`, `models/` and `plots/` folders already contain the output of these steps, so the app starts without re-running anything.

## Methodology

**Data.** `india_housing_prices.csv`: 250,000 properties, 23 columns, 20 states, 42 cities, 500 localities. It has no missing values and no duplicates.

**Preprocessing** (`src/preprocessing.py`)
- A data-quality step fills missing values (median for numbers, mode for categories) and flags IQR outliers without removing them. The result is saved to `data/data_quality_report.csv`. For this dataset it finds 0 missing values and 20,020 `Price_per_SqFt` outliers.
- `Amenity_Score` is the number of amenities listed for a property.
- State, City and Locality are mean-encoded with the average price. The other categoricals are one-hot encoded (`drop_first=True`).
- 11 numeric features are standardised with `StandardScaler`.

**Targets.** Both are rule-based, as the assignment suggests:
- `Good_Investment = 1` when the property's price per sq ft is below its locality median **and** it has at least 2 BHK. 37.7% of properties qualify.
- `Future_Price_5Y = Price × (1 + r)^5`, where `r` is a growth rate between 6% and 12% per city (seeded with 42), plus 1% for ready-to-move properties.

**Models** (`src/train.py`). Five classifiers (Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost) and five regressors (Linear Regression, Decision Tree, Random Forest, AdaBoost, XGBoost). They train on a 50,000-row sample with an 80/20 split. The best classifier (by F1) and the best regressor (by R²) are saved to `models/` and registered in the MLflow model registry as `RealEstate_Classifier` and `RealEstate_Regressor`.

**MLflow.** Each model is one run in the experiment `Real_Estate_Investment_Advisor` and logs its hyperparameters, metrics, the model itself, and (for classifiers) a confusion-matrix image. The 10 runs of an earlier, pre-fix training session are kept in the experiment `Real_Estate_Investment_Advisor_legacy`.

## Results

Classification (test set of 10,000 properties):

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8871 | 0.8356 | 0.8622 | 0.8487 | 0.9649 |
| **Decision Tree** | **0.9919** | 0.9948 | 0.9831 | **0.9889** | 0.9964 |
| Random Forest | 0.9897 | 0.9978 | 0.9741 | 0.9858 | 0.9995 |
| AdaBoost | 0.9910 | 0.9961 | 0.9793 | 0.9876 | 0.9997 |
| XGBoost | 0.9907 | 0.9885 | 0.9861 | 0.9873 | 0.9997 |

Best classifier confusion matrix (Decision Tree): 6,309 true negatives, 19 false positives, 62 false negatives, 3,610 true positives.

Regression (values in lakhs):

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Linear Regression | 1237.10 | 35.17 | 27.19 | 0.9748 |
| Decision Tree | 535.11 | 23.13 | 16.57 | 0.9891 |
| Random Forest | 410.48 | 20.26 | 14.84 | 0.9916 |
| AdaBoost | 1494.90 | 38.66 | 31.82 | 0.9695 |
| **XGBoost** | **14.80** | **3.85** | **2.95** | **0.9997** |

## App

`streamlit run app.py` has four tabs:
- **Predictive Analytics:** a property form. It returns the investment verdict with a model confidence score, the 5-year price forecast with projected ROI, and the features that drive each model.
- **Market Dashboard:** an interactive state × BHK price heatmap, a city price chart, a price trend by year built, and a gallery of the 20 EDA plots.
- **Property Explorer:** filter the 250,000 properties by state, city, price, BHK and property type.
- **Model Performance:** every model's metrics and the best classifier's confusion matrix.

## Findings

- Prices run from 10 to 500 lakhs, are spread evenly across that range, and average about 254 lakhs. Size runs from 500 to 5,000 sq ft.
- Average price per sq ft is nearly the same across states (0.129 to 0.133 lakhs). Average price by city ranges from about 251 to 258 lakhs.
- Price barely depends on any other feature. Its correlation with size is −0.003, and parking, furnishing, facing, public transport, nearby schools and hospitals, and amenities all show flat bars in the EDA plots.
- Owner types and availability statuses are split almost evenly (about a third each, and about half each).

These flat patterns mean the dataset is synthetic. Real markets show strong location and size effects.

## Use cases

- Investors screening listings for long-term return.
- Buyers looking for properties priced below their locality.
- Real estate platforms adding automated investment analysis and price forecasts to listings.

## Limitations

- **The scores are very high because the labels are formulas.** `Good_Investment` is computed from price per sq ft and BHK, and `Future_Price_5Y` from price and city, so the models largely rediscover those formulas. This reflects how the targets were defined, not how well the models would predict real market outcomes. The confidence score is often 100% for the same reason, because the classifier is a single decision tree.
- Encoders and the scaler are fitted on all rows before the train/test split. With 250,000 rows the effect is small.
- The EDA cannot cover crime rate or infrastructure score, because the dataset has no such columns.
- The data is synthetic: every locality appears in all 42 cities, and about 47% of properties have a floor number above the building's total floors.
- MLflow stores the artifact location of experiments as an absolute path. If the project moves to another machine, create a fresh `mlflow.db` by deleting it and re-running `src/train.py`.

## Project layout

```
india_housing_prices.csv     raw data
src/preprocessing.py         cleaning, features, encoding, scaling
src/eda.py                   EDA plots
src/train.py                 model training, MLflow tracking and registry
src/inference.py             builds model input from form values (shared by the app and tests)
app.py                       Streamlit app
tests/                       checks the app's features match training features
data/  models/  plots/       generated data, saved models and encoders, EDA images
mlflow.db  mlruns/           MLflow tracking database and artifacts
```
