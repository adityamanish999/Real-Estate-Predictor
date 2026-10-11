# Project Guide: Real Estate Investment Advisor

A complete explanation of this project, written so that you can present it and answer your professor's questions. Read sections 1–5 to understand the project, section 6 to understand the code, and sections 8–10 to prepare for the viva.

---

## 1. The project in 30 seconds

> "An investor wants to know two things about a property: **is it a good buy?** and **what will it be worth in 5 years?** I built a machine-learning system that answers both. A *classification* model says Good Investment or not. A *regression* model predicts the price after 5 years. I trained 5 models for each question, tracked all 10 in MLflow, and put the best ones behind a Streamlit web app."

### The problem statement in plain words

| Assignment says | What it means |
|---|---|
| Classify "Good Investment" | Predict a yes/no label for each property. This is **classification**. |
| Predict price after 5 years | Predict a number. This is **regression**. |
| Preprocess, engineer features | Clean the data, convert text to numbers, create useful new columns. |
| Streamlit app | A web page where a user types in property details and sees predictions. |
| MLflow | A tool that records every training experiment (settings, scores, saved models) so they can be compared. |

**Business use:** investors screening listings, buyers looking for under-priced properties, and real estate websites showing data-backed forecasts.

---

## 1b. What was fixed after the first version

If your professor asks how the project evolved, this is the honest answer. A review of the first version found problems, and they were fixed before submission:

| Problem found | What it caused | Fix |
|---|---|---|
| The app computed price per sq ft in **rupees**, but the model was trained on **lakhs** | On 2,000 real properties the app predicted "Good Investment" for **0%**. After the fix it is **36.6%**, close to the true 37.7%. The 5-year forecast was affected much less (median growth multiple 1.54 both ways) | One shared function, `src/inference.py`, plus a test |
| Dropdowns offered values the data doesn't have (`House`, `Penthouse`, `Fully Furnished`…) | The choice was silently ignored and a default category was used | Options are now read from the data |
| No inputs for transport, parking, security, facing, owner type | Those were always set to their default | Inputs added |
| Brief asked for confusion matrix, MSE, hyperparameters, model registry, confidence, feature importance, filters, all 20 EDA questions, missing-value and outlier handling | Missing | All added (see section 8) |
| No tests | The unit bug went unnoticed | `tests/test_inference.py` checks app features against training features |

---

## 2. Tech stack and why each tool

| Tool | What it is | Where it is used here | Why chosen |
|---|---|---|---|
| **Python** | Programming language | Everything | Standard for data science |
| **pandas** | Table (DataFrame) library | Reading the CSV, cleaning, grouping, building features | Handles 250,000 rows easily |
| **NumPy** | Fast number arrays | Math in preprocessing (growth rates, seeds), the app | Pandas and scikit-learn are built on it |
| **scikit-learn** | Machine-learning library | `StandardScaler`, train/test split, 3 classifiers, 3 regressors, all metrics | Simple, reliable, standard |
| **XGBoost** | Gradient-boosted trees | One classifier and one regressor | Usually the strongest model on table data |
| **joblib** | Saves Python objects to files | `.pkl` files in `models/` (models, scaler, encoders) | Recommended way to save scikit-learn objects |
| **matplotlib + seaborn** | Plotting | `src/eda.py` makes the 22 EDA images; confusion-matrix images | Standard for static charts |
| **Plotly** | Interactive charts | Heatmap, city and trend charts in the app | Hover and zoom inside Streamlit |
| **MLflow** | Experiment tracking | `src/train.py` logs parameters, metrics, models; registry | Required by the assignment |
| **SQLite** (the "SQL") | A database stored in one file | `mlflow.db`: MLflow keeps its runs and metrics there | Needs no server. The project's own code writes no SQL. |
| **Streamlit** | Web-app framework for Python | `app.py` | Builds a UI in pure Python, no HTML/JS needed |
| **Pillow (PIL)** | Image loading | Shows the EDA PNGs in the app | Streamlit's image display uses it |
| **pytest** | Test runner | `tests/test_inference.py` | Checks app features equal training features |
| **Git** | Version control | The repo | Standard |

`requirements.txt` lists pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, xgboost, streamlit, mlflow, joblib, pillow, pytest and jupyter.

Do not claim a Jupyter notebook: `jupyter` is in `requirements.txt` but the project has no notebooks.

---

## 3. What each file is

```
Real-Estate-Predictor/
├── india_housing_prices.csv      RAW DATA: 250,000 properties, 23 columns
├── src/
│   ├── preprocessing.py          STEP 1: clean, create features/targets, encode, scale
│   ├── eda.py                    STEP 2: draws the 22 analysis plots
│   ├── train.py                  STEP 3: trains 10 models, logs to MLflow, saves best 2
│   └── inference.py              Used by the app: turns form input into model input
├── app.py                        STEP 4: the Streamlit web app
├── tests/test_inference.py       Checks the app builds the same features as training
├── data/
│   ├── clean_data_for_eda.csv    Cleaned data BEFORE encoding (readable). Used by EDA and the app
│   ├── model_ready_data.csv      Data AFTER encoding and scaling. Used to train
│   └── data_quality_report.csv   Missing values and outlier counts per column
├── models/
│   ├── best_classifier.pkl       Winning "Good Investment" model (Decision Tree)
│   ├── best_regressor.pkl        Winning 5-year-price model (XGBoost)
│   ├── scaler.pkl                The fitted StandardScaler
│   ├── feature_columns.pkl       The 28 feature names, in training order
│   ├── State/City/Locality_encoding.pkl   dicts: place name → average price
│   └── metrics.json              Scores of all 10 models (the app's "Model Performance" tab reads it)
├── plots/                        22 PNG images from eda.py
├── mlflow.db                     MLflow's SQLite database (runs, metrics, registry)
├── mlruns/                       MLflow's saved model files and images
├── requirements.txt              Python packages needed
├── README.md                     Project documentation
└── CLAUDE.md                     Notes for AI coding assistants (can be ignored in the presentation)
```

---

## 4. Order of execution: start to end

```
 india_housing_prices.csv                              (START: raw data)
          │
          ▼
 python src/preprocessing.py
          │  creates: data/clean_data_for_eda.csv, data/model_ready_data.csv,
          │           data/data_quality_report.csv,
          │           models/scaler.pkl, feature_columns.pkl, *_encoding.pkl
          │
          ├────────────────────────────┐
          ▼                            ▼
 python src/eda.py            python src/train.py
   reads clean_data_for_eda     reads model_ready_data.csv
   makes plots/*.png            trains 10 models → MLflow (mlflow.db, mlruns/)
          │                     saves models/best_classifier.pkl, best_regressor.pkl, metrics.json
          │                            │
          └────────────┬───────────────┘
                       ▼
          streamlit run app.py                          (END: browser app)
            loads models + encoders + scaler (via src/inference.py)
            user fills form → features built → classifier + regressor → result cards
```

- **Start point:** `india_housing_prices.csv`, processed by `python src/preprocessing.py`.
- **End point:** `streamlit run app.py`, which the user sees in the browser.
- `eda.py` and `train.py` are independent of each other. Both need only `preprocessing.py` to have run first.
- The steps never import each other. They communicate through **files** on disk.
- Always run commands from the project root, because the paths are relative.
- First-time setup: `python3 -m venv .venv`, activate it, then `pip install -r requirements.txt`.
- To see experiments: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- Optional check: `pytest tests` (confirms the app builds the same features as training).

---

## 5. Key concepts you will be asked about

**Classification vs regression.** Classification predicts a category (Good / Not good). Regression predicts a number (price in lakhs).

**Feature vs target.** Features are inputs (BHK, size, city…). The target is what we predict (`Good_Investment`, `Future_Price_5Y`).

**How the targets are created.** The dataset has no "good investment" or "future price" column, so the assignment says to define them by rules:
- `Good_Investment = 1` if the property's **price per sq ft is below its locality's median** *and* it has **at least 2 BHK**. (The brief's "Price per Sq Ft: cheaper than median → good" plus a "multi-factor" condition.) 37.7% of properties are labelled good.
- `Future_Price_5Y = Price × (1 + r)^5`. Each city gets a random but fixed growth rate `r` between 6% and 12% (seed 42, so it is reproducible). Ready-to-move properties get +1%. (The brief's "by location" growth-rate approach.)

**Why the scores are so high (important).** Because the targets are calculated from the input columns, the models are mostly *rediscovering the formula*. The 99% accuracy and 0.9997 R² reflect that, not real-world predictive power. Say this yourself before the professor does (see Q&A).

---

## 6. What every function does

### `src/preprocessing.py` (run first)

**`load_data(file_path)`**: reads the CSV into a pandas DataFrame.

**`data_quality_report(df)`**: the "handle missing values and outliers" step.
- Counts missing values per column. If any exist it fills numeric columns with the **median** and text columns with the **mode** (most common value). This dataset has 0 missing values, so nothing changes.
- Finds outliers in price, size and price per sq ft with the **IQR rule** (outside Q1 − 1.5·IQR to Q3 + 1.5·IQR). It only **reports** them (20,020 for price per sq ft: it is price ÷ size, so it has a long right tail up to 0.99 while most values are below 0.33) and does not delete rows.
- Saves the counts to `data/data_quality_report.csv`.

**`feature_engineering(df)`**: creates new columns.
1. Drops duplicate rows (there are none).
2. Runs the data-quality step.
3. `Amenity_Score` = number of comma-separated amenities (the brief's "amenity density score").
4. `Good_Investment`: the rule above, using `groupby('Locality')['Price_per_SqFt'].transform('median')`.
5. `Growth_Rate` per city and `Future_Price_5Y` = price × (1 + rate)^5.
6. Saves `data/clean_data_for_eda.csv` (human-readable, before encoding).

**`encode_and_scale(df)`**: turns the table into numbers for the models.
1. Drops `ID`, `Amenities` (text), `Growth_Rate` (it would leak the answer).
2. **Mean encoding** of State, City, Locality: each place is replaced by the average price there. It is used because Locality has 500 values; one-hot would add 500 columns. The mapping dicts are saved so the app can reuse them.
3. **One-hot encoding** of the 8 small categoricals (property type, furnishing, transport, parking, security, facing, owner type, availability), with `drop_first=True` so one category per column is the implicit baseline. This avoids redundant columns.
4. **StandardScaler** on 11 numeric columns (subtract mean, divide by standard deviation). Saved as `scaler.pkl`.
5. Saves `feature_columns.pkl` (28 names) and `model_ready_data.csv`.

### `src/eda.py`

**`save(name)`**: saves the current matplotlib figure to `plots/<name>.png` and closes it.

**`perform_eda(file_path)`**: draws one chart per assignment question:
- Q1–5: price distribution, size distribution, price per sq ft by property type, size vs price, outlier boxplots.
- Q6–10: price per sq ft by state, price by city, median age by locality, BHK across cities (heatmap), top-5 localities and their price trend by year built.
- Q11–15: correlation heatmap, schools vs price per sq ft, hospitals vs price per sq ft, price by furnishing, price per sq ft by facing.
- Q16–20: owner-type counts, availability counts, parking vs price, amenities vs price per sq ft, public transport vs price per sq ft and share of good investments.
- Plus price vs age.

The brief's "crime rate" and "infrastructure score" analyses are not possible: the dataset has no such columns. Say so.

### `src/train.py`

**`load_data(file_path, sample_size=50000)`**: reads `model_ready_data.csv` and randomly samples 50,000 rows (`random_state=42`) so training takes minutes, not hours.

**`log_confusion_matrix(cm, name)`**: draws the 2×2 confusion matrix and attaches the image to the current MLflow run.

**`register_best(model_uri, registered_name)`**: puts a saved model into the **MLflow Model Registry**.

**`train_classification(...)`**: for each of 5 models (Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost):
1. Starts an MLflow run (`mlflow.start_run`).
2. Fits the model and predicts on the test set.
3. Computes accuracy, precision, recall, F1, ROC AUC and the confusion matrix.
4. Logs the model type, all hyperparameters (`get_params()`), the metrics, the confusion-matrix image, and the model itself.
5. Keeps track of the best by **F1**. At the end saves it to `models/best_classifier.pkl` and registers it as `RealEstate_Classifier`.

**`train_regression(...)`**: same pattern for Linear Regression, Decision Tree, Random Forest, AdaBoost and XGBoost, scoring MSE, RMSE, MAE and R². Best by **R²** → `models/best_regressor.pkl`, registered as `RealEstate_Regressor`.

**Main block**: points MLflow at `sqlite:///mlflow.db`, sets the experiment `Real_Estate_Investment_Advisor`, makes the **80/20 train/test split** (`random_state=42`), runs both trainers, writes `models/metrics.json`.

### `src/inference.py`

**`load_artifacts()`**: loads both models, the scaler, the feature list and the three encoding dicts.

**`build_input_row(raw, features, encodings, scaler)`**: converts one property's form values into exactly what the models were trained on:
1. Computes price per sq ft (**in lakhs, rounded to 2 decimals**, as in the dataset), age (`2025 − Year_Built`) and the amenity count.
2. Looks up State/City/Locality in the encoding dicts.
3. Sets the one-hot column for each chosen category.
4. Fills all missing feature columns with 0, orders the columns like training, and applies the saved scaler.

### `app.py` (the Streamlit app)

- **Page setup and CSS**: dark theme and card styling.
- **`load_assets()`**: loads models once (`@st.cache_resource` keeps them in memory between interactions).
- **`load_options()`**: dropdown options are read from the data, so they always match what the models know. City choices depend on the chosen State.
- **`top_importances(model, names)`**: returns the 10 most important features of a model.
- **Tab 1, Predictive Analytics:** the form → `build_input_row` → `clf.predict` (good/not good) + `clf.predict_proba` (confidence) + `reg.predict` (future price) → two result cards, plus a feature-importance expander.
- **`load_market_data()`**: loads the clean CSV once (`@st.cache_data`).
- **Tab 2, Market Dashboard:** Plotly heatmap (average price by state × BHK), city price bars, price trend by year built, and a gallery of all EDA plots.
- **Tab 3, Property Explorer:** filters (state, city, price, BHK, type, "only good investments") over the 250,000 properties, with live counts and a table.
- **Tab 4, Model Performance:** reads `metrics.json` and shows all model scores and the best classifier's confusion matrix.

### `tests/test_inference.py`

- **`test_app_features_match_training_features`**: for 2,000 real rows, the features built by the app code must equal the ones `preprocessing.py` produced.
- **`test_price_per_sqft_is_in_lakhs`**: guards against a unit bug that existed earlier (the app used rupees while training used lakhs).

### MLflow notes (technical details)

- Scikit-learn models are saved with `serialization_format="cloudpickle"`, because MLflow 3's default format refuses to save tree models.
- MLflow records each experiment's artifact folder as an **absolute path** on the machine that trained it. On another computer the MLflow UI may not find the model files. To fix that, delete `mlflow.db` and `mlruns/`, then re-run `python src/train.py` (about 7 minutes). The app itself is not affected, because it uses `models/*.pkl`.
- The original 10 runs are kept in the experiment `Real_Estate_Investment_Advisor_legacy` (see Q29).

---

## 7. Results (from the final training run)

**Classification** (10,000 test properties):

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8871 | 0.8356 | 0.8622 | 0.8487 | 0.9649 |
| **Decision Tree** ← best | 0.9919 | 0.9948 | 0.9831 | 0.9889 | 0.9964 |
| Random Forest | 0.9897 | 0.9978 | 0.9741 | 0.9858 | 0.9995 |
| AdaBoost | 0.9910 | 0.9961 | 0.9793 | 0.9876 | 0.9997 |
| XGBoost | 0.9907 | 0.9885 | 0.9861 | 0.9873 | 0.9997 |

Decision Tree confusion matrix `[[TN, FP], [FN, TP]]` = `[[6309, 19], [62, 3610]]`.
Read it as: 6,309 non-good properties correctly rejected, 3,610 good ones correctly found, 19 wrongly called good, 62 good ones missed.

**Regression** (in lakhs):

| Model | MSE | RMSE | MAE | R² |
|---|---|---|---|---|
| Linear Regression | 1237.10 | 35.17 | 27.19 | 0.9748 |
| Decision Tree | 535.11 | 23.13 | 16.57 | 0.9891 |
| Random Forest | 410.48 | 20.26 | 14.84 | 0.9916 |
| AdaBoost | 1494.90 | 38.66 | 31.82 | 0.9695 |
| **XGBoost** ← best | 14.80 | 3.85 | 2.95 | 0.9997 |

**Dataset facts:** 250,000 rows; 23 columns; 20 states, 42 cities, 500 localities; no missing values; no duplicates; price 10–500 lakhs; size 500–5,000 sq ft; built 1990–2023.

---

## 8. Assignment checklist (be ready to show each item)

| Requirement | Status | Where |
|---|---|---|
| Handle missing values and duplicates | Done. The data has none, and the code handles them anyway | `preprocessing.py` |
| Scale numeric features | Done | `StandardScaler` |
| Encode categorical features | Done | mean encoding + one-hot |
| New features (price per sq ft, amenity score) | Done | `feature_engineering` |
| "Good Investment" label | Done, rule-based | `feature_engineering` |
| Outlier detection | Done (IQR, reported, not removed) | `data_quality_report` |
| EDA, 20 questions | Done, all 20 (crime/infrastructure are impossible: no such columns) | `eda.py`, `plots/` |
| ≥5 classifiers, ≥5 regressors | Done, 5 + 5 | `train.py` |
| Accuracy, precision, recall, ROC AUC, F1, **confusion matrix** | Done | `train.py` |
| RMSE, MAE, R², **MSE** | Done | `train.py` |
| MLflow: params, metrics, artifacts | Done | `mlflow.db`, `mlruns/` |
| MLflow **model registry** | Done: `RealEstate_Classifier`, `RealEstate_Regressor` | `register_best` |
| Streamlit input form | Done | tab 1 |
| Filter by area, price, BHK | Done, in the Property Explorer tab | tab 3 |
| "Good Investment?" and "price after 5 years" | Done | tab 1 |
| Visual insights (heatmap, trends) | Done | tab 2 |
| Confidence score and feature importance | Done | tab 1 |
| Documentation | Done | `README.md` |

**Honest limits** (say them before you are asked):
1. The labels are formulas, so the near-perfect scores say little about real-world accuracy (Q&A #6).
2. The "5-year trend" uses *year built* as a stand-in for time. The dataset has no price history.
3. The app runs locally; it is not hosted on a public server.
4. Only the best model of each kind is registered, with no staging or alias workflow.
5. The classifier's confidence is often 100%, because it is a single decision tree whose leaves are pure.
6. Only the classifier gives a confidence score; the regressor gives a single number, with no interval.
7. The encoders and scaler are fitted on all data before the train/test split. With 250k rows the effect is tiny, but a stricter pipeline would fit them on the training part only.
8. The app was tested automatically (Streamlit's `AppTest`) but not by clicking through every page in a browser, so check the visuals once before presenting.
9. The data is synthetic. It shows almost no relationships (e.g. correlation between size and price is −0.003), which is why the EDA bars are flat.

---

## 9. Live demo script

1. `streamlit run app.py` (from the project root).
2. **Predictive Analytics:** choose a State and see the City list change. Try **price 30 L, size 3000 sq ft** → *Good Investment*. Then **price 450 L, size 600 sq ft** → *Not a Good Investment*. Point out the confidence and the 5-year forecast (about 1.4× today's price) and open "feature importance". Explain that price per sq ft and BHK drive the classifier, and current price drives the forecast.
3. **Market Dashboard:** hover over the heatmap, change the trend's state, open the EDA gallery.
4. **Property Explorer:** filter to 3 BHK in one state, tick "only Good Investment".
5. **Model Performance:** show the tables and the confusion matrix.
6. In a second terminal: `mlflow ui --backend-store-uri sqlite:///mlflow.db`, open `http://localhost:5000`. Show the experiment with 10 runs, click one for parameters, metrics and the confusion-matrix image, then the **Models** page for the registry.

---

## 10. How to answer questions

### Problem and data

**1. What problem does it solve?** It helps an investor decide whether a property is a good buy and what it will be worth in 5 years, using data instead of guesses.

**2. Why two models?** Different questions: one gives a yes/no (classification), the other a number (regression).

**3. Tell me about the dataset.** 250,000 properties across 20 states and 42 cities, with price, size, BHK, location, age, amenities, nearby schools and hospitals, and so on. No missing values or duplicates.

**4. Where do the targets come from?** The dataset has neither, so I defined them using the rules from the brief: good = priced below the locality median per sq ft with at least 2 BHK; future price = price grown at a city-specific 6–12% per year for 5 years.

**5. Why is the growth rate random?** There is no real price history in the data. Each city gets a fixed, reproducible rate (seed 42) to imitate "different growth per location", one of the options in the brief. It is a modelling assumption, not a measured fact.

**6. Your accuracy is 99%. Isn't that suspicious?** Yes, and I know why. The labels are computed from columns that are also model inputs (price per sq ft, BHK, price, city). The models learn the rule, which shows the pipeline works but doesn't prove real-world predictive power. With real appreciation history, we'd expect lower and more honest scores. Logistic Regression scored lower (88.7%) because the rule compares against a locality median and that isn't a straight line, while trees handle it naturally.

### Preprocessing

**7. How did you handle missing values and outliers?** A check counts missing values and fills them (median for numbers, mode for categories). This dataset has none. Outliers are found with the IQR rule and reported but not removed, because removing them would change the data the models learn from, and the data is evenly spread.

**8. Why mean (target-style) encoding for State, City and Locality?** Locality has 500 values; one-hot would add 500 columns. Replacing each place with its average price gives one useful number per place.

**9. Why `drop_first=True`?** With a column per category, the last is predictable from the others. Dropping one avoids redundant columns and a problem called the dummy-variable trap.

**10. Why scale features?** Features are on very different ranges (BHK 1–5, size up to 5,000). Scaling helps models like Logistic Regression and Linear Regression. Tree models don't need it, but one scaler for all is simpler. The scaler is saved so the app applies identical scaling.

**11. Any data leakage?** Yes, two forms. (a) Labels derive from input columns, as in Q6. (b) Encoders and the scaler are fitted before the train/test split. On 250k rows the effect is negligible, but a stricter pipeline would fit them on the training split only.

**12. Why did you drop `Growth_Rate`?** It determines the future price directly; keeping it would hand the regressor the answer.

### EDA

**13. What did the EDA show?** Prices are spread evenly across 10–500 lakhs. Average price per sq ft is almost identical across states (0.129–0.133 lakhs). Parking, furnishing, facing, transport, schools, hospitals and amenities show essentially no effect on price. That indicates synthetic data.

**14. What about crime rate and infrastructure score?** The dataset doesn't contain them, so those analyses can't be done. I covered all 20 listed questions.

**15. What do "price trends" mean here?** There is no time series. I plot average price by *year built* as a proxy.

### Models and metrics

**16. Which models did you use?** Classification: Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost. Regression: Linear Regression, Decision Tree, Random Forest, AdaBoost, XGBoost.

**17. How did you pick the best?** Classification by **F1**, regression by **R²**, on the same held-out 20% test set.

**18. Why F1 and not accuracy?** About 38% of properties are good, so classes are unequal. F1 balances precision (how many "good" calls were right) and recall (how many good ones were found).

**19. Explain precision, recall, confusion matrix.** Precision = true "good" calls ÷ all "good" calls. Recall = true "good" calls ÷ all really-good properties. The confusion matrix shows the four outcomes. For the Decision Tree, 19 false positives and 62 false negatives out of 10,000.

**20. Explain ROC AUC.** The probability that the model ranks a random good property above a random not-good one. 0.5 is random guessing, 1.0 is perfect.

**21. Explain RMSE, MAE, MSE, R².** MAE = average error in lakhs. MSE = average squared error; RMSE = its square root, in lakhs, and punishes big errors more. R² = share of price variation the model explains (1.0 is perfect). XGBoost: RMSE 3.85 lakhs, R² 0.9997.

**22. Why does XGBoost win regression but a Decision Tree win classification?** Both do very well. The classification scores are within 0.3% of each other, and the plain Decision Tree edges out by F1 by a hair. For regression, boosting corrects errors step by step and captures the multiplication (price × growth) far better than the others.

**23. Is it overfitted?** I evaluated on a held-out 20% the models never saw, so the scores are honest *for these labels*. The high scores come from the label design, not memorisation.

**24. Why only 50,000 rows?** Speed. Training 10 models on 250,000 rows takes much longer, and 50,000 is plenty for this data.

**25. What is `random_state=42`?** A fixed seed so splits and models are reproducible.

### MLflow

**26. What does MLflow do here?** Each of the 10 models is a *run*. It records hyperparameters, metrics, the confusion-matrix image and the model file, so models can be compared side by side. The two winners are registered in the **Model Registry**.

**27. Where is the data stored?** In `mlflow.db`, a SQLite database, plus files in `mlruns/`. View it with `mlflow ui`.

**28. What is a registered model?** A named, versioned model (`RealEstate_Classifier`, `RealEstate_Regressor`) that other applications could load by name.

**29. What is `Real_Estate_Investment_Advisor_legacy`?** The 10 runs from an earlier training session, kept for history. The current runs are in `Real_Estate_Investment_Advisor`.

### Streamlit and code

**30. How does the app make a prediction?** The form values go to `build_input_row`, which computes price per sq ft, age and amenity count, looks up place encodings, sets one-hot columns, orders them like training and scales them. The classifier and regressor then predict.

**31. Why save the scaler and encodings?** The app must transform new input *exactly* as training did.

**32. Why `@st.cache_resource` / `@st.cache_data`?** Streamlit re-runs the whole script on every click. Caching loads the models and the 250k-row table once.

**33. What is the confidence score?** The classifier's predicted probability for its chosen class (`predict_proba`). It is often 100% because a decision tree's leaves are very pure on this rule-based label.

**34. What does feature importance show?** Which inputs the model relies on most. For the classifier it is price per sq ft (66%) and BHK (32%), exactly the two inputs of the labelling rule. For the regressor it is current price (96%), because the future price is the current price times a growth factor.

**35. What is the test for?** It proves the app creates the same features as training, which catches mistakes like using rupees instead of lakhs. That mistake did exist earlier and made the app predict "not good" for everything.

**36. What bugs did you find and fix?** See section 1b: the rupees-versus-lakhs mismatch (the app called every property "not good"), dropdown values that didn't match the data, and missing inputs. A test now prevents the unit bug from coming back.

**37. What would you improve?** Use real price-history data for the targets; fit encoders and scaler on training data only; add forecast intervals; cross-validation and hyperparameter tuning; deploy to Streamlit Community Cloud or Docker; add a model-stage workflow in the registry.

---

## 11. Glossary

- **Lakh**: 100,000 rupees. Prices are in lakhs.
- **BHK**: bedrooms, hall, kitchen.
- **DataFrame**: a pandas table.
- **Feature / target**: input column / what we predict.
- **Train/test split**: 80% of data to learn from, 20% to test on.
- **Overfitting**: memorising training data and doing badly on new data.
- **Data leakage**: information from the answer sneaking into the inputs.
- **One-hot encoding**: one 0/1 column per category.
- **Mean (target) encoding**: replace a category with the average target value of that category.
- **StandardScaler**: rescale a column to mean 0, standard deviation 1.
- **IQR**: the spread of the middle 50% of values; used to find outliers.
- **Decision Tree / Random Forest**: a flowchart of yes/no questions / many such trees voting.
- **Boosting (AdaBoost, XGBoost)**: many small models, each fixing the previous one's mistakes.
- **Precision / Recall / F1**: accuracy of positive calls / share of positives found / their balance.
- **ROC AUC**: ranking quality of a classifier, 0.5 to 1.0.
- **RMSE / MAE / MSE / R²**: regression error measures.
- **Confusion matrix**: table of correct and wrong predictions by class.
- **MLflow run / experiment / registry**: one training record / a group of runs / named, versioned models.
- **Cache (Streamlit)**: remember a result so it isn't recomputed on every click.
- **Pickle (.pkl)**: a saved Python object.
