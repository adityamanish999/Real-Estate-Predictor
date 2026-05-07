import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from xgboost import XGBClassifier, XGBRegressor

import warnings
warnings.filterwarnings('ignore')

def load_data(file_path, sample_size=50000):
    print("Loading preprocessed data...")
    df = pd.read_csv(file_path)
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} rows for faster training.")
    return df

def train_classification(X_train, X_test, y_train, y_test):
    print("\n--- Training Classification Models ---")
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Decision_Tree": DecisionTreeClassifier(max_depth=10),
        "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1),
        "AdaBoost": AdaBoostClassifier(n_estimators=50),
        "XGBoost_Classifier": XGBClassifier(n_estimators=50, max_depth=6, eval_metric='logloss', use_label_encoder=False)
    }

    best_model = None
    best_f1 = 0
    best_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            try:
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except:
                roc_auc = 0
                
            print(f"[{name}] Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
            
            mlflow.log_param("model_type", "classification")
            mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "roc_auc": roc_auc})
            
            if isinstance(model, XGBClassifier):
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
                
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = name

    print(f"Best Classification Model: {best_name} with F1: {best_f1:.4f}")
    joblib.dump(best_model, 'models/best_classifier.pkl')

def train_regression(X_train, X_test, y_train, y_test):
    print("\n--- Training Regression Models ---")
    models = {
        "Linear_Regression": LinearRegression(),
        "Decision_Tree_Reg": DecisionTreeRegressor(max_depth=10),
        "Random_Forest_Reg": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1),
        "AdaBoost_Reg": AdaBoostRegressor(n_estimators=50),
        "XGBoost_Regressor": XGBRegressor(n_estimators=50, max_depth=6)
    }

    best_model = None
    best_r2 = -float('inf')
    best_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
                
            print(f"[{name}] RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
            
            mlflow.log_param("model_type", "regression")
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
            
            if isinstance(model, XGBRegressor):
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
                
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name

    print(f"Best Regression Model: {best_name} with R2: {best_r2:.4f}")
    joblib.dump(best_model, 'models/best_regressor.pkl')

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Real_Estate_Investment_Advisor")

    df = load_data('data/model_ready_data.csv')
    
    # Classification Setup
    X_cls = df.drop(columns=['Good_Investment', 'Future_Price_5Y'])
    y_cls = df['Good_Investment']
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    train_classification(Xc_train, Xc_test, yc_train, yc_test)

    # Regression Setup
    X_reg = df.drop(columns=['Good_Investment', 'Future_Price_5Y'])
    y_reg = df['Future_Price_5Y']
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    train_regression(Xr_train, Xr_test, yr_train, yr_test)
    
    print("\nTraining complete! Best models saved in 'models/' directory.")
