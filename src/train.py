import pandas as pd
import numpy as np
import os
import json
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix
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

def log_confusion_matrix(cm, name):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f'Confusion Matrix - {name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1], labels=['Not Good', 'Good'])
    ax.set_yticks([0, 1], labels=['Not Good', 'Good'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout()
    mlflow.log_figure(fig, 'confusion_matrix.png')
    plt.close(fig)


def register_best(model_uri, registered_name):
    version = mlflow.register_model(model_uri, registered_name)
    print(f"Registered '{registered_name}' version {version.version}")


def train_classification(X_train, X_test, y_train, y_test):
    print("\n--- Training Classification Models ---")
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision_Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),
        "XGBoost_Classifier": XGBClassifier(n_estimators=50, max_depth=6, eval_metric='logloss', random_state=42)
    }

    best_model = None
    best_f1 = 0
    best_name = ""
    best_uri = None
    results = {}

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
            
            cm = confusion_matrix(y_test, preds, labels=[0, 1])
            print(f"[{name}] Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}")

            mlflow.log_param("model_type", "classification")
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "roc_auc": roc_auc})
            log_confusion_matrix(cm, name)

            if isinstance(model, XGBClassifier):
                info = mlflow.xgboost.log_model(model, name="model")
            else:
                info = mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

            results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
                             "roc_auc": roc_auc, "confusion_matrix": cm.tolist()}

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = name
                best_uri = info.model_uri

    print(f"Best Classification Model: {best_name} with F1: {best_f1:.4f}")
    joblib.dump(best_model, 'models/best_classifier.pkl')
    register_best(best_uri, "RealEstate_Classifier")
    return results, best_name

def train_regression(X_train, X_test, y_train, y_test):
    print("\n--- Training Regression Models ---")
    models = {
        "Linear_Regression": LinearRegression(),
        "Decision_Tree_Reg": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random_Forest_Reg": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42),
        "AdaBoost_Reg": AdaBoostRegressor(n_estimators=50, random_state=42),
        "XGBoost_Regressor": XGBRegressor(n_estimators=50, max_depth=6, random_state=42)
    }

    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    best_uri = None
    results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
                
            print(f"[{name}] MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
            
            mlflow.log_param("model_type", "regression")
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({"mse": mse, "rmse": rmse, "mae": mae, "r2": r2})

            if isinstance(model, XGBRegressor):
                info = mlflow.xgboost.log_model(model, name="model")
            else:
                info = mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

            results[name] = {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name
                best_uri = info.model_uri

    print(f"Best Regression Model: {best_name} with R2: {best_r2:.4f}")
    joblib.dump(best_model, 'models/best_regressor.pkl')
    register_best(best_uri, "RealEstate_Regressor")
    return results, best_name

if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Real_Estate_Investment_Advisor")

    df = load_data('data/model_ready_data.csv')
    
    # Classification Setup
    X_cls = df.drop(columns=['Good_Investment', 'Future_Price_5Y'])
    y_cls = df['Good_Investment']
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    
    cls_results, best_cls = train_classification(Xc_train, Xc_test, yc_train, yc_test)

    # Regression Setup
    X_reg = df.drop(columns=['Good_Investment', 'Future_Price_5Y'])
    y_reg = df['Future_Price_5Y']
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_results, best_reg = train_regression(Xr_train, Xr_test, yr_train, yr_test)

    with open('models/metrics.json', 'w') as f:
        json.dump({"classification": cls_results, "regression": reg_results,
                   "best": {"classifier": best_cls, "regressor": best_reg}}, f, indent=2)
    
    print("\nTraining complete! Best models saved in 'models/' directory.")
