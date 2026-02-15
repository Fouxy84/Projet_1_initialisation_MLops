#api/load_data.py
import json
from flask import json
from os import name
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import lightgbm as lgb
import xgboost as xgb
import mlflow

# metrique cout metier
def business_cost(y_true, y_pred, fn_cost=10, fp_cost=1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * fn_cost + fp * fp_cost

# meilleur seuil
def find_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.05, 0.95, 0.05)
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cost = business_cost(y_true, y_pred)
        costs.append(cost)
    best_idx = np.argmin(costs)
    return thresholds[best_idx], costs

BASE_DIR = Path(r"C:/Users/coach/Desktop/datascientest/OpenClassrooms/Projects_MLops/Projet_1_initialisation_MLops")
DATA_PROC = BASE_DIR / "data" / "proceed"
DATA_PATH = DATA_PROC / "homecredit_features.csv"
MODEL_DIR = BASE_DIR / "models"


df = pd.read_csv(DATA_PATH, low_memory=False)
print(df.shape)

train_df = df[df["TARGET"].notna()]
train_autoML = train_df.drop(columns=["SK_ID_CURR"])
X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
X = X.replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy="median") #remplace les valeurs manquantes par la médiane de chaque colonne
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
y = train_df["TARGET"]
X.columns = (
    X.columns
    .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlflow.set_tracking_uri("file:/content/drive/MyDrive/Colab Notebooks/Projet_1_initialisation_MLops/notebook/mlruns")
mlflow.set_experiment("HomeCredit_Scoring_all_best_models")

with mlflow.start_run(run_name="XGBoost_best_model"):
    model_xgb = xgb.XGBClassifier(learning_rate=0.1, n_estimators=400, max_depth=6, eval_metric="logloss",random_state=42)
    model_xgb.fit(X_train, y_train)
    y_proba = model_xgb.predict_proba(X_test)[:,1]
    best_t, best_cost = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_t).astype(int)
    cost = business_cost(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Log metrics
    mlflow.log_metric("AUC_xgb", auc)
    mlflow.log_metric("Accuracy_xgb", acc)
    mlflow.log_metric("Business_Cost_xgb",cost)
    mlflow.log_metric("Threshold_xgb", best_t)
    # Log parameters
    mlflow.log_params(model_xgb.get_params())
    mlflow.set_tags({"model_type": "XGBoost_best","project": "HomeCredit_Scoring_all_best_models","optimization": "Business_Cost"})
    # Log artifacts
    artefact_dir = Path("artifacts/xgb")
    artefact_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump(imputer, "artifacts/xgb/imputer.joblib")
    joblib.dump(list(X_train.columns), "artifacts/xgb/features.joblib")
    with open("artifacts/xgb/threshold.json", "w") as f:
        json.dump({"best_threshold": float(best_t)}, f)
    
    mlflow.log_artifact("artifacts/xgb/imputer.joblib")
    mlflow.log_artifact("artifacts/xgb/features.joblib")
    mlflow.log_artifact("artifacts/xgb/threshold.json")

    # Register model
    mlflow.sklearn.log_model(model_xgb,name="model_xgb",registered_model_name="HomeCredit_Scoring_final_XGBoost",input_example=X_train.iloc[:5])
    


with mlflow.start_run(run_name="LightGBM_best_model"):
    model_lgb = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=200, num_leaves=31, random_state=42)
    model_lgb.fit(X_train, y_train)
    y_proba = model_lgb.predict_proba(X_test)[:,1]
    best_t, best_cost = find_best_threshold(y_test, y_proba)
    y_pred = (y_proba >= best_t).astype(int)
    cost = business_cost(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Log metrics
    mlflow.log_metric("AUC_lgb", auc)
    mlflow.log_metric("Accuracy_lgb", acc)
    mlflow.log_metric("Business_Cost_lgb",cost)
    mlflow.log_metric("Threshold_lgb", best_t)
    # Log parameters
    mlflow.log_params(model_lgb.get_params())
    mlflow.set_tags({"model_type": "LightGBM_best","project": "HomeCredit_Scoring_all_best_models","optimization": "Business_Cost"})
    # Log artifacts
    artefact_dir = Path("artifacts/lgb")
    artefact_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump(imputer, "artifacts/lgb/imputer.joblib")
    joblib.dump(list(X_train.columns), "artifacts/lgb/features.joblib")
    with open("artifacts/lgb/threshold.json", "w") as f:
        json.dump({"best_threshold": float(best_t)}, f)
    mlflow.log_artifact("artifacts/lgb/imputer.joblib")
    mlflow.log_artifact("artifacts/lgb/features.joblib")
    mlflow.log_artifact("artifacts/lgb/threshold.json")

    # Register model
    mlflow.sklearn.log_model(model_lgb,name="model_lgb",registered_model_name="HomeCredit_Scoring_final_LightGBM",input_example=X_train.iloc[:5])
    
