# api/load_data.py
import json
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import joblib
import lightgbm as lgb
import xgboost as xgb
import mlflow
import mlflow.sklearn

from utilis import business_cost, find_best_threshold


# ============================================================
# Paths & data
# ============================================================
BASE_DIR = Path(r"C:/Users/coach/Desktop/datascientest/OpenClassrooms/Projects_MLops/Projet_1_initialisation_MLops")
DATA_PATH = BASE_DIR / "data" / "proceed" / "homecredit_features.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
print("Raw shape:", df.shape)

train_df = df[df["TARGET"].notna()]

X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
y = train_df["TARGET"]

# sample pour accélérer
X = X.sample(n=100000, random_state=42)
y = y.loc[X.index]

print("Train shape:", X.shape)

# nettoyage
X.columns = (
    X.columns
    .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
)

X = X.replace([np.inf, -np.inf], np.nan).astype(np.float64)

nan_cols = X.columns[X.isna().all()]
if len(nan_cols) > 0:
    print("Colonnes supprimées:", list(nan_cols))
    X = X.drop(columns=nan_cols)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# imputer
imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

features = list(X_train.columns)


# ============================================================
# MLflow config (UNIQUE SOURCE OF TRUTH)
# ============================================================
#mlflow.set_tracking_uri(f"sqlite:///{(BASE_DIR / 'notebook' / 'mlruns' / 'mlflow.db').as_posix()}")
mlflow.set_tracking_uri("file:./notebook/mlruns")
mlflow.set_experiment("HomeCredit_Scoring_all_best_models")

print("MLflow URI:", mlflow.get_tracking_uri())

# ---- Save random inference pool (JSON serializable) - 100 clients from inference test
with mlflow.start_run(run_name="inference_pool"):
    
    sample_df = X_test.sample(n=100, random_state=452)

    inference_pool = [
        {
        "Client_index": int(idx),
        "features": row.to_dict()
        }
    for idx, row in sample_df.iterrows()
    ]

    with open("inference_pool.json", "w") as f:
        json.dump(inference_pool, f)

    mlflow.log_artifact("inference_pool.json")

# ============================================================
# XGBoost
# ============================================================
with mlflow.start_run(run_name="XGBoost_best_model"):
    model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=400,
        max_depth=6,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(y_test, y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    mlflow.log_metrics({
        "AUC": roc_auc_score(y_test, y_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Business_Cost": business_cost(y_test, y_pred),
        "Threshold": threshold,
    })

    mlflow.log_params(model.get_params())

    # ---- ARTIFACTS (STRUCTURE API-COMPATIBLE)
    mlflow.log_artifact(
        joblib.dump(imputer, "imputer.joblib")[0],
        artifact_path="xgb",
    )
    mlflow.log_artifact(
        joblib.dump(features, "features.joblib")[0],
        artifact_path="xgb",
    )

    with open("threshold.json", "w") as f:
        json.dump({"best_threshold": float(threshold)}, f)
    mlflow.log_artifact("threshold.json", artifact_path="xgb")

    # ---- MODEL REGISTRY
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="HomeCredit_Scoring_final_XGBoost",
        input_example=X_train.iloc[:5],
    )

    # Save model to artifacts
    joblib.dump(model, "artifacts/xgb/model.joblib")


# ============================================================
# LightGBM
# ============================================================
with mlflow.start_run(run_name="LightGBM_best_model"):
    model = lgb.LGBMClassifier(
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=31,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = find_best_threshold(y_test, y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    mlflow.log_metrics({
        "AUC": roc_auc_score(y_test, y_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Business_Cost": business_cost(y_test, y_pred),
        "Threshold": threshold,
    })

    mlflow.log_params(model.get_params())

    mlflow.log_artifact(
        joblib.dump(imputer, "imputer.joblib")[0],
        artifact_path="lgb",
    )
    mlflow.log_artifact(
        joblib.dump(features, "features.joblib")[0],
        artifact_path="lgb",
    )

    with open("threshold.json", "w") as f:
        json.dump({"best_threshold": float(threshold)}, f)
    mlflow.log_artifact("threshold.json", artifact_path="lgb")

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="HomeCredit_Scoring_final_LightGBM",
        input_example=X_train.iloc[:5]
    )

    # Save model to artifacts
    joblib.dump(model, "artifacts/lgb/model.joblib")
