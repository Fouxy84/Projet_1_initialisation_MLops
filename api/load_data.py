# # api/load_data.py
# import json
# import pandas as pd
# import numpy as np
# from pathlib import Path

# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split

# import joblib
# import lightgbm as lgb
# import xgboost as xgb
# import mlflow
# import mlflow.sklearn

# import onnxmltools
# import onnx
# #from skl2onnx.common.data_types import FloatTensorType
# from onnxmltools.convert.common.data_types import FloatTensorType
# from utilis import business_cost, find_best_threshold


# # ============================================================
# # Paths & data
# # ============================================================
# BASE_DIR = Path(r"C:/Users/coach/Desktop/datascientest/OpenClassrooms/Projects_MLops/PROJET_2/Projet_1_initialisation_MLops")
# DATA_PATH = BASE_DIR / "data" / "proceed" / "homecredit_features.csv"

# df = pd.read_csv(DATA_PATH, low_memory=False)
# print("Raw shape:", df.shape)

# train_df = df[df["TARGET"].notna()]

# X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
# y = train_df["TARGET"]

# # sample pour accélérer
# # = X.sample(n=200000, random_state=4)
# #y = y.loc[X.index]

# print("Train shape:", X.shape)

# # nettoyage
# X.columns = (
#     X.columns
#     .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
#     .str.replace(r"_+", "_", regex=True)
# )

# X = X.replace([np.inf, -np.inf], np.nan).astype(np.float64)

# nan_cols = X.columns[X.isna().all()]
# if len(nan_cols) > 0:
#     print("Colonnes supprimées:", list(nan_cols))
#     X = X.drop(columns=nan_cols)

# # split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # imputer
# imputer = SimpleImputer(strategy="median")
# X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
# X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# features = list(X_train.columns)
# onnx_feature_names = [f"f{i}" for i in range(len(features))]
# X_train_onnx = X_train.copy()
# X_train_onnx.columns = onnx_feature_names
# X_test_onnx = X_test.copy()
# X_test_onnx.columns = onnx_feature_names    
# # ============================================================
# # MLflow config (UNIQUE SOURCE OF TRUTH)
# # ============================================================
# #mlflow.set_tracking_uri(f"sqlite:///{(BASE_DIR / 'notebook' / 'mlruns' / 'mlflow.db').as_posix()}")
# mlflow.set_tracking_uri("file:./notebook/mlruns")
# mlflow.set_experiment("HomeCredit_Scoring_all_best_models")

# print("MLflow URI:", mlflow.get_tracking_uri())

# # ---- Save random inference pool (JSON serializable) - 100 clients from inference test
# with mlflow.start_run(run_name="inference_pool"):
    
#     sample_df = X_test.sample(n=1000, random_state=452)

#     inference_pool = [
#         {
#         "Client_index": int(idx),
#         "features": row.to_dict()
#         }
#     for idx, row in sample_df.iterrows()
#     ]

#     with open("inference_pool.json", "w") as f:
#         json.dump(inference_pool, f)

#     mlflow.log_artifact("inference_pool.json")

# # ============================================================
# # XGBoost
# # ============================================================
# with mlflow.start_run(run_name="XGBoost_best_model"):
#     model = xgb.XGBClassifier(
#         learning_rate=0.1,
#         n_estimators=400,
#         max_depth=6,
#         eval_metric="logloss",
#         random_state=42,
#     )

#     #model.fit(X_train, y_train)
#     model.fit(X_train_onnx, y_train)

#     initial_type = [("input", FloatTensorType([None, len(features)]))]
#     onnx_model = onnxmltools.convert_xgboost(model,initial_types=initial_type)
#     onnx.save_model(onnx_model, "xgb_model.onnx")
#     mlflow.log_artifact("xgb_model.onnx", artifact_path="xgb")

#     y_proba = onnx_model.predict_proba(X_test_onnx)[:, 1]

#     threshold = find_best_threshold(y_test, y_proba)

#     y_pred = (y_proba >= threshold).astype(int)

#     mlflow.log_metrics({
#         "AUC": roc_auc_score(y_test, y_proba),
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Business_Cost": business_cost(y_test, y_pred),
#         "Threshold": threshold,
#     })

#     mlflow.log_params(onnx_model.get_params())

#     # ---- ARTIFACTS (STRUCTURE API-COMPATIBLE)
#     mlflow.log_artifact(
#         joblib.dump(imputer, "imputer.joblib")[0],
#         artifact_path="xgb",
#     )
#     mlflow.log_artifact(
#         joblib.dump(features, "features.joblib")[0],
#         artifact_path="xgb",
#     )

#     with open("threshold.json", "w") as f:
#         json.dump({"best_threshold": float(threshold)}, f)
#     mlflow.log_artifact("threshold.json", artifact_path="xgb")

#     # ---- MODEL REGISTRY
#     mlflow.sklearn.log_model(
#         onnx_model,
#         artifact_path="model",
#         registered_model_name="HomeCredit_Scoring_final_XGBoost",
#         input_example=X_train_onnx.iloc[:5],
#     )

#     # Save model to artifacts
#     joblib.dump(onnx_model, "artifacts/xgb/model.joblib")


# # ============================================================
# # LightGBM
# # ============================================================
# with mlflow.start_run(run_name="LightGBM_best_model"):
#     model = lgb.LGBMClassifier(
#         learning_rate=0.05,
#         n_estimators=200,
#         num_leaves=31,
#         random_state=42,
#     )

#     model.fit(X_train_onnx, y_train)
   
#     initial_type = [("input", FloatTensorType([None, len(features)]))]
#     onnx_model = onnxmltools.convert_lightgbm(model,initial_types=initial_type)
#     onnx.save_model(onnx_model, "lgb_model.onnx")
#     mlflow.log_artifact("lgb_model.onnx", artifact_path="lgb")

#     y_proba = onnx_model.predict_proba(X_test_onnx)[:, 1]
#     threshold = find_best_threshold(y_test, y_proba)

#     y_pred = (y_proba >= threshold).astype(int)

#     mlflow.log_metrics({
#         "AUC": roc_auc_score(y_test, y_proba),
#         "Accuracy": accuracy_score(y_test, y_pred),
#         "Business_Cost": business_cost(y_test, y_pred),
#         "Threshold": threshold,
#     })

#     mlflow.log_params(onnx_model.get_params())

#     mlflow.log_artifact(
#         joblib.dump(imputer, "imputer.joblib")[0],
#         artifact_path="lgb",
#     )
#     mlflow.log_artifact(
#         joblib.dump(features, "features.joblib")[0],
#         artifact_path="lgb",
#     )

#     with open("threshold.json", "w") as f:
#         json.dump({"best_threshold": float(threshold)}, f)
#     mlflow.log_artifact("threshold.json", artifact_path="lgb")

#     mlflow.sklearn.log_model(
#         onnx_model,
#         artifact_path="model",
#         registered_model_name="HomeCredit_Scoring_final_LightGBM",
#         input_example=X_train_onnx.iloc[:5]
#     )

#     # Save model to artifacts
#     joblib.dump(onnx_model, "artifacts/lgb/model.joblib")


import json
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import joblib
import xgboost as xgb
import lightgbm as lgb

import mlflow
import mlflow.sklearn
import mlflow.onnx

import onnxmltools
import onnx
from onnxmltools.convert.common.data_types import FloatTensorType

from utilis import business_cost, find_best_threshold


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(".")

DATA_PATH = BASE_DIR / "data/proceed/homecredit_features.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH, low_memory=False)
print("Raw shape:", df.shape)

train_df = df[df["TARGET"].notna()]

X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
y = train_df["TARGET"]

print("Train shape:", X.shape)


# ============================================================
# CLEANING
# ============================================================
X.columns = (
    X.columns
    .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
)

X = X.replace([np.inf, -np.inf], np.nan).astype(np.float32)

# drop full NaN cols
nan_cols = X.columns[X.isna().all()]
if len(nan_cols) > 0:
    print("Dropped columns:", list(nan_cols))
    X = X.drop(columns=nan_cols)


# ============================================================
# SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ============================================================
# IMPUTER
# ============================================================
imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

features = list(X_train.columns)
onnx_feature_names = [f"f{i}" for i in range(len(features))]

X_train_onnx = X_train.copy()
X_train_onnx.columns = onnx_feature_names

X_test_onnx = X_test.copy()
X_test_onnx.columns = onnx_feature_names

# ============================================================
# MLflow CONFIG
# ============================================================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("HomeCredit_Scoring")

print("MLflow URI:", mlflow.get_tracking_uri())


# ============================================================
# SAVE INFERENCE POOL
# ============================================================
with mlflow.start_run(run_name="inference_pool"):

    sample_df = X_test_onnx.sample(n=500, random_state=42)

    inference_pool = [
        {
            "client_id": int(idx),
            "features": row.to_dict()
        }
        for idx, row in sample_df.iterrows()
    ]

    with open("inference_pool.json", "w") as f:
        json.dump(inference_pool, f)

    mlflow.log_artifact("inference_pool.json")


# ============================================================
# TRAIN FUNCTION
# ============================================================
def train_and_log(model, model_name):

    with mlflow.start_run(run_name=model_name):

        # ------------------------
        # TRAIN
        # ------------------------
        model.fit(X_train_onnx, y_train)

        # ------------------------
        # EVAL (SKLEARN MODEL)
        # ------------------------
        y_proba = model.predict_proba(X_test_onnx)[:, 1]

        threshold = find_best_threshold(y_test, y_proba)
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "AUC": roc_auc_score(y_test, y_proba),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Business_Cost": business_cost(y_test, y_pred),
            "Threshold": threshold,
        }

        mlflow.log_metrics(metrics)
        mlflow.log_params(model.get_params())

        print(f"\n{model_name} metrics:")
        print(metrics)

        # ------------------------
        # SAVE SKLEARN MODEL
        # ------------------------
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_train.iloc[:5]
        )

        # ------------------------
        # CONVERT TO ONNX
        # ------------------------
        initial_type = [("input", FloatTensorType([None, X_train_onnx.shape[1]]))]

        if "XGB" in model_name:
            onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
        else:
            onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type)

        onnx_path = ARTIFACTS_DIR / f"{model_name}.onnx"
        onnx.save_model(onnx_model, onnx_path)

        mlflow.onnx.log_model(onnx_model, artifact_path="onnx_model")

        # ------------------------
        # SAVE PREPROCESSING
        # ------------------------
        joblib.dump(imputer, ARTIFACTS_DIR / f"{model_name}_imputer.joblib")
        joblib.dump(features, ARTIFACTS_DIR / f"{model_name}_features.joblib")
        joblib.dump(onnx_feature_names, ARTIFACTS_DIR / f"{model_name}_onnx_feature.joblib")

        with open(ARTIFACTS_DIR / f"{model_name}_threshold.json", "w") as f:
            json.dump({"threshold": float(threshold)}, f)

        mlflow.log_artifact(ARTIFACTS_DIR / f"{model_name}_threshold.json")

        print(f"{model_name} saved successfully")


# ============================================================
# RUN MODELS
# ============================================================

# XGBOOST
xgb_model = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=400,
    max_depth=6,
    eval_metric="logloss",
    random_state=42,
)

train_and_log(xgb_model, "XGBoost")


# LIGHTGBM
lgb_model = lgb.LGBMClassifier(
    learning_rate=0.05,
    n_estimators=200,
    num_leaves=31,
    random_state=42,
)

train_and_log(lgb_model, "LightGBM")