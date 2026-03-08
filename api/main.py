# api/main.py

import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import mlflow

import time
import json
from pathlib import Path
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from api.load_mlflow_models import load_model_bundle
from api.logger import log_prediction
from elasticsearch import Elasticsearch 

LOG_FILE = "prediction_logs.json"

app = FastAPI(title="HomeCredit Scoring API")
MODELS = {}

# ======================================================
# ELASTICSEARCH CONNECTION
# ======================================================

es = Elasticsearch("http://localhost:9200")

def log_prediction(data):

    try:
        es.index(
            index="mlops_predictions",
            document=data
        )
    except Exception as e:
        print("Elastic logging failed:", e)

# ======================================================
# LOAD MODELS AT STARTUP (FAIL FAST)
# ======================================================

def predict_random_sample(model_key: str,client_index: int):
    start_time = time.time()
    bundle = MODELS[model_key]
    pool = bundle["inference_pool"]
    item = next((x for x in pool if x["Client_index"]==client_index),None)
    if item is None:
        raise HTTPException(
            status_code=404,
            detail=f"Client_index {client_index} not found in inference pool"
        )

  
    sample = item["features"]
    input_df = pd.DataFrame([sample])
    input_df = input_df.reindex(columns=bundle["features"])
    #input_df = bundle["imputer"].transform(input_df)

    pred = bundle["model"].predict_proba(input_df)[:, 1]
    # Gestion robuste des sorties MLflow
    if hasattr(pred, "__len__"):
        proba = float(pred[0])
    else:
        proba = float(pred)

    prediction = int(proba >= bundle["threshold"])
    latency = time.time() - start_time
    log_data = {
    "timestamp": time.time(),
    "model": model_key,
    "client_index": client_index,
    "prediction_probability": proba,
    "prediction": prediction,
    "latency": latency
    }
    log_prediction(log_data)
    
    return {
        "model": model_key,
        "client_index": client_index,
        "prediction_probability": proba,
        "prediction": prediction,
        "threshold": bundle["threshold"],
    }


class request_index(BaseModel):
    Client_index : int
    

# ======================================================
# LOAD MODELS AT STARTUP
# ======================================================
@app.on_event("startup")
def load_models():

    global MODELS

    # Ne pas charger les modèles pendant les tests CI
    if os.getenv("CI") == "true":
     return

    try:
        MODELS = {
            "xgboost": load_model_bundle(
                model_name="HomeCredit_Scoring_final_XGBoost",
                stage=None,
                flavor="xgb"
            ),
            "lightgbm": load_model_bundle(
                model_name="HomeCredit_Scoring_final_LightGBM",
                stage=None,
                flavor="lgb"
            )
        }
    except Exception as e:
        raise RuntimeError(f"Failed to load ML models: {e}")
    
@app.get("/health")
def health():
    if not MODELS:
        return {
            "status": "API running in CI mode (no models loaded)",
            "available_models": [],
            "size fichier test": 0,
            "list of index": []
        }
    pool = MODELS["xgboost"]["inference_pool"]
    list_index = [item["Client_index"] for item in pool] 
    
    return {
        "status": "API running well",
        "available_models": list(MODELS.keys()),
        "size fichier test": len(pool),
        "list of index":list_index
    }

@app.get("/models/info")
def models_info():
    info = {}

    for model_key, bundle in MODELS.items():
        pool = bundle["inference_pool"]

        info[model_key] = {
            "model_name": bundle["model_name"],
            "model_version": bundle["model_version"],
            "run_id": bundle["run_id"],
            "threshold": bundle["threshold"],
            "nb_features": len(bundle["features"]),
            "nb_clients_inference_pool": len(pool),
        }

    return info

@app.get("/monitoring/drift")
def detect_drift():

    response = es.search(
        index="mlops_predictions",
        size=500
    )

    data = [hit["_source"] for hit in response["hits"]["hits"]]

    df = pd.DataFrame(data)

    if len(df) < 20:
        return {"message": "Not enough data for drift detection"}

    reference = df.iloc[:10]
    current = df.iloc[10:]

    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference,
        current_data=current
    )

    result = report.as_dict()

    drift_score = result["metrics"][0]["result"]["dataset_drift"]
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

    drift_doc = {
        "timestamp": time.time(),
        "dataset_drift": drift_score,
        "drifted_features_share": drift_share
    }

    es.index(
        index="mlops_drift_metrics",
        document=drift_doc
    )

    return drift_doc

@app.post("/predict/XGBoost")
def predict_xgboost_random(request:request_index):
    return predict_random_sample("xgboost",request.Client_index)

@app.post("/predict/LightGBM")
def predict_lightgbm_random(request:request_index):
    return predict_random_sample("lightgbm",request.Client_index)


if __name__ == "__main__":
    
    print("MLFLOW URI:", mlflow.get_tracking_uri())
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )


