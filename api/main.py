# api/main.py

import os
from flask import logging
import pandas as pd
from fastapi import FastAPI, HTTPException, logger
from pydantic import BaseModel
from scipy import stats
import uvicorn
import mlflow
from datetime import datetime
import time
import json
from pathlib import Path
import cProfile
import pstats
import io
#from evidently.report import Report
#from evidently.presets import DataDriftPreset
#from evidently.metrics import DataDriftTable
import onnxruntime as rt
import numpy as np
from pathlib import Path
import cProfile
import pstats
import io
import logging

from api.load_mlflow_models import load_model_bundle
from api.logger import log_prediction
#from load_mlflow_models import load_model_bundle
#from logger import log_prediction
from elasticsearch import Elasticsearch 

LOG_FILE = "prediction_logs.json"

app = FastAPI(title="HomeCredit Scoring API")
MODELS = {}

sessions = {}
input_names = {}
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
sessions = {
    "xgboost": rt.InferenceSession(str(ARTIFACTS_DIR / "XGBoost.onnx")),
    "lightgbm": rt.InferenceSession(str(ARTIFACTS_DIR / "LightGBM.onnx")),
}

for name, path in {
    "xgboost": ARTIFACTS_DIR / "XGBoost.onnx",
    "lightgbm": ARTIFACTS_DIR / "LightGBM.onnx",
}.items():
    if not path.exists():
        raise FileNotFoundError(f"{name} model not found at {path}")
    print(f"Loaded {name} from {path}")

#sessions = {"xgboost": rt.InferenceSession("api/xgb_model.onnx"),"lightgbm": rt.InferenceSession("api/lgb_model.onnx"),}
input_names = {k: v.get_inputs()[0].name for k, v in sessions.items()}

# ======================================================
# ELASTICSEARCH CONNECTION
# ======================================================

ELASTIC_HOST = os.getenv("ELASTIC_HOST", "http://localhost:9200")
es = Elasticsearch(ELASTIC_HOST)

# ======================================================
# LOAD MODELS AT STARTUP (FAIL FAST)
# ======================================================

def predict_random_sample(model_key: str,client_index: int):
    profiler = cProfile.Profile()
    profiler.enable()

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
    input_df = input_df.fillna(0)
    input_df = input_df.reindex(columns=bundle["features"])
    session = sessions[model_key]
    input_name = input_names[model_key]

    if input_df.shape[1] == 0:
        pred = bundle["model"].predict_proba(input_df)[:, 1]
        proba = float(pred[0])
    else:
        input_array = input_df.to_numpy().astype(np.float32)
        outputs = session.run(None, {input_name: input_array})
        proba = float(outputs[1][0][1])

    #input_array = input_df.to_numpy().astype(np.float32)
    #outputs = session.run(None, {input_name: input_array})
    #proba = float(outputs[1][0][1])

    prediction = int(proba >= bundle["threshold"])
    latency = time.time() - start_time

    log_data = {
    "timestamp": datetime.now().isoformat(),
    "model": model_key,
    "client_index": client_index,
    "prediction_probability": proba,
    "prediction": prediction,
    "latency": latency,
    "features": sample
    }

    log_prediction(log_data)
    print("LOGGING:", log_data)
    
    profiler.disable()
    
    # --- PROFILING ---
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(100) #afiichage 100 focntions les plus couteuses en temps cumulés
    profilt_out = s.getvalue()

    print("===== PROFILING =====")
    print(profilt_out)

    # --- STRUCTURED LOGGING ---
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")

    top_functions = []

    for func, stat in list(stats.stats.items())[:10]:
        filename, line, name = func
        cc, nc, tt, ct, callers = stat

        top_functions.append({
            "function": name,
            "file": filename.split("/")[-1],
            "cumulative_time": round(ct, 6),
            "total_calls": nc
        })

    logger = logging.getLogger("mlops_api")

    logger.info(json.dumps({
        "type": "profiling",
        "model": model_key,
        "client_index": client_index,
        "latency": round(latency, 6),
        "top_functions": top_functions
    }))

    return {
        "model": model_key,
        "client_index": client_index,
        "prediction_probability": proba,
        "prediction": prediction,
        "threshold": bundle["threshold"],
        "latency_seconds": latency,
        "profiling": profilt_out
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
def get_drift_metrics():
    if not es.indices.exists(index="mlops_drift_metrics"):
        return {"message": "No drift metrics available yet"}

    response = es.search(
        index="mlops_drift_metrics",
        size=10,
        sort=[{"timestamp": {"order": "desc"}}]
    )

    results = [hit["_source"] for hit in response["hits"]["hits"]]

    return {
        "count": len(results),
        "drift_metrics": results
    }

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


