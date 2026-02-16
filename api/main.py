# api/main.py
import time
import pandas as pd
from fastapi import FastAPI, HTTPException

from inference_data import ClientData
from load_mlflow_models import load_model_bundle

app = FastAPI(title="HomeCredit Scoring API")


# ======================================================
# LOAD MODELS AT STARTUP (FAIL FAST)
# ======================================================
try:
    MODELS = {
        "xgboost": load_model_bundle(
            model_name="HomeCredit_Scoring_final_XGBoost",
            stage=None,
            flavor="xgb",
        ),
        "lightgbm": load_model_bundle(
            model_name="HomeCredit_Scoring_final_LightGBM",
            stage=None,
            flavor="lgb",
        ),
    }
except Exception as e:
    raise RuntimeError(f"Failed to load ML models: {e}")

def _predict_with_model(model_key: str, data: ClientData):
    start_time = time.time()

    bundle = MODELS[model_key]
    model = bundle["model"]
    imputer = bundle["imputer"]
    features = bundle["features"]
    threshold = bundle["threshold"]

    # ---- Input preprocessing
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df.reindex(columns=features)

    input_df = imputer.transform(input_df)

    # ---- Prediction
    proba = float(model.predict(input_df)[0][1])
    prediction = int(proba >= threshold)
    
    print(type(model))
    print(model.predict(input_df))

    latency = time.time() - start_time

    return {
        "model": model_key,
        "prediction_probability": float(proba),
        "prediction": prediction,
        "threshold": threshold,
        "latency_seconds": round(latency, 4),
    }

import random

def predict_random_sample(model_key: str):
    bundle = MODELS[model_key]

    pool = bundle["inference_pool"]
    if not pool:
        raise HTTPException(status_code=500, detail="Inference pool not available")

    # ---- Pick random client
    sample = random.choice(pool)
    input_df = pd.DataFrame([sample])

    # ---- Reorder + impute
    input_df = input_df.reindex(columns=bundle["features"])
    input_df = bundle["imputer"].transform(input_df)

    # ---- Predict
    proba = float(bundle["model"].predict(input_df)[0][1])
    prediction = int(proba >= bundle["threshold"])

    return {
        "model": model_key,
        "prediction_probability": float(proba),
        "prediction": prediction,
        "threshold": bundle["threshold"],
        "sample_used": sample,
    }



# ======================================================
# ROUTES
# ======================================================
@app.get("/")
def health():
    return {
        "status": "API running well",
        "available_models": list(MODELS.keys()),
    }



@app.post("/predict/XGBoost")
def predict_xgboost_random():
    return predict_random_sample("xgboost")


@app.post("/predict/LightGBM")
def predict_lightgbm_random():
    return predict_random_sample("lightgbm")

