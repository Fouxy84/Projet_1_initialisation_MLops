# api/main.py
import joblib
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
import pandas as pd
import load_mlflow_models
from inference_data import ClientData
from load_mlflow_models import load_model_bundle
import time
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="HomeCredit Scoring API")
best_thresold = 0.5
MODELS = {
    "xgboost": load_model_bundle("HomeCredit_Scoring_final_XGBoost","Production","xgb"),
    "lightgbm": load_model_bundle("HomeCredit_Scoring_final_LightGBM","Production","lgb"),
}


@app.get("/")
def health():
    return {"status": "API running well", "available_models": list(MODELS.keys())}

@app.post("/predict/{model_name}")
def predict(model_name: str, data: ClientData):
    try:
        start_time = time.time()
        model_info = MODELS[model_name]
        imputer = model_info["imputer"]
        model = model_info["model"]
        features = model_info["features"]
        threshold = model_info["threshold"]

        input_df = pd.DataFrame([data.dict()])
        input_df = input_df[features]
        input_df = imputer.transform(input_df)
        proba = model.predict_proba(input_df)[0][1]
        prediction = int(proba >= threshold)
        latency = time.time() - start_time
        return {"prediction_probability": float(proba),"prediction": prediction,"latency_seconds": round(latency, 4)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
