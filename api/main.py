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
best_thresold =0.1

try:
    MODELS = {"xgboost": load_model_bundle(model_name="HomeCredit_Scoring_final_XGBoost",stage=None,flavor="xgb"),
        "lightgbm": load_model_bundle( model_name="HomeCredit_Scoring_final_LightGBM",stage=None,flavor="lgb")}
except Exception as e:
    # Fail fast if models cannot be loaded
    raise RuntimeError(f" Failed to load ML models: {e}")

bundle = load_model_bundle("HomeCredit_Scoring_final_XGBoost",stage=None,flavor="xgb")
print(bundle.keys())
print(len(bundle["features"]))

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

        input_df = pd.DataFrame([data.dict()]) #dataframe
        input_df = input_df[features] #featuring
        input_df = input_df.reindex(columns=features) #featuring
        input_df = imputer.transform(input_df) # imputation valeurs manquantes
        proba = model.predict_proba(input_df)[0][1] # proba prediction
        prediction = int(proba >= threshold) # seuil de décision
        latency = time.time() - start_time #latence de traitement
        return {"model": model_name, 
                "prediction_probability": float(proba),
                "prediction": prediction,
                "latence": round(latency, 4)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
