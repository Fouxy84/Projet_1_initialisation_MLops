# api/main.py
import joblib
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
import pandas as pd
from api.inference_data import ClientData
import time
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="HomeCredit Scoring API")

#MODEL_PATH = "file:///C:/Users/coach/Desktop/datascientest/OpenClassrooms/Projects_MLops/Projet_1_initialisation_MLops/notebook/mlruns/models/HomeCredit_Scoring_final_LightGBM/version-5"
#MODEL_PATH = r"C:\Users\coach\Desktop\datascientest\OpenClassrooms\Projects_MLops\mlruns-20260212T220513Z-1-001\mlruns\models\HomeCredit_Scoring_final_LightGBM\version-5"
#model = mlflow.pyfunc.load_model(MODEL_PATH)

BASE_DIR = Path(r"C:/Users/coach/Desktop/datascientest/OpenClassrooms/Projects_MLops/Projet_1_initialisation_MLops")
MODEL_PATH = BASE_DIR / "models" / "lightgbm_model.joblib"

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]
imputer = artifact["imputer"]

best_thresold = 0.5


@app.get("/")
def health():
    return {"status": "API running well"}

@app.post("/predict")
def predict(data: ClientData):
    try:
        start_time = time.time()
        input_df = pd.DataFrame([data.dict()])
        input_df = imputer.transform(input_df)
        proba = model.predict_proba(input_df)[0][1]
        prediction = int(proba >= best_thresold)
        latency = time.time() - start_time
        return {"prediction_probability": float(proba),"prediction": prediction,"latency_seconds": round(latency, 4)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
