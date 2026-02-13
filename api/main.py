# api/main.py
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
import pandas as pd

import time
from pydantic import BaseModel

app = FastAPI(title="HomeCredit Scoring API")

MODEL_PATH = "file:///C:/Users/coach/Desktop/datascientest/OpenClassrooms/Projects_MLops/Projet_1_initialisation_MLops/notebook/mlruns/models/HomeCredit_Scoring_final_LightGBM/version-5"
MODEL_PATH = r"C:\Users\coach\Desktop\datascientest\OpenClassrooms\Projects_MLops\mlruns-20260212T220513Z-1-001\mlruns\models\HomeCredit_Scoring_final_LightGBM\version-5"

#model = mlflow.pyfunc.load_model(MODEL_PATH)

best_thresold = 0.5

# ---- Schema Pydantic ----
class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_EMPLOYED: float
    DAYS_BIRTH: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str 
    NAME_HOUSING_TYPE: str

@app.get("/")
def health():
    return {"status": "API running well"}

@app.post("/predict")
def predict(data: ClientData):
    try:
        start_time = time.time()
        input_df = pd.DataFrame([data.dict()])
        proba = 0.2 #model.predict(input_df)[0]
        prediction = int(proba >= best_thresold)
        latency = time.time() - start_time
        
        return {"prediction_probability": float(proba),"prediction": prediction,"latency_seconds": round(latency, 4)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
