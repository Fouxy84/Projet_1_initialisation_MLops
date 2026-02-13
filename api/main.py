# api/main.py
import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
import pandas as pd
#from app.schema import ClientData
from api.load_data import model_lgb
#from api.schema import ClientData
import time
from pydantic import BaseModel

app = FastAPI(title="HomeCredit Scoring API")

#model_path = "file:///C:/Users/coach/Desktop/datascientest/OpenClassrooms/Projects_MLops/Projet_1_initialisation_MLops/notebook/mlruns/models/HomeCredit_Scoring_final_LightGBM/version-5"
#model = mlflow.pyfunc.load_model(model_path)
best_thresold = 0.5

# ---- Schema Pydantic ----
class ClientData(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_EMPLOYED: float
    DAYS_BIRTH: float
    # ajouter toutes les features nécessaires

@app.get("/")
def health():
    return {"status": "API running well"}

@app.post("/predict")
def predict(data: ClientData):
    try:
        start_time = time.time()
        input_df = pd.DataFrame([data.dict()])
        # selon ton modèle MLflow
        proba = model_lgb.predict(input_df)[0]
        prediction = int(proba >= best_thresold)
        latency = time.time() - start_time
        
        return {"prediction_probability": float(proba),"prediction": prediction,"latency_seconds": round(latency, 4)}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
