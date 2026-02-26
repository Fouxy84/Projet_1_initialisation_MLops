# api/main.py
import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from load_mlflow_models import load_model_bundle

app = FastAPI(title="HomeCredit Scoring API")


# ======================================================
# LOAD MODELS AT STARTUP (FAIL FAST)
# ======================================================
try:
    MODELS = {
        "xgboost": load_model_bundle(model_name="HomeCredit_Scoring_final_XGBoost",stage=None,flavor="xgb"),
        "lightgbm": load_model_bundle(model_name="HomeCredit_Scoring_final_LightGBM",stage=None,flavor="lgb")
    }
except Exception as e:
    raise RuntimeError(f"Failed to load ML models: {e}")


def predict_random_sample(model_key: str,client_index: int):
    
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

    pred = bundle["model"].predict(input_df)
    # Gestion robuste des sorties MLflow
    if hasattr(pred, "__len__"):
        proba = float(pred[0])
    else:
        proba = float(pred)

    prediction = int(proba >= bundle["threshold"])

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
# ROUTES
# ======================================================
@app.get("/health")
def health():
    pool = MODELS["xgboost"]["inference_pool"]
    list_index = [item["Client_index"] for item in pool] 
    
    return {
        "status": "API running well",
        "available_models": list(MODELS.keys()),
        "size fichier test": len(pool),
        "list of index":list_index
    }


@app.post("/predict/XGBoost")
def predict_xgboost_random(request:request_index):
    return predict_random_sample("xgboost",request.Client_index)


@app.post("/predict/LightGBM")
def predict_lightgbm_random(request:request_index):
    return predict_random_sample("lightgbm",request.Client_index)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )


