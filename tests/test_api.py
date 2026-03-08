# tests/test_api.py

from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import api.main as main
import numpy as np
#import main 

fake_model = MagicMock()
fake_model.predict.return_value = np.array([0.8])

fake_bundle = {
    "model": fake_model,
    "threshold": 0.5,
    "features": [],
    "inference_pool": [
        {
            "Client_index": 1,
            "features": {}
        }
    ],
    "model_name": "fake_model",
    "model_version": 1,
    "run_id": "test_run"
}

client = TestClient(main.app)

main.MODELS = {
    "xgboost": fake_bundle,
    "lightgbm": fake_bundle
}
client = TestClient(main.app)

def setup_module():
    main.MODELS = {
        "xgboost": fake_bundle,
        "lightgbm": fake_bundle
    }

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_models_info():
    r = client.get("/models/info")
    assert r.status_code == 200

def test_predict():
    payload = {"Client_index": 1}
    r = client.post("/predict/XGBoost", json=payload)
    data = r.json()
    assert r.status_code == 200
    assert "prediction_probability" in data
    assert "prediction" in data
    assert abs(data["prediction_probability"] - 0.8) < 1e-6

def test_client_not_found():
    payload = {"Client_index": 999}
    r = client.post("/predict/XGBoost", json=payload)
    assert r.status_code == 404

def test_missing_field():
    r = client.post("/predict/XGBoost",json={})
    assert r.status_code == 422

def test_wrong_type():
    r = client.post("/predict/XGBoost",json={"Client_index": "abc"})
    assert r.status_code == 422