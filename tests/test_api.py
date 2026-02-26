# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 23:39:31 2026

@author: coach
"""

# tests/test_api.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ======================================================
# Import app
# ======================================================
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "api"))

import main


# ======================================================
# Fake model pour les tests
# ======================================================
class DummyModel:
    def predict(self, X):
        # retourne une probabilité fixe
        return np.array([0.6])


# ======================================================
# Fixture : client API avec MODELS mocké
# ======================================================
@pytest.fixture(scope="module")
def client(monkeypatch):
    fake_pool = [
        {
            "Client_index": 123,
            "features": {"feat1": 1.0, "feat2": 2.0}
        }
    ]

    fake_bundle = {
        "model": DummyModel(),
        "features": ["feat1", "feat2"],
        "threshold": 0.5,
        "model_name": "dummy_model",
        "model_version": "1",
        "run_id": "abc123",
        "inference_pool": fake_pool,
    }

    monkeypatch.setattr(
        main,
        "MODELS",
        {
            "xgboost": fake_bundle,
            "lightgbm": fake_bundle,
        },
    )

    return TestClient(main.app)


# ======================================================
# Tests
# ======================================================

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "API running well"
    assert "xgboost" in data["available_models"]
    assert data["size fichier test"] == 1
    assert data["list of index"] == [123]


def test_models_info(client):
    response = client.get("/models/info")
    assert response.status_code == 200

    data = response.json()
    assert "xgboost" in data
    assert data["xgboost"]["model_name"] == "dummy_model"
    assert data["xgboost"]["nb_features"] == 2


def test_predict_xgboost_ok(client):
    response = client.post(
        "/predict/XGBoost",
        json={"Client_index": 123},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["model"] == "xgboost"
    assert data["client_index"] == 123
    assert 0 <= data["prediction_probability"] <= 1
    assert data["prediction"] in [0, 1]


def test_predict_lightgbm_ok(client):
    response = client.post(
        "/predict/LightGBM",
        json={"Client_index": 123},
    )

    assert response.status_code == 200


def test_predict_client_not_found(client):
    response = client.post(
        "/predict/XGBoost",
        json={"Client_index": 999},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"]