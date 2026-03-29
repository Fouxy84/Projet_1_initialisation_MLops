# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:13:42 2026

@author: coach
"""
import logging
import json
import os
from datetime import datetime

LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "prediction_logs.json")

handler = logging.FileHandler(LOG_FILE)

logger = logging.getLogger("mlops_api")
logger.setLevel(logging.INFO)

#handler = logging.FileHandler("prediction_logs.json")
#handler = logging.FileHandler("/logs/prediction_logs.json")
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)


def log_prediction(data):
    print("LOG CALLED")
    log_entry = {"timestamp": datetime.utcnow().isoformat(),**data}
    logger.info(json.dumps(log_entry))

if __name__ == "__main__":
   
    log_data = {
    "timestamp": datetime.utcnow().isoformat(),
    "model": "xgboost",
    "client_index": 4242,
    "prediction_probability": 0.85,
    "prediction": 1,
    "latency": 0.1,
    "features": {}
    }
    log_prediction(log_data)