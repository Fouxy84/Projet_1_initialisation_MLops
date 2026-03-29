import os
import logging
import json
import pandas as pd
from datetime import datetime

LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "prediction_logs.json")

logger = logging.getLogger("mlops_api")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def log_prediction(data):
    print("LOG CALLED")
    log_entry = {"timestamp": datetime.utcnow().isoformat(),**data}
    logger.info(json.dumps(log_entry))

