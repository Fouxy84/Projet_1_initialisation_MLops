import os
import logging
import json
import pandas as pd
from datetime import datetime

LOG_DIR = os.getenv("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)

PRED_LOG_FILE = os.path.join(LOG_DIR, "prediction_logs.json")
PROF_LOG_FILE = os.path.join(LOG_DIR, "profiling_logs.json")

# ---------------------------
# LOGGER PREDICTION
# ---------------------------
prediction_logger = logging.getLogger("prediction_logger")
prediction_logger.setLevel(logging.INFO)

if not prediction_logger.handlers:
    pred_handler = logging.FileHandler(PRED_LOG_FILE)
    pred_handler.setFormatter(logging.Formatter('%(message)s'))
    prediction_logger.addHandler(pred_handler)

# ---------------------------
# LOGGER PROFILING
# ---------------------------
profiling_logger = logging.getLogger("profiling_logger")
profiling_logger.setLevel(logging.INFO)

if not profiling_logger.handlers:
    prof_handler = logging.FileHandler(PROF_LOG_FILE)
    prof_handler.setFormatter(logging.Formatter('%(message)s'))
    profiling_logger.addHandler(prof_handler)

# ---------------------------
# FUNCTIONS
# ---------------------------
def log_prediction(data):
    log_entry = {"timestamp": datetime.utcnow().isoformat(),**data}
    prediction_logger.info(json.dumps(log_entry))


def log_profiling(data):
    log_entry = {"timestamp": datetime.utcnow().isoformat(),**data}
    profiling_logger.info(json.dumps(log_entry))

