# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:13:42 2026

@author: coach
"""
import logging
import json
from datetime import datetime

logger = logging.getLogger("mlops_api")
logger.setLevel(logging.INFO)

#handler = logging.FileHandler("prediction_logs.json")
handler = logging.FileHandler("/logs/prediction_logs.json")
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)


def log_prediction(data):

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        **data
    }

    logger.info(json.dumps(log_entry))