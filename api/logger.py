# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 22:13:42 2026

@author: coach
"""

import logging
import json
from datetime import datetime
from elasticsearch import Elasticsearch

logger = logging.getLogger("mlops_api")
logger.setLevel(logging.INFO)

handler = logging.FileHandler("prediction_logs.json")
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)
es = Elasticsearch("http://localhost:9200")

def log_prediction(data):
    logger.info(json.dumps(data))
    es.index(index="mlops_predictions",document=data)