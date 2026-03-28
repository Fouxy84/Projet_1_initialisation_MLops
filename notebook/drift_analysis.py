#notebook/drift_analysis.py

###############################################################
# Importations
###############################################################
from matplotlib.pylab import sample
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")

from evidently import Report
from evidently.presets import DataDriftPreset
import time
from pathlib import Path

###############################################################
# load data, cleaning & visualization
###############################################################
#load data
reference_path = Path(r"C:\Users\coach\Desktop\datascientest\OpenClassrooms\Projects_MLops\Projet_1_initialisation_MLops\data\proceed\homecredit_features.csv")
current_path = Path(r"C:\Users\coach\Desktop\datascientest\OpenClassrooms\Projects_MLops\Projet_MLops_1\Projet_1_initialisation_MLops\logs\prediction_logs.json")
reference_raw = pd.read_csv(reference_path)
current_raw = pd.read_json(current_path, lines=True)
print("Reference shape:", reference_raw.shape)
print("Current shape:", current_raw.shape)
#clean data
current_log = pd.json_normalize(current_raw["features"])
reference = reference_raw.select_dtypes(include=["number"])
current = current_log.select_dtypes(include=["number"])
print("Reference 2 shape:", reference.shape)
print("Current 2 shape:", current.shape)

common_cols = list(set(reference.columns).intersection(set(current.columns)))
reference = reference[common_cols]
current = current[common_cols]
print("Final aligned shape:")
print("Reference:", reference.shape)
print("Current:", current.shape)

current_raw["timestamp"] = pd.to_datetime(current_raw["timestamp"])
current_raw.set_index("timestamp")["prediction_probability"].plot(title="Prediction over time",figsize=(10,5))
plt.show()

###############################################################
# Drift & monitoring
###############################################################
# Drift detection
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference,current_data=current)
report.show()
# Extract drift results
result = report.as_dict()
drift_score = result["metrics"][0]["result"]["dataset_drift"]
drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]
print("Dataset drift:", drift_score)
print("Drifted features (%):", drift_share)

###############################################################
# elasticsearch saving
###############################################################
doc = {"timestamp": time.time(),"dataset_drift": drift_score,"drifted_features_share": drift_share}

es.index(index="mlops_drift_metrics",document=doc)

###############################################################
# drift analysis
###############################################################
'''
L’analyse montre que le dataset_drift est de [True/False] avec une proportion de variables dérivées de [X%].
Certaines variables présentent des différences de distribution entre les données de référence et les données de production.
Ces écarts peuvent être causés par :
- une évolution des comportements utilisateurs,
- une modification des données d’entrée,
- ou une dérive dans la collecte des données.
## Impact
Cette dérive peut entraîner une baisse de performance du modèle en production.
## Recommandations
- Mettre en place un monitoring continu
- Déclencher un réentraînement du modèle si nécessaire
'''