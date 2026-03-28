#notebook/drift_analysis.py

###############################################################
# Importations
###############################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")

from evidently.report import Report
from evidently.presets import DataDriftPreset
import time

###############################################################
# load data, cleaning & visualization
###############################################################
#load data
reference = pd.read_csv("reference_data.csv")
current_raw = pd.read_json("prediction_logs.json", lines=True)
print("Reference shape:", reference.shape)
print("Current shape:", current_raw.shape)
#clean data
reference = reference.select_dtypes(include=["number"])
current = current_raw.select_dtypes(include=["number"])
#visualize
for col in reference.columns:
    if col in current.columns:
        plt.figure()
        sns.kdeplot(reference[col], label="Reference")
        sns.kdeplot(current[col], label="Current")
        plt.title(f"Distribution comparison: {col}")
        plt.legend()
        plt.show()

current_raw["timestamp"] = pd.to_datetime(current_raw["timestamp"])
current_raw.set_index("timestamp")["prediction_probability"].plot(title="Prediction over time")
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