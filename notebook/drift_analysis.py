#notebook/drift_analysis.py

###############################################################
# Importations
print("1/ Importing libraries...")
###############################################################
import re

from matplotlib.pylab import sample
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")

from evidently import Report
from evidently.presets import DataDriftPreset
import time
from pathlib import Path
import json

###############################################################
# load data, cleaning & visualization
print("2/ Loading and cleaning data...")
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

reference = reference.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
current = current.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
reference = reference.fillna(0)
current = current.fillna(0)

def remove_constant(df):
    return df.loc[:, df.nunique() > 1]
reference = remove_constant(reference)
current = remove_constant(current)

print("Reference 2 shape:", reference.shape)
print("Current 2 shape:", current.shape)
print("NaN in reference:", reference.isna().sum().sum())
print("NaN in current:", current.isna().sum().sum())
print("Constant cols:", (reference.nunique() <= 1).sum())

common_cols = list(set(reference.columns).intersection(set(current.columns)))
reference = reference[common_cols]
current = current[common_cols]
print("Final aligned shape:")
print("Reference:", reference.shape)
print("Current:", current.shape)

current_raw["timestamp"] = pd.to_datetime(current_raw["timestamp"])
current_raw.set_index("timestamp")["prediction_probability"].plot(title="Prediction over time",figsize=(10,5))
plt.show(block=False)
plt.tight_layout()
plt.savefig("prediction_over_time.png")
plt.pause(30)   # Afficher le graphique pendant 30 secondes
plt.close()

###############################################################
# Drift & monitoring
print("3/ Performing drift analysis...")
###############################################################
# Drift detection
report = Report([DataDriftPreset()])
#report = Report([DataDriftPreset(method="psi")],include_tests="True")
drift_eval = report.run(reference_data=reference,current_data=current)
drift_eval.save_html("drift_report.html")
drift_eval.save_json("drift_report.json")
print(drift_eval)

result = drift_eval.dict()
#drift_score = result["metrics"][0]["result"]["dataset_drift"]
#drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]


drift_share = result["metrics"][0]["value"]["share"]
drift_score = drift_share > 0.5

print("Dataset drift:", drift_score)
print("Drifted features (%):", drift_share)
###############################################################
# elasticsearch saving
print("4/ Saving results to Elasticsearch...")
###############################################################
doc = {"timestamp": time.time(),"dataset_drift": drift_score,"drifted_features_share": drift_share}

es.index(index="mlops_drift_metrics",document=doc)

###############################################################
# drift analysis
###############################################################
'''
### Analyse du drift des données

L’analyse met en évidence un data drift de l'ordre de 89% des variables présentes entre les données de référence (homecredit_features.csv)
 et les données de production (logs/prediction_logs.json).
 l'ecart est signficatif car les modèles ont ete entrainés sur 49% des données de référence (150k clients sur 307k)
 ceci pour une question de temps de calcul et de ressources, mais cela a pu introduire un biais dans les données d'entrainement.
'''
print("5/ Drift analysis completed.")