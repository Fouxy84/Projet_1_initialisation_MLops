---
title: HomeCredit MLOps API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# ML_ops_OC_2

# Home Credit MLOps Project

## Objective
Build an end-to-end MLOps pipeline to predict loan default risk.

## Structure
-api/ # FastAPI application (inference) 
- artifacts/ # exported models (ONNX, preprocessing) 
- src/ # training pipeline 
- notebooks/ # experiments & EDA logs/ # prediction logs 
- fluentd/ # logging configuration

## Stack
- Python
- Scikit-learn
- XGBoost
- LightGBM
- MLflow
- ONNX Runtime
- FastAPI
- Docker / Docker Compose
- Elasticsearch
- Fluentd
- Grafana


## api & docker
#MLFLOW
cd 'C:\Users\coach\Desktop\datascientest\OpenClassrooms\PROJET_2\Projet_1_initialisation_MLops\notebook\mlruns' 
mlflow ui

#API
uvicorn main:app --reload


#Docker
docker build --no-cache -t homecredit_scorer .
docker run -p 8000:8000 homecredit_scorer

#Pipeline CI/CD:
1. Push du code sur GitHub
2. GitHub Actions lance les tests avec pytest
3. Si les tests passent :
   - build de l'image Docker
   - lancement du conteneur pour test API
4. Déploiement automatique vers Hugging Face Spaces MLops_2 https://huggingface.co/spaces/Fouxy84/MLops_2
5. L’API est accessible via Swagger Localhost:8000/docs

# Monitoring
lancement docker-compose: 
docker-compose build --no-cache  # construction l'image d'un conteneur
docker-compose up -d             # run de l'image ==> creation d'un conteneur
docker ps                        # liste des conteneurs en cours
docker-compose down              # remove image + conteneur


