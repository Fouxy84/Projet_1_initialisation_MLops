# ML_ops_OC_1

# Home Credit MLOps Project

## Objective
Build an end-to-end MLOps pipeline to predict loan default risk.

## Structure
- data/: datasets (ignored)
- src/: preprocessing and training code
- notebooks/: experiments
- mlruns/: MLflow tracking

## Stack
- Python
- Scikit-learn
- XGBoost
- LightGBM
- MLflow


## api & docker
#MLFLOW
cd 'C:\Users\coach\Desktop\datascientest\OpenClassrooms\Projects_MLops\Projet_1_initialisation_MLops\notebook\mlruns' mlflow ui

#API
uvicorn main:app --reload

#Docker
docker build -t homecredit_scorer .
docker run -p 8000:8000 homecredit_scorer


