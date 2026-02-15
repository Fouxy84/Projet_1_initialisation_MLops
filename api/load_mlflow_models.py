from mlflow.tracking import MlflowClient
import mlflow.pyfunc

def load_latest_model(registered_model_name: str):
    client = MlflowClient()
    # Récupère toutes les versions du modèle enregistré triées par version décroissante
    versions = client.get_latest_versions(registered_model_name, stages=["None", "Staging", "Production"])
    
    if not versions:
        raise ValueError(f"Aucun modèle trouvé pour : {registered_model_name}")
    
    # Choisir la version la plus récente (ou avec stage Production si vous préférez)
    latest_model_version = versions[0]
    
    model_uri = f"models:/{registered_model_name}/{latest_model_version.version}"
    
    # Charge le modèle avec MLflow pyfunc
    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Chargé modèle {registered_model_name} version {latest_model_version.version}")
    return model