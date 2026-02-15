
# api/load_mlflow_models.py
import json
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path


MLFLOW_TRACKING_URI = "file:/notebook/mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

def load_model_bundle(model_name: str, stage: str = "Production", model_str):
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)

    model_version = client.get_latest_versions(model_name, [stage])[0]
    run_id = model_version.run_id

    artifacts_path = Path(
        mlflow.artifacts.download_artifacts(run_id=run_id)
    )

    imputer = joblib.load(artifacts_path / "artifacts" / model_str /"imputer.joblib")
    features = joblib.load(artifacts_path / "artifacts" / model_str / "features.joblib")

    with open(artifacts_path / "artifacts" / model_str / "threshold.json") as f:
        threshold = json.load(f)["best_threshold"]

    return {
        "model": model,
        "imputer": imputer,
        "features": features,
        "threshold": threshold
    }
