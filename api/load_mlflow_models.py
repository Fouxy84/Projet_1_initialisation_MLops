# api/load_mlflow_models.py
import mlflow
import json
import joblib
from pathlib import Path
from typing import Optional

# ======================================================
# GLOBAL MLFLOW CONFIG (API = READ-ONLY)
# ======================================================
BASE_DIR = Path(__file__).resolve().parents[1]
MLFLOW_DB = BASE_DIR / "notebook" / "mlruns" / "mlflow.db"

#mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
mlflow.set_tracking_uri("file:./notebook/mlruns")
def load_inference_pool():
    with open("/app/api/inference_pool.json") as f:
        return json.load(f)

# ======================================================
# LOAD MODEL + ASSOCIATED ARTIFACTS
# ======================================================
def load_model_bundle(
    model_name: str,
    stage: Optional[str] = None,
    flavor: str = "xgb",
):
   
    client = mlflow.tracking.MlflowClient()

    # ---- Resolve model version
    versions = (
        client.get_latest_versions(model_name, [stage])
        if stage
        else client.get_latest_versions(model_name)
    )

    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    model_version = versions[0]
    run_id = model_version.run_id
    run = client.get_run(run_id)
    
    # Adjust artifact_uri for container environment
    artifact_uri = run.info.artifact_uri
    host_path = "file:///C:/Users/coach/Desktop/datascientest/OpenClassrooms/Projects_MLops/Projet_MLops_1/Projet_1_initialisation_MLops/"
    if artifact_uri.startswith(host_path):
        artifact_uri = artifact_uri.replace(host_path, "file:/app/")
    
    # In container, artifacts are at /app/artifacts
    artifact_root = Path("/app/artifacts")
    
    flavor_dir = artifact_root / flavor
    if not flavor_dir.exists():
        raise FileNotFoundError(
            f"Artifacts folder '{flavor}' not found for model '{model_name}'"
        )

    # ---- Load model (sklearn model)
    model = joblib.load(flavor_dir / "model.joblib")

    # ---- Load inference artifacts
    imputer = joblib.load(flavor_dir / "imputer.joblib")
    features = joblib.load(flavor_dir / "features.joblib")

    with open(flavor_dir / "threshold.json", "r") as f:
        threshold = json.load(f)["best_threshold"]
    
    # ---- Load inference pool (for testing)
    inference_pool_path = artifact_root / "inference_pool.json"
    if inference_pool_path.exists():
        with open(inference_pool_path, "r") as f:
            inference_pool = json.load(f)
    else:
        inference_pool = []

    return {
        "model": model,
        "imputer": imputer,
        "features": features,
        "threshold": threshold,
        "model_name": model_name,
        "inference_pool": load_inference_pool(),
        "model_version": model_version.version,
        "run_id": run_id,
    }