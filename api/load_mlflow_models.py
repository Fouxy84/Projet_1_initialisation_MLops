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

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")

def load_inference_pool():
    client = mlflow.tracking.MlflowClient()

    # récupérer le run "inference_pool"
    experiments = client.search_experiments()
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id for exp in experiments],
        filter_string="tags.mlflow.runName = 'inference_pool'",
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No MLflow run named 'inference_pool' found")

    run = runs[0]
    artifact_path = Path(run.info.artifact_uri.replace("file:///", ""))

    with open(artifact_path / "inference_pool.json") as f:
        return json.load(f)

# ======================================================
# LOAD MODEL + ASSOCIATED ARTIFACTS
# ======================================================
def load_model_bundle(
    model_name: str,
    stage: Optional[str] = None,
    flavor: str = "xgb",
):
   
    # ---- Resolve model URI
    model_uri = (
        f"models:/{model_name}/{stage}"
        if stage
        else f"models:/{model_name}/latest"
    )

    # ---- Load model (pyfunc wrapper)
    model = mlflow.pyfunc.load_model(model_uri)

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
    artifact_root = Path(run.info.artifact_uri.replace("file:///", ""))

    flavor_dir = artifact_root / flavor
    if not flavor_dir.exists():
        raise FileNotFoundError(
            f"Artifacts folder '{flavor}' not found for model '{model_name}'"
        )

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