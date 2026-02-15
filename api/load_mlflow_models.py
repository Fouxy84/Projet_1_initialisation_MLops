
import mlflow
import json
import joblib
from pathlib import Path

# ============================
# GLOBAL MLFLOW CONFIG (CRUCIAL)
# ============================
BASE_DIR = Path(__file__).resolve().parents[1]
MLFLOW_DB = BASE_DIR / "notebook" / "mlruns" / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")

# ============================
# LOAD MODEL + ARTIFACTS
# ============================
def load_model_bundle(model_name: str, stage: str | None = None, flavor: str = None):

    if stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"

    # ---- Load model (pyfunc wrapper)
    model = mlflow.pyfunc.load_model(model_uri)
    client = mlflow.tracking.MlflowClient()
    # ---- Resolve model version safely
    versions = (
        client.get_latest_versions(model_name, [stage])
        if stage
        else client.get_latest_versions(model_name)
    )

    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}'")

    model_version = versions[0]
    run_id = model_version.run_id

    # ---- Resolve artifact directory (portable)
    run = client.get_run(run_id)
    artifact_path = Path(run.info.artifact_uri.replace("file:///", ""))

    # ---- Load artifacts
    imputer = joblib.load(artifact_path / flavor / "imputer.joblib")
    features = joblib.load(artifact_path / flavor / "features.joblib")

    with open(artifact_path / flavor / "threshold.json") as f:
        threshold = json.load(f)["best_threshold"]

    return {
        "model": model,
        "imputer": imputer,
        "features": features,
        "threshold": threshold,
        "model_version": model_version.version,
        "run_id": run_id,
    }
