import joblib
from api.dummy_model import DummyModel

# Dummy model for xgb
model_xgb = DummyModel()
joblib.dump(model_xgb, "artifacts/xgb/model.joblib")

# Dummy model for lgb
model_lgb = DummyModel()
joblib.dump(model_lgb, "artifacts/lgb/model.joblib")

print("Dummy models saved")