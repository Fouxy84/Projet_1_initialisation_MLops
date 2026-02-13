#api/load_data.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR = Path(r"C:/Users/coach/Desktop/datascientest/OpenClassrooms/Projects_MLops/Projet_1_initialisation_MLops")
DATA_PROC = BASE_DIR / "data" / "proceed"
DATA_PATH = DATA_PROC / "homecredit_features.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "lightgbm_model.joblib"
#DATA_PATH = "/content/drive/MyDrive/Colab Notebooks/Projet_1_initialisation_MLops/data/proceed/homecredit_features.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
print(df.shape)

train_df = df[df["TARGET"].notna()]
train_autoML = train_df.drop(columns=["SK_ID_CURR"])
X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
X = X.replace([np.inf, -np.inf], np.nan)
imputer = SimpleImputer(strategy="median") #remplace les valeurs manquantes par la médiane de chaque colonne
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
y = train_df["TARGET"]
X.columns = (
    X.columns
    .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
    .str.replace(r"_+", "_", regex=True)
)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

import lightgbm as lgb
model_lgb = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=200, num_leaves=31, random_state=42)
model_lgb.fit(X_train, y_train)

joblib.dump(
    {
        "model": model_lgb,
        "features": X.columns.tolist(),
        "imputer": imputer
    },
    MODEL_PATH
)

print(f"Model saved at: {MODEL_PATH}")