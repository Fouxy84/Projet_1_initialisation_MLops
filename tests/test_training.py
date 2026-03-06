
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


# ============================================================
# Import des fonctions métier
# ============================================================

#BASE_DIR = Path(__file__).resolve().parents[1]
#sys.path.append(str(BASE_DIR / "api"))
from api.utilis import business_cost, find_best_threshold

# ============================================================
# Fixtures données (rapides & contrôlées)
# ============================================================

@pytest.fixture(scope="session")
def raw_dataset():
    """
    Dataset synthétique pour tests rapides et reproductibles
    (évite dépendance au dataset réel).
    """

    rng = np.random.default_rng(42)

    n_samples = 200
    n_features = 25

    data = rng.normal(size=(n_samples, n_features))

    df = pd.DataFrame(
        data,
        columns=[f"feature_{i}" for i in range(n_features)]
    )

    df["TARGET"] = rng.integers(0, 2, size=n_samples)
    df["SK_ID_CURR"] = range(n_samples)

    return df

@pytest.fixture(scope="session")
def prepared_data(raw_dataset):
    """
    Prépare les données (nettoyage, split, imputation)
    une seule fois pour tous les tests.
    """
    df = raw_dataset.copy()

    train_df = df[df["TARGET"].notna()]

    X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = train_df["TARGET"]

    # Nettoyage des noms de colonnes
    X.columns = (
        X.columns
        .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )

    # Gestion des valeurs infinies
    X = X.replace([np.inf, -np.inf], np.nan)

    # Suppression des colonnes entièrement NaN
    nan_cols = X.columns[X.isna().all()]
    X = X.drop(columns=nan_cols)

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Imputation
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test


# ============================================================
# Tests sur les données préparées
# ============================================================

def test_dataset_loaded(raw_dataset):
    assert isinstance(raw_dataset, pd.DataFrame)
    assert "TARGET" in raw_dataset.columns
    assert "SK_ID_CURR" in raw_dataset.columns


def test_train_test_split(prepared_data):
    X_train, X_test, y_train, y_test = prepared_data

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_no_nan_after_imputation(prepared_data):
    X_train, X_test, _, _ = prepared_data

    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


def test_features_consistency(prepared_data):
    X_train, X_test, _, _ = prepared_data

    assert list(X_train.columns) == list(X_test.columns)
    assert len(X_train.columns) > 0


# ============================================================
# Tests des fonctions métier
# ============================================================

def test_business_cost_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    cost = business_cost(y_true, y_pred, fn_cost=10, fp_cost=1)

    # 1 FN (10) + 1 FP (1) = 11
    assert cost == 11


def test_business_cost_zero():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    assert business_cost(y_true, y_pred) == 0


def test_find_best_threshold_valid_range():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.3, 0.7, 0.9])

    threshold = find_best_threshold(y_true, y_proba)

    assert 0.05 <= threshold <= 0.95


def test_find_best_threshold_type():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.4, 0.9])

    threshold = find_best_threshold(y_true, y_proba)

    assert isinstance(threshold, float)