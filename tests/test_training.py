# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 23:36:28 2026

@author: coach
"""

# tests/test_training.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ============================================================
# Import des fonctions métier
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "api"))

from load_data import business_cost, find_best_threshold


# ============================================================
# Fixtures données (rapides & contrôlées)
# ============================================================

@pytest.fixture(scope="session")
def raw_dataset():
    data_path = BASE_DIR / "data" / "proceed" / "homecredit_features.csv"
    df = pd.read_csv(data_path, low_memory=False)

    # on garde un petit échantillon pour les tests
    return df.sample(n=5000, random_state=42)


@pytest.fixture
def prepared_data(raw_dataset):
    df = raw_dataset

    train_df = df[df["TARGET"].notna()]

    X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = train_df["TARGET"]

    # nettoyage colonnes
    X.columns = (
        X.columns
        .str.replace(r"[^0-9a-zA-Z_]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )

    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float64)

    nan_cols = X.columns[X.isna().all()]
    X = X.drop(columns=nan_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test


# ============================================================
# Tests sur les données
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
# Tests fonctions métier
# ============================================================

def test_business_cost_basic():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    cost = business_cost(y_true, y_pred, fn_cost=10, fp_cost=1)

    assert cost == 11


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