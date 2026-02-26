# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 23:30:56 2026

@author: coach
"""

import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path

# ======================================================
# Ajouter src au PYTHONPATH
# ======================================================
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from featuring import (
    one_hot_encoder,
    load_application,
    process_bureau,
    process_previous,
    process_pos,
    process_installments,
    process_credit_card,
)

# ============================================
# One Hot Encoder
# ============================================

def test_one_hot_encoder_creates_columns():
    df = pd.DataFrame({
        "A": ["x", "y", "x"],
        "B": [1, 2, 3]
    })

    df_encoded = one_hot_encoder(df, max_categories=5)

    assert isinstance(df_encoded, pd.DataFrame)
    assert any(col.startswith("A_") for col in df_encoded.columns)


def test_one_hot_encoder_no_nan():
    df = pd.DataFrame({
        "A": ["x", None, "y"]
    })

    df_encoded = one_hot_encoder(df)

    assert df_encoded.isna().sum().sum() == 0


# ============================================
# Application
# ============================================

def test_load_application_returns_dataframe():
    df = load_application()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_application_features_created():
    df = load_application()

    expected_features = [
        "DAYS_EMPLOYED_PERC",
        "INCOME_CREDIT_PERC",
        "INCOME_PER_PERSON",
        "ANNUITY_INCOME_PERC",
        "PAYMENT_RATE",
    ]

    for feature in expected_features:
        assert feature in df.columns


# ============================================
# Bureau
# ============================================

def test_process_bureau_index():
    df = process_bureau()

    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "SK_ID_CURR"


# ============================================
# Previous Applications
# ============================================

def test_process_previous_columns():
    df = process_previous()

    assert isinstance(df, pd.DataFrame)
    assert any(col.startswith("PREV_") for col in df.columns)


# ============================================
# POS CASH
# ============================================

def test_process_pos_columns():
    df = process_pos()

    assert isinstance(df, pd.DataFrame)
    assert any(col.startswith("POS_") for col in df.columns)


# ============================================
# Installments
# ============================================

def test_process_installments_columns():
    df = process_installments()

    assert isinstance(df, pd.DataFrame)
    assert any(col.startswith("INST_") for col in df.columns)


# ============================================
# Credit Card
# ============================================

def test_process_credit_card_columns():
    df = process_credit_card()

    assert isinstance(df, pd.DataFrame)
    assert any(col.startswith("CC_") for col in df.columns)