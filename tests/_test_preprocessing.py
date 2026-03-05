
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# ======================================================
# Import module
# ======================================================

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

import featuring


# ======================================================
# Fixtures : DataFrames mockés MINIMAUX
# ======================================================

@pytest.fixture
def application_df():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "CODE_GENDER": ["M", "F"],
        "FLAG_OWN_CAR": ["Y", "N"],
        "FLAG_OWN_REALTY": ["Y", "Y"],
        "DAYS_EMPLOYED": [100, 365243],
        "DAYS_BIRTH": [-10000, -12000],
        "AMT_INCOME_TOTAL": [100000, 120000],
        "AMT_CREDIT": [200000, 250000],
        "CNT_FAM_MEMBERS": [2, 3],
        "AMT_ANNUITY": [10000, 12000],
    })


@pytest.fixture
def bureau_df():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_BUREAU": [10, 11, 20],
        "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
    })


@pytest.fixture
def bureau_balance_df():
    return pd.DataFrame({
        "SK_ID_BUREAU": [10, 10, 20],
        "MONTHS_BALANCE": [-1, -2, -1],
    })


@pytest.fixture
def previous_df():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "AMT_APPLICATION": [100000, 120000],
        "AMT_CREDIT": [110000, 130000],
    })


@pytest.fixture
def installments_df():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "AMT_PAYMENT": [1000, 2000],
        "AMT_INSTALMENT": [1200, 2200],
    })


@pytest.fixture
def pos_df():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "MONTHS_BALANCE": [-1, -2, -1],
    })


@pytest.fixture
def credit_card_df():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "SK_ID_PREV": [100, 200],
        "AMT_BALANCE": [5000, 6000],
    })


# ======================================================
# Monkeypatch pd.read_csv
# ======================================================

@pytest.fixture
def mock_read_csv(
    monkeypatch,
    application_df,
    bureau_df,
    bureau_balance_df,
    previous_df,
    installments_df,
    pos_df,
    credit_card_df,
):
    def fake_read_csv(path, *args, **kwargs):
        name = Path(path).name

        if "application_train" in name:
            return application_df.copy()
        if "bureau_balance" in name:
            return bureau_balance_df.copy()
        if "bureau.csv" in name:
            return bureau_df.copy()
        if "previous_application" in name:
            return previous_df.copy()
        if "installments_payments" in name:
            return installments_df.copy()
        if "POS_CASH_balance" in name:
            return pos_df.copy()
        if "credit_card_balance" in name:
            return credit_card_df.copy()

        raise ValueError(f"Unexpected file: {name}")

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)


# ======================================================
# Tests one_hot_encoder
# ======================================================

def test_one_hot_encoder_basic():
    df = pd.DataFrame({"cat": ["a", "b"], "num": [1, 2]})
    out = featuring.one_hot_encoder(df)

    assert isinstance(out, pd.DataFrame)
    assert out.isna().sum().sum() == 0


# ======================================================
# Tests load_application
# ======================================================

def test_load_application_fast(mock_read_csv):
    df = featuring.load_application()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    engineered = {
        "DAYS_EMPLOYED_PERC",
        "INCOME_CREDIT_PERC",
        "INCOME_PER_PERSON",
        "ANNUITY_INCOME_PERC",
        "PAYMENT_RATE",
    }

    assert engineered.issubset(df.columns)


# ======================================================
# Tests process_* (rapides & mockés)
# ======================================================

@pytest.mark.parametrize(
    "func,prefix",
    [
        (featuring.process_bureau, "BURO_"),
        (featuring.process_previous, "PREV_"),
        (featuring.process_pos, "POS_"),
        (featuring.process_installments, "INST_"),
        (featuring.process_credit_card, "CC_"),
    ],
)
def test_process_functions_fast(func, prefix, mock_read_csv):
    df = func()

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert any(col.startswith(prefix) for col in df.columns)