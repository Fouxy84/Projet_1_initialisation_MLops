# src/featuring.py

import pandas as pd
import numpy as np
import gc
from pathlib import Path


# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROC = BASE_DIR / "data" / "proceed"
DATA_PROC.mkdir(exist_ok=True, parents=True)

# =========================
# UTILS
# =========================

def reduce_memory(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df


def one_hot(df):
    cat_cols = df.select_dtypes(include="object").columns
    df = pd.get_dummies(
        df,
        columns=cat_cols,
        dummy_na=True
    )
    return df

# =========================
# APPLICATION
# =========================

def load_application(nrows=None):
    print("Loading application data...")
    train = pd.read_csv(
        DATA_RAW / "application_train.csv",
        nrows=nrows
    )
    test = pd.read_csv(
        DATA_RAW / "application_test.csv",
        nrows=nrows
    )
    df = pd.concat([train, test], axis=0)
    df = df[df["CODE_GENDER"] != "XNA"]
    # Binary encode
    for col in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col], _ = pd.factorize(df[col])
    # Fix anomaly
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    # Feature engineering
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df = one_hot(df)
    df = reduce_memory(df)
    return df


# =========================
# BUREAU
# =========================

def process_bureau(nrows=None):
    print("Processing bureau...")
    bureau = pd.read_csv(
        DATA_RAW / "bureau.csv",
        nrows=nrows
    )
    bureau = one_hot(bureau)
    agg = bureau.groupby("SK_ID_CURR").agg(
        {
            "AMT_CREDIT_SUM": ["mean", "sum"],
            "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
            "DAYS_CREDIT": ["mean"],
            "CREDIT_DAY_OVERDUE": ["mean", "max"]
        }
    )
    agg.columns = [
        "BURO_" + "_".join(col).upper()
        for col in agg.columns
    ]
    agg = reduce_memory(agg)
    del bureau
    gc.collect()
    return agg


# =========================
# PREVIOUS APPLICATIONS
# =========================

def process_previous(nrows=None):
    print("Processing previous applications...")
    prev = pd.read_csv(
        DATA_RAW / "previous_application.csv",
        nrows=nrows
    )
    prev = one_hot(prev)
    prev["APP_CREDIT_PERC"] = (
        prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    )
    agg = prev.groupby("SK_ID_CURR").agg(
        {
            "AMT_ANNUITY": ["mean"],
            "AMT_CREDIT": ["mean"],
            "APP_CREDIT_PERC": ["mean"]
        }
    )
    agg.columns = [
        "PREV_" + "_".join(col).upper()
        for col in agg.columns
    ]
    agg = reduce_memory(agg)
    del prev
    gc.collect()
    return agg


# =========================
# INSTALLMENTS
# =========================

def process_installments(nrows=None):
    print("Processing installments...")
    ins = pd.read_csv(
        DATA_RAW / "installments_payments.csv",
        nrows=nrows
    )
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    agg = ins.groupby("SK_ID_CURR").agg(
        {
            "PAYMENT_PERC": ["mean"],
            "AMT_PAYMENT": ["sum"],
            "AMT_INSTALMENT": ["sum"]
        }
    )
    agg.columns = [
        "INST_" + "_".join(col).upper()
        for col in agg.columns
    ]
    agg = reduce_memory(agg)
    del ins
    gc.collect()
    return agg


# =========================
# CREDIT CARD
# =========================

def process_credit_card(nrows=None):
    print("Processing credit card...")
    cc = pd.read_csv(
        DATA_RAW / "credit_card_balance.csv",
        nrows=nrows
    )
    cc = one_hot(cc)
    agg = cc.groupby("SK_ID_CURR").agg(
        {
            "AMT_BALANCE": ["mean", "max"],
            "AMT_CREDIT_LIMIT_ACTUAL": ["mean"],
            "SK_DPD": ["mean"]
        }
    )
    agg.columns = [
        "CC_" + "_".join(col).upper()
        for col in agg.columns
    ]
    agg = reduce_memory(agg)
    del cc
    gc.collect()
    return agg

# =========================
# BUILD DATASET
# =========================

def build_dataset(debug=False):
    print("Starting preprocessing...")
    nrows = 30000 if debug else None
    df = load_application(nrows)
    bureau = process_bureau(nrows)
    df = df.join(bureau, on="SK_ID_CURR")
    df = reduce_memory(df)
    del bureau
    gc.collect()
    prev = process_previous(nrows)
    df = df.join(prev, on="SK_ID_CURR")
    df = reduce_memory(df)
    del prev
    gc.collect()
    ins = process_installments(nrows)
    df = df.join(ins, on="SK_ID_CURR")
    df = reduce_memory(df)
    del ins
    gc.collect()
    cc = process_credit_card(nrows)
    df = df.join(cc, on="SK_ID_CURR")
    df = reduce_memory(df)
    del cc
    gc.collect()
    na_ratio = df.isna().mean()
    drop_cols = na_ratio[na_ratio > 0.8].index
    print("Dropping", len(drop_cols), "sparse columns")
    df = df.drop(columns=drop_cols)
    df.to_csv(DATA_PROC / "homecredit_features.csv", index=False)
    print("Final shape:", df.shape)
    print("Saved to:", DATA_PROC / "homecredit_features.csv")
    return df


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    build_dataset(debug=False)
