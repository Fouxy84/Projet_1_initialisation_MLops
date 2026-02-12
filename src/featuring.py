import pandas as pd
import numpy as np
import gc
from pathlib import Path


# ============================================
# Paths
# ============================================

BASE_DIR = Path(r"C:/Users/coach/Desktop/datascientest/OpenClassrooms/Projects_MLops/Projet_1_initialisation_MLops")
RAW = BASE_DIR / "data" / "raw"
PROC = BASE_DIR / "data" / "proceed"

# ============================================
# One Hot Encoder
# ============================================

def one_hot_encoder(df, max_categories=10, nan_as_category=True):
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    small_cat_cols = [
        c for c in cat_cols
        if df[c].nunique() <= max_categories
    ]
    print(f"OHE columns: {len(small_cat_cols)} / {len(cat_cols)}")

    df = pd.get_dummies(
        df,
        columns=small_cat_cols,
        dummy_na=nan_as_category
    )

    # Label encode large categories
    for col in cat_cols:
        if col not in small_cat_cols:
            df[col], _ = pd.factorize(df[col])

    return df


# ============================================
# Application
# ============================================

def load_application():
    train = pd.read_csv(RAW / "application_train.csv")
    test = pd.read_csv(RAW / "application_test.csv")
    print("Train:", train.shape)
    print("Test :", test.shape)
    df = pd.concat([train, test], axis=0)
    # Remove XNA
    df = df[df["CODE_GENDER"] != "XNA"]

    # Binary encoding
    for col in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[col], _ = pd.factorize(df[col])
    # Anomaly
    #df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    df.replace({"DAYS_EMPLOYED": {365243: np.nan}}, inplace=True)
    # Features
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df = one_hot_encoder(df)
    return df


# ============================================
# Bureau
# ============================================

def process_bureau():
    bureau = pd.read_csv(RAW / "bureau.csv")
    bb = pd.read_csv(RAW / "bureau_balance.csv")
    bb = one_hot_encoder(bb)
    bureau = one_hot_encoder(bureau)
    # BB agg
    bb_agg = bb.groupby("SK_ID_BUREAU").agg({
        "MONTHS_BALANCE": ["min","max","size"]
    })
    bb_agg.columns = [
        "BB_" + "_".join(col) for col in bb_agg.columns
    ]
    bureau = bureau.join(bb_agg, on="SK_ID_BUREAU")
    bureau.drop("SK_ID_BUREAU", axis=1, inplace=True)
    # Main agg
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(["min","max","mean","sum"])
    bureau_agg.columns = [
        "BURO_" + "_".join(col) for col in bureau_agg.columns
    ]
    del bureau, bb, bb_agg
    gc.collect()
    return bureau_agg


# ============================================
# Previous Applications
# ============================================

def process_previous():
    prev = pd.read_csv(RAW / "previous_application.csv")
    prev = one_hot_encoder(prev)
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    prev_agg = prev.groupby("SK_ID_CURR").agg(["min","max","mean","sum"])
    prev_agg.columns = [
        "PREV_" + "_".join(col) for col in prev_agg.columns
    ]
    del prev
    gc.collect()
    return prev_agg


# ============================================
# POS CASH
# ============================================

def process_pos():
    pos = pd.read_csv(RAW / "POS_CASH_balance.csv")
    pos = one_hot_encoder(pos)
    pos_agg = pos.groupby("SK_ID_CURR").agg(["max","mean","size"])
    pos_agg.columns = [
        "POS_" + "_".join(col) for col in pos_agg.columns
    ]
    del pos
    gc.collect()
    return pos_agg

# ============================================
# Installments
# ============================================

def process_installments():
    ins = pd.read_csv(RAW / "installments_payments.csv")
    ins = one_hot_encoder(ins)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins_agg = ins.groupby("SK_ID_CURR").agg(["min","max","mean","sum"])
    ins_agg.columns = [
        "INST_" + "_".join(col) for col in ins_agg.columns
    ]
    del ins
    gc.collect()
    return ins_agg


# ============================================
# Credit Card
# ============================================

def process_credit_card():
    cc = pd.read_csv(RAW / "credit_card_balance.csv")
    cc = one_hot_encoder(cc)
    cc.drop("SK_ID_PREV", axis=1, inplace=True)
    cc_agg = cc.groupby("SK_ID_CURR").agg(["min","max","mean","sum"])
    cc_agg.columns = [
        "CC_" + "_".join(col) for col in cc_agg.columns
    ]
    del cc
    gc.collect()
    return cc_agg


# ============================================
# Build Dataset
# ============================================

def build_dataset():
    print("Loading application...")
    df = load_application()
    print("Processing bureau...")
    bureau = process_bureau()
    df = df.join(bureau, on="SK_ID_CURR")
    print("Processing previous...")
    prev = process_previous()
    df = df.join(prev, on="SK_ID_CURR")
    print("Processing POS...")
    pos = process_pos()
    df = df.join(pos, on="SK_ID_CURR")
    print("Processing installments...")
    ins = process_installments()
    df = df.join(ins, on="SK_ID_CURR")
    print("Processing credit card...")
    cc = process_credit_card()
    df = df.join(cc, on="SK_ID_CURR")
    print("Final shape:", df.shape)
    df = df.astype(np.float32)
    # Save
    output = PROC / "homecredit_features.csv"
    df.to_csv(output, index=False)
    print("Saved to:", output)


# ============================================
# Main
# ============================================

if __name__ == "__main__":
    build_dataset()
