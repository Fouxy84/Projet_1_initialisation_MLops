import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
# ============================================================
# Utils métier
# ============================================================
def business_cost(y_true, y_pred, fn_cost=10, fp_cost=1):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * fn_cost + fp * fp_cost


def find_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.05, 0.95, 0.05)
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        costs.append(business_cost(y_true, y_pred))
    best_idx = np.argmin(costs)
    return thresholds[best_idx]