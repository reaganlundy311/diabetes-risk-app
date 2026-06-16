"""Subgroup fairness metrics for Project 04.

`subgroup_metrics(model, X, y, by)` computes per-subgroup AUC, sensitivity,
specificity, and prevalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def _safe_auc(y_true, y_prob) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def bucket_bmi(bmi: float) -> str:
    if bmi is None or pd.isna(bmi):
        return "unknown"
    if bmi < 18.5:
        return "under (<18.5)"
    if bmi < 25:
        return "healthy (18.5-25)"
    if bmi < 30:
        return "overweight (25-30)"
    return "obese (30+)"


def bucket_age(age: float) -> str:
    if age is None or pd.isna(age):
        return "unknown"
    if age < 30:
        return "<30"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def subgroup_metrics(model, X: pd.DataFrame, y, by, threshold: float = 0.5) -> pd.DataFrame:
    """Compute per-subgroup metrics.

    Parameters
    ----------
    model     : classifier with .predict_proba
    X         : pd.DataFrame — feature matrix
    y         : array-like — ground-truth labels
    by        : str (column name or 'bmi_bucket' / 'age_bucket') or array-like
    threshold : float — decision threshold

    Returns
    -------
    pd.DataFrame — columns: subgroup, n, prevalence, auc, sensitivity, specificity
    """
    y = np.asarray(y).astype(int)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    if isinstance(by, str):
        if by == "bmi_bucket" and "bmi" in X.columns:
            groups = X["bmi"].map(bucket_bmi).values
        elif by == "age_bucket" and "age" in X.columns:
            groups = X["age"].map(bucket_age).values
        elif by in X.columns:
            groups = X[by].values
        else:
            raise KeyError(f"Subgroup column {by!r} not found in X")
    else:
        groups = np.asarray(by)
        if len(groups) != len(y):
            raise ValueError("`by` array length must match y")

    rows = []
    for g in pd.unique(pd.Series(groups)):
        mask = groups == g
        if mask.sum() == 0:
            continue
        yi, pi, pri = y[mask], preds[mask], probs[mask]
        cm = confusion_matrix(yi, pi, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        auc  = _safe_auc(yi, pri)
        rows.append({
            "subgroup":    str(g),
            "n":           int(mask.sum()),
            "prevalence":  float(round(yi.mean(), 4)),
            "auc":         float(round(auc, 4)) if not np.isnan(auc) else float("nan"),
            "sensitivity": float(round(sens, 4)) if not np.isnan(sens) else float("nan"),
            "specificity": float(round(spec, 4)) if not np.isnan(spec) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("subgroup").reset_index(drop=True)
