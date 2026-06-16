"""Pipeline for Project 04 — Diabetes Risk.

Functions
---------
train(X, y, seed=42)
    Fit a HistGradientBoostingClassifier wrapped in CalibratedClassifierCV.
predict_proba(model, X)
    Return P(diabetes=1) for each row.
calibration_curve(y_true, y_prob, n_bins=10)
    Reliability-diagram data — observed positive fraction in each predicted-probability bin.
threshold_sweep(y_true, y_prob, thresholds=None)
    Per-threshold sensitivity / specificity / F1 trade-off table.
lift_gain_table(y_true, y_prob, n_deciles=10)
    Cumulative gain and lift by targeted population decile.

This module is intentionally framework-light: scikit-learn + numpy + pandas only.
Not medical advice — educational use only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix


# ---------------------------------------------------------------- training ---

def train(X, y, seed: int = 42):
    """Train a calibrated HistGradientBoosting model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    seed : int, default 42

    Returns
    -------
    CalibratedClassifierCV
        A fitted, calibrated classifier with .predict_proba available.
    """
    np.random.seed(seed)
    base = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_depth=5,
        random_state=seed,
    )
    # cv=3 is safe down to ~30 rows.
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X, y)
    return clf


def predict_proba(model, X) -> np.ndarray:
    """Return the probability of the positive class (diabetes=1)."""
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------- calibration ---

def calibration_curve(y_true, y_prob, n_bins: int = 10):
    """Compute reliability-diagram data.

    Returns
    -------
    bin_centers : np.ndarray
    observed    : np.ndarray  (empirical fraction of positives per bin)
    expected    : np.ndarray  (same as bin_centers — perfect-calibration reference)
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)

    bin_centers, observed = [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        bin_centers.append(float(y_prob[mask].mean()))
        observed.append(float(y_true[mask].mean()))
    bin_centers = np.array(bin_centers)
    observed = np.array(observed)
    expected = bin_centers.copy()
    return bin_centers, observed, expected


# ---------------------------------------------------------- thresholding ---

def threshold_sweep(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Sweep decision threshold and report sens / spec / F1 at each.

    Returns a DataFrame with columns: threshold, sensitivity, specificity, f1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    rows = []
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) else 0.0
        rows.append({
            "threshold":   float(round(t, 4)),
            "sensitivity": float(round(sens, 4)),
            "specificity": float(round(spec, 4)),
            "f1":          float(round(f1, 4)),
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


# ------------------------------------------------------------- lift / gain ---

def lift_gain_table(y_true, y_prob, n_deciles: int = 10) -> pd.DataFrame:
    """Report cumulative gain and lift for top-risk population deciles.

    Rows answer questions like: if we target the top 10%, 20%, or 30% of
    people ranked by model risk, what percent of true diabetic patients do we
    identify, and how many times better is that than random targeting?
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if len(y_true) == 0:
        raise ValueError("lift_gain_table requires at least one row")
    if n_deciles < 1:
        raise ValueError("n_deciles must be >= 1")

    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    prob_sorted = y_prob[order]
    total_rows = len(y_sorted)
    total_positives = int(y_sorted.sum())

    rows = []
    previous_n = 0
    for decile in range(1, n_deciles + 1):
        target_n = int(np.ceil(total_rows * decile / n_deciles))
        target_n = min(total_rows, max(target_n, previous_n + 1))
        captured = int(y_sorted[:target_n].sum())
        population_pct = target_n / total_rows
        gain = captured / total_positives if total_positives else 0.0
        lift = gain / population_pct if population_pct else 0.0
        decile_positives = int(y_sorted[previous_n:target_n].sum())
        decile_size = target_n - previous_n
        decile_rate = decile_positives / decile_size if decile_size else 0.0
        min_score = float(prob_sorted[target_n - 1])
        rows.append({
            "decile": decile,
            "target_population_pct": round(population_pct * 100, 1),
            "targeted_people": target_n,
            "diabetic_patients_found": captured,
            "diabetic_patients_identified_pct": round(gain * 100, 1),
            "lift": round(lift, 2),
            "decile_positive_rate_pct": round(decile_rate * 100, 1),
            "min_score_in_group": round(min_score, 4),
        })
        previous_n = target_n

    return pd.DataFrame(rows)
