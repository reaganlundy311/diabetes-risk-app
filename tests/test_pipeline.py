"""Offline tests for the Diabetes Risk educational demo.

Runs in <30 s on a laptop. Synthetic data only, no network.

Run from the project root:
    pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Make the project root importable without pip-installing it.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from data.loader import load_synthetic, load_synthetic_brfss    # noqa: E402
from src.fairness import subgroup_metrics                       # noqa: E402
from src.lifestyle_coach import llm_compose, triggered_flags   # noqa: E402
from src.pipeline import (                                      # noqa: E402
    calibration_curve,
    feature_attribution,
    lift_gain_table,
    predict_proba,
    threshold_sweep,
    train,
)

from app.streamlit_app import largest_subgroup_gap              # noqa: E402

# ---- shared fixtures (trained once for the whole module) ----
_DF    = load_synthetic(n=400, seed=42)
_X     = _DF.drop(columns=["outcome"])
_Y     = _DF["outcome"].values
_MODEL = train(_X, _Y, seed=42)
_PROBS = predict_proba(_MODEL, _X)


# ================================================================ pipeline ===

def test_train_returns_calibrated_model():
    """train() returns something with predict_proba; probs are in [0, 1]."""
    assert hasattr(_MODEL, "predict_proba")
    probs = predict_proba(_MODEL, _X.iloc[:10])
    assert probs.shape == (10,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_calibration_curve_equal_length_arrays():
    """calibration_curve returns three arrays of equal, positive length."""
    centers, observed, expected = calibration_curve(_Y, _PROBS, n_bins=10)
    assert len(centers) == len(observed) == len(expected)
    assert len(centers) > 0
    assert np.all((observed >= 0.0) & (observed <= 1.0))


def test_threshold_sweep_monotonic_tradeoff():
    """Sensitivity is non-increasing; specificity is non-decreasing in threshold."""
    df = threshold_sweep(_Y, _PROBS)
    assert {"threshold", "sensitivity", "specificity", "f1"}.issubset(df.columns)
    assert len(df) >= 5
    sens_diffs = np.diff(df["sensitivity"].values)
    spec_diffs = np.diff(df["specificity"].values)
    assert (sens_diffs <= 1e-6).all(), f"sensitivity must be monotonic: {sens_diffs}"
    assert (spec_diffs >= -1e-6).all(), f"specificity must be monotonic: {spec_diffs}"


def test_lift_gain_table_has_ten_deciles():
    """Lift/gain table shows cumulative patient capture for 10 population deciles."""
    out = lift_gain_table(_Y, _PROBS, n_deciles=10)
    assert len(out) == 10
    assert {
        "target_population_pct",
        "diabetic_patients_identified_pct",
        "lift",
    }.issubset(out.columns)
    assert out["target_population_pct"].iloc[0] == 10.0
    assert out["target_population_pct"].iloc[-1] == 100.0
    assert out["diabetic_patients_identified_pct"].is_monotonic_increasing
    assert out["diabetic_patients_identified_pct"].iloc[-1] == 100.0
    expected_lift = (
        (out["diabetic_patients_found"] / _Y.sum())
        / (out["targeted_people"] / len(_Y))
    ).round(2)
    assert np.allclose(out["lift"], expected_lift)


def test_feature_attribution_returns_ranked_features():
    """Per-patient attribution reports one row per feature with directions."""
    out = feature_attribution(_MODEL, _X.iloc[0], _X)
    assert len(out) == _X.shape[1]
    assert {"feature", "risk_contribution", "direction"}.issubset(out.columns)
    assert set(out["feature"]) == set(_X.columns)


def test_synthetic_brfss_schema():
    """BRFSS fallback data has a usable survey schema."""
    df = load_synthetic_brfss(n=200, seed=42)
    expected = {
        "bmi", "age_group", "sex_female", "high_bp", "general_health",
        "physical_health_bad_days", "mental_health_bad_days",
        "checkup_within_year", "outcome",
    }
    assert expected.issubset(df.columns)
    assert set(df["outcome"].unique()).issubset({0, 1})


# =========================================================== lifestyle coach ===

def test_llm_compose_contains_disclaimer():
    """Every coach message must include 'not medical advice' (case-insensitive)."""
    patient = pd.Series({"bmi": 33.0, "glucose": 155, "blood_pressure": 90, "age": 50})
    msg = llm_compose(patient, risk=0.62)
    assert "not medical advice" in msg.lower(), msg
    flags = list(triggered_flags(patient))
    assert {"bmi", "glucose", "blood_pressure", "age"}.issubset(set(flags))


def test_llm_compose_handles_clean_patient():
    """A patient with no flags still gets a disclaimer-bounded message."""
    patient = {"bmi": 22.0, "glucose": 95, "blood_pressure": 70, "age": 25}
    msg = llm_compose(patient, risk=0.05)
    assert "not medical advice" in msg.lower()


def test_llm_compose_falls_back_without_api_key():
    """Requesting the live coach without a key still returns the safe template."""
    patient = {"bmi": 31.0, "glucose": 150, "blood_pressure": 88, "age": 48}
    msg = llm_compose(patient, risk=0.63, prefer_llm=True, api_key=None)
    assert "not medical advice" in msg.lower()
    assert "higher" in msg.lower()


# ================================================================= fairness ===

def test_subgroup_metrics_by_bmi_bucket():
    """subgroup_metrics returns one row per non-empty BMI bucket; n sums to total."""
    out = subgroup_metrics(_MODEL, _X, _Y, by="bmi_bucket")
    assert {"subgroup", "n", "prevalence", "auc", "sensitivity", "specificity"}.issubset(out.columns)
    assert len(out) >= 1
    assert out["n"].sum() == len(_Y)


def test_subgroup_metrics_by_age_bucket():
    """subgroup_metrics works with age_bucket too."""
    out = subgroup_metrics(_MODEL, _X, _Y, by="age_bucket")
    assert out["n"].sum() == len(_Y)


def test_largest_subgroup_gap_is_max_minus_min():
    table = pd.DataFrame({
        "subgroup": ["A", "B", "C"],
        "sensitivity": [0.25, 0.80, 0.50],
    })
    gap = largest_subgroup_gap(table)
    assert gap == {"gap": 0.55, "best": "B", "worst": "A"}
