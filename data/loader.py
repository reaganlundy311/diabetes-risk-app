"""Data loaders for Project 04 — Diabetes Risk.

Two functions, mirroring the notebook template's "real-or-synthetic" rule:
- load_pima(path=None) — pull the public Pima Indians Diabetes CSV.
- load_synthetic(n=500) — same 8-feature schema, fully offline.

Schema (9 columns total):
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, dpf, age, outcome
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Public Brownlee mirror of the UCI Pima Indians Diabetes dataset (768 rows).
DEFAULT_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

COLS = [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "dpf", "age", "outcome",
]

ZERO_AS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


def impute_pima_missing_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Replace PIMA's impossible zero measurements with column medians.

    PIMA uses 0 to represent missing values in several measurement columns.
    Median imputation keeps all rows while avoiding fake measurements like
    glucose=0 or bmi=0.
    """
    cleaned = df.copy()
    for col in ZERO_AS_MISSING:
        cleaned[col] = cleaned[col].replace(0, np.nan)
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    cleaned.attrs["imputation"] = "median_imputed_zero_measurements"
    return cleaned


def load_pima(path: str | None = None, clean: bool = True) -> pd.DataFrame:
    """Load the real Pima Indians Diabetes dataset.

    Tries the public URL (or a local path if given). On any failure, falls back
    to ``load_synthetic`` so the rest of the code keeps running.

    Notes
    -----
    The Pima dataset encodes missing values as 0 for glucose / blood_pressure /
    skin_thickness / insulin / bmi. By default, this loader handles the Week-2
    cleanup by replacing those impossible zeros with each column's median.
    """
    src = path or DEFAULT_URL
    try:
        df = pd.read_csv(src, names=COLS)
        df.attrs["source"] = "real_pima"
        if clean:
            df = impute_pima_missing_zeros(df)
            df.attrs["source"] = "real_pima"
        print(f"[loader] Loaded Pima dataset from {src} — {len(df)} rows.")
        return df
    except Exception as e:
        print(f"[loader] Could not load Pima from {src} ({type(e).__name__}); "
              f"falling back to synthetic.")
        fallback = load_synthetic()
        fallback.attrs["source"] = "synthetic_fallback"
        return fallback


def load_synthetic(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthesize a small dataset with the exact same schema as Pima.

    The synthetic distribution is deliberately *similar but not identical* to
    Pima, so swapping in the real data will produce metric shifts worth
    investigating.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "pregnancies": rng.integers(0, 12, n),
        "glucose":        rng.normal(120, 32, n).clip(50, 220).round().astype(int),
        "blood_pressure": rng.normal(72, 12, n).clip(40, 110).round().astype(int),
        "skin_thickness": rng.normal(23, 9, n).clip(0, 60).round().astype(int),
        "insulin":        rng.normal(95, 80, n).clip(0, 600).round().astype(int),
        "bmi":            rng.normal(32, 6, n).clip(15, 55).round(1),
        "dpf":            rng.normal(0.45, 0.3, n).clip(0.05, 2.5).round(3),
        "age":            rng.integers(21, 70, n),
    })
    # Build a noisy logistic relationship so a real model can learn it.
    logit = (
        -8.5
        + 0.035 * df["glucose"]
        + 0.04  * df["bmi"]
        + 0.025 * df["age"]
        + 0.5   * df["dpf"]
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["outcome"] = (rng.random(n) < prob).astype(int)
    df.attrs["source"] = "synthetic"
    return df
