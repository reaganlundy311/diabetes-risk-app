"""Data loaders for Project 04 — Diabetes Risk.

Two functions, mirroring the notebook template's "real-or-synthetic" rule:
- load_pima(path=None) — pull the public Pima Indians Diabetes CSV.
- load_synthetic(n=500) — same 8-feature schema, fully offline.
- load_brfss(...) — pull CDC BRFSS and map survey columns to a model schema.

Schema (9 columns total):
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, dpf, age, outcome
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from urllib.request import urlopen

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

# CDC BRFSS 2022 XPT archive. The raw file is large, so callers can sample rows.
BRFSS_URL = "https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip"
BRFSS_FEATURES = [
    "bmi",
    "age_group",
    "sex_female",
    "high_bp",
    "general_health",
    "physical_health_bad_days",
    "mental_health_bad_days",
    "checkup_within_year",
]


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


def _clean_brfss_days(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.where(out.between(1, 30), 0)
    return out.fillna(0)


def transform_brfss(raw: pd.DataFrame) -> pd.DataFrame:
    """Map CDC BRFSS survey columns into a compact diabetes-risk schema.

    Outcome is diabetes yes/no from `_DIABETE4`: 1 means diabetes, 2 means no.
    Prediabetes, pregnancy-related diabetes, refusals, and unknowns are removed.
    """
    needed = ["_DIABETE4", "_BMI5", "_AGEG5YR", "SEXVAR", "BPHIGH6", "GENHLTH", "PHYSHLTH", "MENTHLTH", "CHECKUP1"]
    missing = [col for col in needed if col not in raw.columns]
    if missing:
        raise KeyError(f"BRFSS columns missing: {missing}")

    df = raw[needed].copy()
    df = df[df["_DIABETE4"].isin([1, 2])]
    df["outcome"] = (df["_DIABETE4"] == 1).astype(int)
    df["bmi"] = pd.to_numeric(df["_BMI5"], errors="coerce") / 100.0
    df["age_group"] = pd.to_numeric(df["_AGEG5YR"], errors="coerce")
    df["sex_female"] = (pd.to_numeric(df["SEXVAR"], errors="coerce") == 2).astype(int)
    df["high_bp"] = (pd.to_numeric(df["BPHIGH6"], errors="coerce") == 1).astype(int)
    df["general_health"] = pd.to_numeric(df["GENHLTH"], errors="coerce")
    df["physical_health_bad_days"] = _clean_brfss_days(df["PHYSHLTH"])
    df["mental_health_bad_days"] = _clean_brfss_days(df["MENTHLTH"])
    df["checkup_within_year"] = (pd.to_numeric(df["CHECKUP1"], errors="coerce") == 1).astype(int)
    out = df[BRFSS_FEATURES + ["outcome"]].dropna()
    out = out[
        out["bmi"].between(10, 80)
        & out["age_group"].between(1, 13)
        & out["general_health"].between(1, 5)
    ].reset_index(drop=True)
    out.attrs["source"] = "cdc_brfss"
    return out


def load_brfss(
    path_or_url: str | None = None,
    sample_n: int = 50000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load and transform CDC BRFSS.

    The public BRFSS file is large. For Streamlit responsiveness, this loader
    returns a reproducible sample when more than `sample_n` rows are available.
    If the CDC file is unavailable, it falls back to synthetic BRFSS-shaped data.
    """
    src = path_or_url or BRFSS_URL
    try:
        if str(src).lower().endswith(".zip"):
            payload = urlopen(src, timeout=30).read() if str(src).startswith("http") else open(src, "rb").read()
            with zipfile.ZipFile(BytesIO(payload)) as zf:
                xpt_name = next(name for name in zf.namelist() if name.lower().endswith(".xpt"))
                with zf.open(xpt_name) as fh:
                    raw = pd.read_sas(fh, format="xport")
        else:
            raw = pd.read_sas(src, format="xport")
        df = transform_brfss(raw)
        if sample_n and len(df) > sample_n:
            df = df.sample(sample_n, random_state=seed).reset_index(drop=True)
            df.attrs["source"] = "cdc_brfss_sample"
        print(f"[loader] Loaded CDC BRFSS from {src} — {len(df)} rows after cleaning.")
        return df
    except Exception as e:
        print(f"[loader] Could not load CDC BRFSS ({type(e).__name__}); falling back to synthetic BRFSS.")
        fallback = load_synthetic_brfss(n=min(sample_n, 50000), seed=seed)
        fallback.attrs["source"] = "synthetic_brfss_fallback"
        return fallback


def load_synthetic_brfss(n: int = 50000, seed: int = 42) -> pd.DataFrame:
    """Generate a BRFSS-shaped survey dataset for offline tests and fallback."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "bmi": rng.normal(29, 7, n).clip(15, 65).round(1),
        "age_group": rng.integers(1, 14, n),
        "sex_female": rng.integers(0, 2, n),
        "high_bp": rng.binomial(1, 0.34, n),
        "general_health": rng.choice([1, 2, 3, 4, 5], n, p=[0.18, 0.32, 0.30, 0.14, 0.06]),
        "physical_health_bad_days": rng.poisson(3, n).clip(0, 30),
        "mental_health_bad_days": rng.poisson(4, n).clip(0, 30),
        "checkup_within_year": rng.binomial(1, 0.78, n),
    })
    logit = (
        -4.6
        + 0.055 * df["bmi"]
        + 0.18 * df["age_group"]
        + 0.95 * df["high_bp"]
        + 0.28 * df["general_health"]
        + 0.015 * df["physical_health_bad_days"]
        + 0.004 * df["mental_health_bad_days"]
    )
    prob = 1 / (1 + np.exp(-logit))
    df["outcome"] = (rng.random(n) < prob).astype(int)
    df.attrs["source"] = "synthetic_brfss"
    return df
