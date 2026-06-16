"""main.py — Project 04: Diabetes Risk Assessment
SciEncephalon AI · Summer Intern Series 2026

Run this file directly in VS Code (F5 or `python main.py`) to:
  1. Load real PIMA data with median imputation for impossible zeros.
  2. Train a calibrated HistGradientBoosting model.
  3. Print calibration curve data.
  4. Print threshold sweep (sensitivity / specificity / F1 table).
  5. Print 10-decile lift/gain targeting analysis.
  6. Run a sample lifestyle-coach message.
  7. Run a fairness audit by BMI bucket and age bucket.

NOT MEDICAL ADVICE — educational use only.
"""

from __future__ import annotations

import sys
import os

# ── make sibling packages importable when run from any CWD ──────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from data.loader import load_pima, load_synthetic
from src.pipeline import train, predict_proba, calibration_curve, threshold_sweep, lift_gain_table
from src.lifestyle_coach import llm_compose
from src.fairness import subgroup_metrics


BANNER = "=" * 65


def section(title: str) -> None:
    print(f"\n{BANNER}\n  {title}\n{BANNER}")


def evaluate_dataset(df: pd.DataFrame, label: str) -> dict:
    X = df.drop(columns=["outcome"])
    y = df["outcome"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model = train(X_train, y_train, seed=42)
    probs = predict_proba(model, X_test)
    preds = (probs >= 0.50).astype(int)
    tn = int(((preds == 0) & (y_test == 0)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "dataset": label,
        "auc": roc_auc_score(y_test, probs),
        "accuracy": accuracy_score(y_test, preds),
        "sensitivity": recall_score(y_test, preds, zero_division=0),
        "specificity": specificity,
        "brier": brier_score_loss(y_test, probs),
    }


# ─────────────────────────────────────────────────────────────── 1. Data ────

section("1 / 6  Loading data")

# Try real PIMA first; falls back to synthetic if no network.
df = load_pima(clean=True)
source = df.attrs.get("source", "unknown")
print(f"Dataset shape: {df.shape}  |  Outcome rate: {df['outcome'].mean():.1%}")
print(f"Dataset source: {source}")
print("Week-2 imputation: impossible PIMA zeros are replaced with column medians.")

section("Week 2  Synthetic baseline vs real PIMA")
comparison = pd.DataFrame([
    evaluate_dataset(load_synthetic(n=600, seed=42), "Synthetic baseline"),
    evaluate_dataset(df, "Real PIMA, median-imputed" if source != "synthetic_fallback" else "Synthetic fallback"),
])
print(comparison.round(3).to_string(index=False))

X_all = df.drop(columns=["outcome"])
y_all = df["outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.25, stratify=y_all, random_state=42
)
print(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}")


# ─────────────────────────────────────────────────────────── 2. Training ────

section("2 / 6  Training calibrated model")

model = train(X_train, y_train, seed=42)
print("Model trained: HistGradientBoosting + isotonic calibration (cv=3)")

probs_test = predict_proba(model, X_test)
print(f"Test-set predicted-prob range: [{probs_test.min():.3f}, {probs_test.max():.3f}]")


# ──────────────────────────────────────────────── 3. Calibration curve ──────

section("3 / 6  Calibration curve  (predicted → observed fractions)")

centers, observed, _ = calibration_curve(y_test, probs_test, n_bins=10)
cal_df = pd.DataFrame({"predicted_prob": centers.round(3), "observed_frac": observed.round(3)})
print(cal_df.to_string(index=False))
print("\nInterpretation: A well-calibrated model has observed ≈ predicted.")
print("Large gaps mean the raw score needs re-scaling before acting on it.")


# ──────────────────────────────────────────────── 4. Threshold sweep ────────

section("4 / 6  Threshold sweep  (sensitivity / specificity / F1)")

sweep = threshold_sweep(y_test, probs_test)
print(sweep.to_string(index=False))
print(
    "\nKey insight: The 0.5 default is rarely right in healthcare.\n"
    "  Lower threshold  → higher sensitivity (catch more cases, more false alarms).\n"
    "  Higher threshold → higher specificity (fewer false alarms, miss more cases)."
)


# ─────────────────────────────────────────────── 5. Lift / gain ─────────────

section("5 / 7  Lift / gain targeting analysis  (10 deciles)")

lift_gain = lift_gain_table(y_test, probs_test, n_deciles=10)
print(lift_gain[[
    "target_population_pct",
    "diabetic_patients_identified_pct",
    "lift",
    "diabetic_patients_found",
    "targeted_people",
]].to_string(index=False))
print(
    "\nInterpretation: target the highest-risk patients first. "
    "For example, the 20% row says what percent of diabetic patients are found "
    "by targeting only the top-risk 20%, and lift shows how many times better "
    "that is than random targeting."
)


# ─────────────────────────────────────────────── 6. Lifestyle coach ─────────

section("6 / 7  Lifestyle-coach sample  (templated stub)")

sample_patient = {
    "bmi":            33.5,
    "glucose":        158,
    "blood_pressure": 88,
    "age":            52,
    "pregnancies":    2,
    "skin_thickness": 25,
    "insulin":        120,
    "dpf":            0.62,
}
# Estimate risk for this patient (reshape to single-row DataFrame)
sample_row = pd.DataFrame([sample_patient])[X_all.columns]
risk_estimate = float(predict_proba(model, sample_row)[0])

print(f"\nSample patient: {sample_patient}")
print(f"Estimated risk: {risk_estimate*100:.1f}%\n")
print(llm_compose(sample_patient, risk=risk_estimate))


# ──────────────────────────────────────────────── 7. Fairness audit ─────────

section("7 / 7  Fairness audit  (subgroup metrics)")

print("\n── By BMI bucket ──")
bmi_table = subgroup_metrics(model, X_test, y_test, by="bmi_bucket")
print(bmi_table.to_string(index=False))

print("\n── By age bucket ──")
age_table = subgroup_metrics(model, X_test, y_test, by="age_bucket")
print(age_table.to_string(index=False))

print(
    "\nLook for gaps in AUC or sensitivity across subgroups.\n"
    "Large gaps may indicate the model is less reliable for certain populations."
)

print(f"\n{BANNER}")
print("  Done!  Run the Streamlit app for an interactive demo:")
print("  streamlit run app/streamlit_app.py")
print(f"{BANNER}\n")
