"""Streamlit app for Project 04 — Diabetes Risk.

Run from the project root:
    streamlit run app/streamlit_app.py

NOT MEDICAL ADVICE — educational use only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, brier_score_loss, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ── make sibling packages importable ─────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from data.loader import load_pima, load_synthetic          # noqa: E402
from src.lifestyle_coach import llm_compose                # noqa: E402
from src.pipeline import (                                 # noqa: E402
    calibration_curve,
    lift_gain_table,
    predict_proba,
    threshold_sweep,
    train,
)
from src.fairness import subgroup_metrics                  # noqa: E402

DISCLAIMER = (
    "**Not medical advice.** This demo is an educational exercise. It is *not* a "
    "diagnostic tool. Talk to a qualified healthcare provider for any medical decision."
)

def get_secret(name: str, default: str | None = None) -> str | None:
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


# ─────────────────────────────────────── model bootstrap (cached per session) ─

def evaluate_dataset(df: pd.DataFrame, label: str) -> dict:
    X = df.drop(columns=["outcome"])
    y = df["outcome"].values
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    model = train(Xtr, ytr, seed=42)
    probs_te = predict_proba(model, Xte)
    preds = (probs_te >= 0.50).astype(int)
    tn = int(((preds == 0) & (yte == 0)).sum())
    fp = int(((preds == 1) & (yte == 0)).sum())
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "label": label,
        "df": df,
        "model": model,
        "feature_cols": list(X.columns),
        "X_test": Xte,
        "y_test": yte,
        "probs_test": probs_te,
        "auc": float(roc_auc_score(yte, probs_te)),
        "accuracy": float(accuracy_score(yte, preds)),
        "sensitivity": float(recall_score(yte, preds, zero_division=0)),
        "specificity": float(specificity),
        "brier": float(brier_score_loss(yte, probs_te)),
    }


@st.cache_resource(show_spinner="Training calibrated model on real PIMA data…")
def get_model_and_eval():
    pima_eval = evaluate_dataset(load_pima(clean=True), "Real PIMA, median-imputed")
    synthetic_eval = evaluate_dataset(load_synthetic(n=600, seed=42), "Synthetic baseline")

    model = pima_eval["model"]
    Xte = pima_eval["X_test"]
    yte = pima_eval["y_test"]
    probs_te = pima_eval["probs_test"]
    sweep = threshold_sweep(yte, probs_te)
    lift_gain = lift_gain_table(yte, probs_te, n_deciles=10)
    centers, observed, expected = calibration_curve(yte, probs_te, n_bins=10)
    comparison = pd.DataFrame([
        {
            "dataset": synthetic_eval["label"],
            "auc": synthetic_eval["auc"],
            "accuracy": synthetic_eval["accuracy"],
            "sensitivity": synthetic_eval["sensitivity"],
            "specificity": synthetic_eval["specificity"],
            "brier": synthetic_eval["brier"],
        },
        {
            "dataset": pima_eval["label"],
            "auc": pima_eval["auc"],
            "accuracy": pima_eval["accuracy"],
            "sensitivity": pima_eval["sensitivity"],
            "specificity": pima_eval["specificity"],
            "brier": pima_eval["brier"],
        },
    ])
    return {
        "model":        model,
        "feature_cols": pima_eval["feature_cols"],
        "X_test":       Xte,
        "y_test":       yte,
        "probs_test":   probs_te,
        "sweep":        sweep,
        "lift_gain":    lift_gain,
        "calibration":  (centers, observed, expected),
        "fairness_bmi": subgroup_metrics(model, Xte, yte, by="bmi_bucket"),
        "fairness_age": subgroup_metrics(model, Xte, yte, by="age_bucket"),
        "comparison":   comparison,
        "data_source":  pima_eval["df"].attrs.get("source", "unknown"),
    }


# ──────────────────────────────────────────────────────────── chart helpers ─

def calibration_chart_data(centers, observed) -> pd.DataFrame:
    return pd.DataFrame({
        "Predicted probability": [float(x) for x in centers],
        "Observed": [float(y) for y in observed],
        "Perfect calibration": [float(x) for x in centers],
    }).set_index("Predicted probability")


def gain_chart_data(lift_gain: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Population targeted (%)": lift_gain["target_population_pct"].astype(float),
        "Model targeting": lift_gain["diabetic_patients_identified_pct"].astype(float),
        "Random targeting": lift_gain["target_population_pct"].astype(float),
    }).set_index("Population targeted (%)")


# ──────────────────────────────────────────────────────────────── main app ──

def main():
    st.set_page_config(
        page_title="Diabetes Risk Demo",
        page_icon=":hospital:",
        layout="wide",
    )

    bundle = get_model_and_eval()
    model       = bundle["model"]
    feature_cols = bundle["feature_cols"]
    sweep       = bundle["sweep"]
    lift_gain   = bundle["lift_gain"]
    centers, observed, _ = bundle["calibration"]

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("Model card")
        st.caption("SciEncephalon AI · Summer Intern Series 2026")
        st.caption("Intern: Reagan Lundy")
        st.markdown(
            "- **Model:** HistGradientBoosting + isotonic calibration.\n"
            "- **Inputs:** 8 non-invasive tabular features (Pima schema).\n"
            "- **Training data:** Real PIMA dataset when online; synthetic fallback if unavailable.\n"
            "- **Data cleanup:** impossible zero measurements are median-imputed.\n"
            "- **AI coach:** OpenAI-powered when configured; safe template fallback otherwise.\n"
            "- **Intended use:** Educational only.\n"
            "- **Known limitation:** PIMA = Pima women only; generalization "
            "to other populations is **NOT** validated."
        )
        st.warning(DISCLAIMER)

    st.title("Project 04 — Diabetes Risk (Educational Demo)")
    st.caption("SciEncephalon AI · Summer Intern Series 2026")
    st.caption("Intern: Reagan Lundy")
    st.warning(DISCLAIMER)

    col_left, col_right = st.columns([1, 1])

    # ── Patient inputs ────────────────────────────────────────────────────────
    with col_left:
        st.subheader("Patient inputs")
        with st.form("patient_form"):
            c1, c2 = st.columns(2)
            with c1:
                pregnancies    = st.number_input("Pregnancies", 0, 20, 1)
                glucose        = st.number_input("Glucose (mg/dL)", 50, 250, 120)
                blood_pressure = st.number_input("Diastolic BP (mm Hg)", 40, 130, 72)
                skin_thickness = st.number_input("Skin thickness (mm)", 0, 80, 23)
            with c2:
                insulin  = st.number_input("Insulin (uU/mL)", 0, 800, 95)
                bmi      = st.number_input("BMI", 12.0, 60.0, 28.0, step=0.1)
                dpf      = st.number_input("Diabetes pedigree score", 0.05, 3.0, 0.45, step=0.01)
                age      = st.number_input("Age", 18, 100, 40)

            threshold = st.slider(
                "Decision threshold (cut-off for 'high risk')",
                0.05, 0.95, 0.50, step=0.01,
            )
            submitted = st.form_submit_button("Estimate risk")

        if submitted:
            row = pd.DataFrame([{
                "pregnancies":    pregnancies,
                "glucose":        glucose,
                "blood_pressure": blood_pressure,
                "skin_thickness": skin_thickness,
                "insulin":        insulin,
                "bmi":            bmi,
                "dpf":            dpf,
                "age":            age,
            }])[feature_cols]
            risk = float(predict_proba(model, row)[0])

            nearest = sweep.iloc[(sweep["threshold"] - threshold).abs().argsort().iloc[0]]

            st.metric("Estimated risk", f"{risk*100:.1f} %",
                      delta=f"threshold = {threshold:.2f}")
            st.write(
                f"**At threshold {nearest['threshold']:.2f}:** "
                f"sensitivity = {nearest['sensitivity']:.2f}, "
                f"specificity = {nearest['specificity']:.2f}"
            )
            label = "HIGH risk band" if risk >= threshold else "Below threshold"
            st.info(f"Decision: **{label}**")

            st.subheader("AI Lifestyle Coach")
            api_key = get_secret("OPENAI_API_KEY")
            model_name = get_secret("OPENAI_MODEL", "gpt-4.1-mini")
            use_live_coach = bool(api_key)
            coach_mode = "OpenAI-powered AI coach" if use_live_coach else "Safe template AI coach"
            st.info(
                f"**{coach_mode}**\n\n"
                "This OpenAI-style coach turns the risk estimate into plain-English lifestyle suggestions. "
                "It is education-only and will not prescribe medication, doses, or calorie targets."
            )
            msg = llm_compose(
                row.iloc[0],
                risk=risk,
                api_key=api_key,
                prefer_llm=use_live_coach,
                model=model_name or "gpt-4.1-mini",
            )
            st.markdown(msg)

    # ── Charts ────────────────────────────────────────────────────────────────
    with col_right:
        st.subheader("Threshold trade-off (test set)")
        st.dataframe(
            sweep.style.format({
                "threshold":   "{:.2f}",
                "sensitivity": "{:.2f}",
                "specificity": "{:.2f}",
                "f1":          "{:.2f}",
            }),
            height=300,
        )
        st.caption(
            "Sensitivity = catch sick patients.  Specificity = don't alarm healthy patients.  "
            "Lower thresholds catch more sick patients but also raise more false alarms."
        )

        st.subheader("Calibration curve")
        st.line_chart(calibration_chart_data(centers, observed), height=320)
        st.caption(
            "Well-calibrated: when the model says 70% it really happens ~70% of the time.  "
            "Well-calibrated is NOT the same as 'accurate'."
        )

    st.subheader("Dataset comparison")
    st.caption(
        "The app trains the main model on cleaned real PIMA data when the dataset URL is reachable. "
        "PIMA zeros in glucose, blood pressure, skin thickness, insulin, and BMI are treated as missing values and replaced with medians."
    )
    st.dataframe(
        bundle["comparison"].style.format({
            "auc": "{:.2f}",
            "accuracy": "{:.2f}",
            "sensitivity": "{:.2f}",
            "specificity": "{:.2f}",
            "brier": "{:.2f}",
        }),
        hide_index=True,
    )
    if bundle["data_source"] == "synthetic_fallback":
        st.warning(
            "The real PIMA URL was not reachable in this run, so the app used synthetic fallback data. "
            "Run again with internet access for the real PIMA results."
        )

    st.subheader("Lift and gain targeting analysis")
    st.caption(
        "Patients are ranked from highest to lowest predicted risk. The table shows what percent of diabetic patients "
        "would be identified by targeting the top 10%, 20%, 30%, and so on. Lift compares that targeting to random selection."
    )
    lift_left, lift_right = st.columns([1, 1])
    with lift_left:
        st.dataframe(
            lift_gain[[
                "target_population_pct",
                "diabetic_patients_identified_pct",
                "lift",
                "diabetic_patients_found",
                "targeted_people",
            ]].rename(columns={
                "target_population_pct": "population targeted (%)",
                "diabetic_patients_identified_pct": "diabetic patients identified (%)",
                "lift": "lift (x)",
                "diabetic_patients_found": "diabetic patients found",
                "targeted_people": "people targeted",
            }).style.format({
                "population targeted (%)": "{:.1f}",
                "diabetic patients identified (%)": "{:.1f}",
                "lift (x)": "{:.2f}",
            }),
            hide_index=True,
            height=390,
        )
    with lift_right:
        st.line_chart(gain_chart_data(lift_gain), height=330)
        st.caption(
            "Example: if the 20% row shows 45% identified and 2.25x lift, the top-risk 20% contains "
            "45% of diabetic patients, which is 2.25 times better than random targeting."
        )

    st.subheader("Subgroup performance audit")
    fairness_left, fairness_right = st.columns(2)
    with fairness_left:
        st.markdown("**BMI buckets**")
        st.dataframe(
            bundle["fairness_bmi"].style.format({
                "prevalence": "{:.2f}",
                "auc": "{:.2f}",
                "sensitivity": "{:.2f}",
                "specificity": "{:.2f}",
            }),
            height=220,
        )
    with fairness_right:
        st.markdown("**Age buckets**")
        st.dataframe(
            bundle["fairness_age"].style.format({
                "prevalence": "{:.2f}",
                "auc": "{:.2f}",
                "sensitivity": "{:.2f}",
                "specificity": "{:.2f}",
            }),
            height=220,
        )

    st.divider()
    st.error(DISCLAIMER)


if __name__ == "__main__":
    main()
