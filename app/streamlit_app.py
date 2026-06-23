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

from data.loader import load_brfss, load_pima, load_synthetic, load_synthetic_brfss  # noqa: E402
from src.lifestyle_coach import llm_compose                # noqa: E402
from src.pipeline import (                                 # noqa: E402
    calibration_curve,
    feature_attribution,
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


@st.cache_resource(show_spinner="Training calibrated model…")
def get_model_and_eval(model_mode: str):
    if model_mode == "CDC BRFSS survey model":
        primary_eval = evaluate_dataset(load_brfss(sample_n=50000), "CDC BRFSS survey sample")
        baseline_eval = evaluate_dataset(load_synthetic_brfss(n=10000, seed=42), "Synthetic BRFSS baseline")
    else:
        primary_eval = evaluate_dataset(load_pima(clean=True), "Real PIMA, median-imputed")
        baseline_eval = evaluate_dataset(load_synthetic(n=600, seed=42), "Synthetic PIMA baseline")

    model = primary_eval["model"]
    Xte = primary_eval["X_test"]
    yte = primary_eval["y_test"]
    probs_te = primary_eval["probs_test"]
    sweep = threshold_sweep(yte, probs_te)
    lift_gain = lift_gain_table(yte, probs_te, n_deciles=10)
    centers, observed, expected = calibration_curve(yte, probs_te, n_bins=10)
    comparison = pd.DataFrame([
        {
            "dataset": baseline_eval["label"],
            "auc": baseline_eval["auc"],
            "accuracy": baseline_eval["accuracy"],
            "sensitivity": baseline_eval["sensitivity"],
            "specificity": baseline_eval["specificity"],
            "brier": baseline_eval["brier"],
        },
        {
            "dataset": primary_eval["label"],
            "auc": primary_eval["auc"],
            "accuracy": primary_eval["accuracy"],
            "sensitivity": primary_eval["sensitivity"],
            "specificity": primary_eval["specificity"],
            "brier": primary_eval["brier"],
        },
    ])
    return {
        "model":        model,
        "feature_cols": primary_eval["feature_cols"],
        "X_test":       Xte,
        "y_test":       yte,
        "probs_test":   probs_te,
        "sweep":        sweep,
        "lift_gain":    lift_gain,
        "calibration":  (centers, observed, expected),
        "fairness_bmi": subgroup_metrics(model, Xte, yte, by="bmi_bucket"),
        "fairness_age": subgroup_metrics(model, Xte, yte, by="age_bucket" if "age" in Xte.columns else "age_group"),
        "comparison":   comparison,
        "data_source":  primary_eval["df"].attrs.get("source", "unknown"),
        "model_mode":   model_mode,
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


def friendly_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "bmi": "BMI",
        "age": "Age",
        "age_group": "Age group",
        "sex_female": "Female sex indicator",
        "high_bp": "High blood pressure history",
        "general_health": "General health rating",
        "physical_health_bad_days": "Physical-health bad days",
        "mental_health_bad_days": "Mental-health bad days",
        "checkup_within_year": "Checkup within past year",
        "pregnancies": "Pregnancies",
        "glucose": "Glucose",
        "blood_pressure": "Blood pressure",
        "skin_thickness": "Skin thickness",
        "insulin": "Insulin",
        "dpf": "Diabetes pedigree score",
    }
    out = df.copy()
    if "feature" in out.columns:
        out["feature"] = out["feature"].map(labels).fillna(out["feature"])
    return out


def largest_subgroup_gap(table: pd.DataFrame, metric: str = "sensitivity") -> dict:
    valid = table.dropna(subset=[metric])
    if valid.empty:
        return {"gap": 0.0, "best": "not available", "worst": "not available"}
    best_row = valid.loc[valid[metric].idxmax()]
    worst_row = valid.loc[valid[metric].idxmin()]
    return {
        "gap": float(best_row[metric] - worst_row[metric]),
        "best": str(best_row["subgroup"]),
        "worst": str(worst_row["subgroup"]),
    }


# ──────────────────────────────────────────────────────────────── main app ──

def main():
    st.set_page_config(
        page_title="Diabetes Risk Demo",
        page_icon=":hospital:",
        layout="wide",
    )

    model_mode = st.sidebar.selectbox(
        "Model/data source",
        ["CDC BRFSS survey model", "PIMA clinical-style model"],
        help="BRFSS is the broad CDC survey stretch goal. PIMA is kept as a smaller clinical-style comparison.",
    )
    bundle = get_model_and_eval(model_mode)
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
            "- **Default data:** CDC BRFSS survey sample with synthetic fallback.\n"
            "- **Alternate data:** PIMA clinical-style model remains available.\n"
            "- **Explanations:** per-patient SHAP-style feature attribution.\n"
            "- **AI coach:** OpenAI-powered when configured; safe template fallback otherwise.\n"
            "- **Intended use:** Educational only.\n"
            "- **Known limitation:** Survey and PIMA models are educational, not clinical."
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
            if model_mode == "CDC BRFSS survey model":
                with c1:
                    bmi = st.number_input("BMI", 12.0, 70.0, 29.0, step=0.1)
                    age_group = st.selectbox(
                        "Age group",
                        list(range(1, 14)),
                        index=6,
                        format_func=lambda x: f"BRFSS age group {x}",
                    )
                    sex_female = st.selectbox("Sex", [0, 1], index=1, format_func=lambda x: "Female" if x else "Male")
                    high_bp = st.selectbox("Ever told you had high blood pressure?", [0, 1], index=0, format_func=lambda x: "Yes" if x else "No")
                with c2:
                    general_health = st.selectbox("General health", [1, 2, 3, 4, 5], index=2, format_func=lambda x: ["Excellent", "Very good", "Good", "Fair", "Poor"][x - 1])
                    physical_health_bad_days = st.number_input("Poor physical-health days in past month", 0, 30, 3)
                    mental_health_bad_days = st.number_input("Poor mental-health days in past month", 0, 30, 4)
                    checkup_within_year = st.selectbox("Routine checkup within past year?", [0, 1], index=1, format_func=lambda x: "Yes" if x else "No")
            else:
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
            if model_mode == "CDC BRFSS survey model":
                row = pd.DataFrame([{
                    "bmi": bmi,
                    "age_group": age_group,
                    "sex_female": sex_female,
                    "high_bp": high_bp,
                    "general_health": general_health,
                    "physical_health_bad_days": physical_health_bad_days,
                    "mental_health_bad_days": mental_health_bad_days,
                    "checkup_within_year": checkup_within_year,
                }])[feature_cols]
                coach_row = {"bmi": bmi, "age": 21 + int(age_group) * 5, "blood_pressure": 90 if high_bp else 72}
            else:
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
                coach_row = row.iloc[0]
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
                coach_row,
                risk=risk,
                api_key=api_key,
                prefer_llm=use_live_coach,
                model=model_name or "gpt-4.1-mini",
            )
            st.markdown(msg)
            st.subheader("Per-patient feature attribution")
            explanation = friendly_feature_names(feature_attribution(model, row.iloc[0], bundle["X_test"]).head(6))
            st.caption("Positive values pushed this patient's risk upward; negative values pushed it downward.")
            st.dataframe(
                explanation[["feature", "patient_value", "baseline_value", "risk_contribution", "direction", "method"]].style.format({
                    "patient_value": "{:.2f}",
                    "baseline_value": "{:.2f}",
                    "risk_contribution": "{:+.3f}",
                }),
                hide_index=True,
                height=260,
            )

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
        "The default app trains on CDC BRFSS survey data when reachable and falls back to synthetic survey-shaped data if needed. "
        "The PIMA mode remains available as a smaller clinical-style comparison with median-imputed zero measurements."
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
    if "fallback" in bundle["data_source"]:
        st.warning(
            "The selected public dataset was not reachable in this run, so the app used synthetic fallback data. "
            "Run again with internet access for the real dataset results."
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
        st.markdown("**Age groups**")
        st.dataframe(
            bundle["fairness_age"].style.format({
                "prevalence": "{:.2f}",
                "auc": "{:.2f}",
                "sensitivity": "{:.2f}",
                "specificity": "{:.2f}",
            }),
            height=220,
        )

    bmi_gap = largest_subgroup_gap(bundle["fairness_bmi"])
    age_gap = largest_subgroup_gap(bundle["fairness_age"])
    st.markdown("**Week 4 fairness conclusion**")
    st.info(
        f"The largest sensitivity gap in this run is **{age_gap['gap']:.2f} across age groups** "
        f"(highest: {age_gap['best']}; lowest: {age_gap['worst']}). "
        f"The BMI sensitivity gap is **{bmi_gap['gap']:.2f}**. Lower sensitivity means the model "
        "misses more true diabetes cases in that subgroup. These gaps should be disclosed and "
        "investigated before any real-world use. Not medical advice."
    )

    st.divider()
    st.error(DISCLAIMER)


if __name__ == "__main__":
    main()
