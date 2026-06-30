"""Lifestyle coach for the Diabetes Risk educational demo.

`llm_compose(patient_row)` reads the patient's most-actionable inputs
(BMI, glucose, blood pressure, age) and returns a personalized, plain-English,
DISCLAIMER-BOUNDED message.

If an OpenAI API key is provided, `llm_compose` can use a live LLM. Otherwise it
falls back to the deterministic template so the app always works.

NEVER prescribe. NEVER dose. ALWAYS include "not medical advice".
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping

DISCLAIMER = (
    "Not medical advice. This message is for education only — please talk to a "
    "qualified healthcare provider before changing your diet, exercise, or medication."
)

_RULES = [
    {
        "key": "bmi",
        "label": "Body Mass Index",
        "trigger": lambda v: v is not None and v > 30,
        "message": (
            "Your BMI is above 30, which is a category called obesity. Small, "
            "sustainable changes — like a daily 20-minute walk and swapping sugary "
            "drinks for water — are a great place to start."
        ),
    },
    {
        "key": "glucose",
        "label": "Plasma glucose",
        "trigger": lambda v: v is not None and v > 140,
        "message": (
            "Your glucose reading is on the higher side. Ask your provider about an "
            "A1C test — it's a simple finger-stick that gives a 3-month average."
        ),
    },
    {
        "key": "blood_pressure",
        "label": "Blood pressure",
        "trigger": lambda v: v is not None and v > 85,
        "message": (
            "Your diastolic blood pressure is elevated. Tracking your BP weekly and "
            "watching sodium intake can help. Bring the log to your next checkup."
        ),
    },
    {
        "key": "age",
        "label": "Age",
        "trigger": lambda v: v is not None and v >= 45,
        "message": (
            "Adults 45+ are routinely screened for Type-2 diabetes. If you haven't "
            "had a screening lately, it's worth bringing up at your next visit."
        ),
    },
]


def _as_mapping(patient_row) -> Mapping:
    """Accept dict, pandas Series, or any mapping-like."""
    if hasattr(patient_row, "to_dict"):
        return patient_row.to_dict()
    return dict(patient_row)


def template_compose(patient_row, risk: float | None = None) -> str:
    """Compose a templated lifestyle message from a single patient row.

    Parameters
    ----------
    patient_row : Mapping (dict, pd.Series, etc.)
        Expected keys (any subset): bmi, glucose, blood_pressure, age.
    risk : float, optional
        Model-estimated probability in [0, 1].

    Returns
    -------
    str — always ends with the "not medical advice" disclaimer.
    """
    row = _as_mapping(patient_row)

    triggered = []
    for rule in _RULES:
        val = row.get(rule["key"])
        try:
            if rule["trigger"](val):
                triggered.append(rule)
        except (TypeError, ValueError):
            continue

    lines = []

    if risk is not None:
        pct = max(0.0, min(1.0, float(risk))) * 100
        band = "lower" if pct < 20 else ("moderate" if pct < 50 else "higher")
        lines.append(
            f"Based on the inputs you shared, the model estimates your "
            f"diabetes-risk band as **{band}** (~{pct:.0f}%)."
        )

    if not triggered:
        lines.append(
            "None of your inputs crossed our simple lifestyle flags. Keep up "
            "balanced meals, regular movement, and routine checkups."
        )
    else:
        lines.append("Here are a few things worth a closer look:")
        for rule in triggered:
            lines.append(f"- **{rule['label']}** — {rule['message']}")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def _safe_patient_summary(patient_row) -> dict:
    row = _as_mapping(patient_row)
    keys = ["glucose", "bmi", "blood_pressure", "age", "dpf"]
    return {key: row.get(key) for key in keys if key in row}


def openai_compose(
    patient_row,
    risk: float | None = None,
    api_key: str | None = None,
    model: str = "gpt-4.1-mini",
) -> str:
    """Compose a lifestyle message using the OpenAI Responses API.

    The prompt intentionally asks for education-only, non-prescriptive guidance.
    If the API call fails, the caller should fall back to `template_compose`.
    """
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is not configured")

    summary = _safe_patient_summary(patient_row)
    flags = ", ".join(triggered_flags(patient_row)) or "none"
    risk_str = f"{max(0.0, min(1.0, float(risk))) * 100:.0f}%" if risk is not None else "unknown"
    instructions = (
        "You are an educational lifestyle coach for a diabetes-risk demo. "
        "Give 2-4 short, empathetic, actionable lifestyle suggestions. "
        "Do not diagnose. Do not prescribe medication. Do not mention doses. "
        "Do not give specific calorie targets. Do not claim certainty. "
        "End with this exact sentence: "
        "\"Not medical advice. Talk to a qualified healthcare provider.\""
    )
    prompt = (
        f"Model-estimated diabetes-risk probability: {risk_str}\n"
        f"Patient summary: {summary}\n"
        f"Triggered lifestyle flags: {flags}\n"
        "Write for a general audience in plain English."
    )
    client = OpenAI(api_key=key)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        max_output_tokens=260,
    )
    text = response.output_text.strip()
    if "not medical advice" not in text.lower():
        text = f"{text}\n\nNot medical advice. Talk to a qualified healthcare provider."
    return text


def llm_compose(
    patient_row,
    risk: float | None = None,
    api_key: str | None = None,
    prefer_llm: bool = False,
    model: str = "gpt-4.1-mini",
) -> str:
    """Compose a lifestyle message, using OpenAI when requested and configured."""
    if prefer_llm:
        try:
            return openai_compose(patient_row, risk=risk, api_key=api_key, model=model)
        except Exception:
            return template_compose(patient_row, risk=risk)
    return template_compose(patient_row, risk=risk)


def triggered_flags(patient_row) -> Iterable[str]:
    """Return the list of rule keys that fired — handy for unit tests."""
    row = _as_mapping(patient_row)
    out = []
    for rule in _RULES:
        try:
            if rule["trigger"](row.get(rule["key"])):
                out.append(rule["key"])
        except (TypeError, ValueError):
            continue
    return out
