"""Lifestyle-coach STUB for Project 04.

`llm_compose(patient_row)` reads the patient's most-actionable inputs
(BMI, glucose, blood pressure, age) and returns a personalized, plain-English,
DISCLAIMER-BOUNDED message.

STRETCH GOAL: replace `llm_compose` body with a real OpenAI / Anthropic call.
  1. pip install openai
  2. Set OPENAI_API_KEY in your .env or VS Code launch.json env vars.
  3. See the commented-out example at the bottom of this file.

NEVER prescribe. NEVER dose. ALWAYS include "not medical advice".
"""

from __future__ import annotations

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


def llm_compose(patient_row, risk: float | None = None) -> str:
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


# ---------------------------------------------------------------------------
# STRETCH GOAL — Real LLM version (Week 3+)
# Uncomment and fill in your API key to replace the stub above.
# ---------------------------------------------------------------------------
# import os, openai
#
# def llm_compose(patient_row, risk=None):
#     row = _as_mapping(patient_row)
#     flags = ", ".join(triggered_flags(patient_row)) or "none"
#     risk_str = f"{risk*100:.0f}%" if risk is not None else "unknown"
#
#     prompt = f"""You are a non-prescriptive lifestyle assistant.
# Patient data (do NOT share back verbatim): {dict(row)}
# Model risk estimate: {risk_str}
# Triggered lifestyle flags: {flags}
#
# Give 2-4 short, empathetic, actionable lifestyle suggestions. 
# Never prescribe medication, doses, or specific calorie counts.
# End with exactly: "Not medical advice. Talk to a qualified healthcare provider."
# """
#     client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#     resp = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=300,
#     )
#     return resp.choices[0].message.content
