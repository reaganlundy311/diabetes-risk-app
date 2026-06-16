# Project 04 — Diabetes Risk Assessment

**SciEncephalon AI · Summer Intern Series 2026**

**Intern:** Reagan Lundy

> **Not medical advice.** This project is an educational exercise for high-school
> interns. It is *not* a diagnostic tool. Anything below this line — model output,
> lifestyle suggestions, dashboards — is for learning only. For any medical
> decision, talk to a qualified healthcare provider.

## Goal

Predict Type-2 diabetes risk from **non-invasive** inputs (BMI, glucose, blood
pressure, family-history pedigree, age, ...) using a calibrated gradient-boosted
model — and go **deeper into evaluation** than a typical "accuracy = X%" demo:

- **Calibration curve.** When the model says "60% chance", does it really happen
  ~60% of the time? Calibrated probabilities are what let a clinician trust a
  number. Well-calibrated is **not** the same as accurate.
- **Sensitivity vs specificity vs threshold sweep.** The 0.5 default cut-off is
  almost never the right answer in healthcare. You'll see *why* by sweeping it.
- **Lift / gain deciles.** Rank patients by predicted risk and ask: if we target
  the top 10%, 20%, or 30% of the population, what percent of diabetic patients
  can we identify, and what is the lift versus random targeting?
- **Lifestyle coach (LLM stub).** A templated message that responds to a
  patient's actionable inputs — `BMI > 30`, `glucose > 140`, etc. — and emits an
  empathetic, disclaimer-bounded suggestion. Stretch goal: swap the stub for a
  real OpenAI / Anthropic call.

## Quick start

```bash
# from the repo root
pip install -r 04_diabetes_risk/requirements.txt

# notebook (synthetic data, runs end-to-end offline)
jupyter notebook 04_diabetes_risk/04_diabetes_risk.ipynb

# tests
pytest 04_diabetes_risk/tests/ -x

# Streamlit demo
streamlit run 04_diabetes_risk/app/streamlit_app.py
```

## Folder layout

```
04_diabetes_risk/
├── 04_diabetes_risk.ipynb     # the notebook (runs offline on synthetic data)
├── README.md                  # you are here
├── MENTOR_NOTES.md            # mentor-only — DO NOT share with intern
├── model_card.md              # what this model is / isn't, who it's for
├── requirements.txt
├── src/
│   ├── pipeline.py            # train / predict / calibration / threshold sweep
│   ├── lifestyle_coach.py     # templated LLM stub (replace as stretch goal)
│   └── fairness.py            # subgroup metrics (BMI / age / sex buckets)
├── app/
│   └── streamlit_app.py       # interactive patient demo
├── tests/
│   └── test_pipeline.py       # offline, <30s
└── data/
    └── loader.py              # PIMA URL + synthetic fallback
```

## Five-week arc

| Week | Dates | Goal | Definition of done |
|---|---|---|---|
| **1** | Jun 1 – Jun 5 | **Baseline on synthetic.** Run the notebook end-to-end. Read every cell. Understand the calibration curve and the threshold sweep. | Notebook runs cleanly. You can explain what "AUC", "sensitivity", "specificity", "calibrated probability" mean in plain English. |
| **2** | Jun 8 – Jun 12 | **Swap in the real PIMA dataset.** Investigate where metrics shift. Look up "Pima missing-encoded-as-zero" and decide your imputation strategy. | Side-by-side metrics (synthetic vs real). Short write-up of surprises and what you did about them. |
| **3** | Jun 15 – Jun 19 | **First stretch goal — pick ONE:** (a) wrap the model in the Streamlit UI and polish it, OR (b) replace `llm_compose` with a real LLM API call (OpenAI / Anthropic). | A working demo URL *or* a live LLM coach that still includes the disclaimer. |
| **4** | Jun 22 – Jun 26 | **Fairness audit + model card.** Use `src/fairness.py` to break metrics down by BMI / age buckets. Fill in `model_card.md`. | Subgroup table in the notebook, model card committed, plain-English explanation of the largest gap you see. |
| **5** | Jun 29 – Jul 3 | **Polish + 5-min demo video.** Lock the notebook, freeze the README, record the video. | Final demo run-through with the mentor. |

## Stretch goals (pick at least one in Week 3)

1. **Real LLM coach.** Replace `src/lifestyle_coach.py::llm_compose` with a real
   OpenAI / Anthropic call. **Keep the disclaimer** and **refuse to generate
   doses or prescriptions** — same guardrails as the stub.
2. **Streamlit Cloud deployment.** Deploy `app/streamlit_app.py` publicly. Add
   the URL to this README.
3. **Fairness audit.** Use `src/fairness.py` to compare AUC / sensitivity across
   BMI buckets and age buckets. Write 1–2 paragraphs on what you find.
4. **SHAP explanations.** Per-patient feature attribution: which input most
   pushed the prediction up or down?
5. **CDC BRFSS.** Replace PIMA with the CDC's BRFSS survey (400k+ rows). Beware
   schema differences — that's the lesson.

## Week 3 implementation

Both Week-3 options are implemented:

- **Public Streamlit app.** Deployed at
  `https://diabetes-risk-reagan-lundy.streamlit.app`.
- **Live LLM coach.** `src/lifestyle_coach.py::llm_compose` uses the OpenAI API
  when `OPENAI_API_KEY` is configured in Streamlit secrets, and falls back to the
  safe template when no key is available.

The Streamlit app includes patient inputs, calibrated risk prediction, threshold
trade-offs, calibration chart, dataset comparison, 10-decile lift/gain analysis,
subgroup performance audit, lifestyle suggestions, and prominent safety
disclaimers.

## Week 2 implementation

The project now uses the real PIMA dataset through `data/loader.py::load_pima`
when the public dataset URL is reachable. The Streamlit app and `main.py` train
their main model on cleaned PIMA data. PIMA's impossible zero values in
`glucose`, `blood_pressure`, `skin_thickness`, `insulin`, and `bmi` are treated
as missing values and replaced with each column's median. Both the app and
`main.py` also show a synthetic-baseline-vs-PIMA metric comparison.

The app also includes a 10-decile lift/gain targeting table. It ranks patients
from highest to lowest predicted risk and reports the cumulative percent of
diabetic patients identified by targeting the top 10%, 20%, 30%, and so on,
along with lift values such as `1.2x` or `2.0x` versus random targeting.

## Live LLM coach setup

The lifestyle coach uses a safe template by default. To turn on the live OpenAI
coach in Streamlit Cloud, open the app's settings, go to **Secrets**, and add:

```toml
OPENAI_API_KEY = "your-api-key-here"
OPENAI_MODEL = "gpt-4.1-mini"
```

Do not commit API keys to GitHub. If no key is configured, the app automatically
falls back to the template coach and still runs normally.

## Ethics checklist (must-haves before any demo)

- [x] Every user-facing string includes **"not medical advice"**.
- [x] The lifestyle coach **never** prescribes medication, dosing, or specific
      calorie targets.
- [x] **Disclosure of dataset bias.** The Pima Indians Diabetes dataset is
      *exclusively Pima women aged 21+*. Generalization to other populations,
      to men, or to younger / older patients is **NOT** validated.
- [x] No PHI is stored. Inputs in the Streamlit app live in browser state only.
- [x] A model card (`model_card.md`) is committed alongside the notebook.

## Why this notebook over a one-liner Kaggle model?

Three lessons most "intro ML" notebooks skip:
1. **Calibration matters more than accuracy** when a human is going to act on
   the number.
2. **The 0.5 threshold is a default, not an answer.** Different deployment
   contexts (primary-care screen vs ER) want very different thresholds.
3. **A model is not a product.** The lifestyle-coach layer is where the model
   becomes something a non-technical person can use — and where most of the
   ethics live.

---

*SciEncephalon AI — We bring clarity to your ambiguity.*
