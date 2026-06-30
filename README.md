# Diabetes Risk Assessment

**SciEncephalon AI · Summer Intern Series 2026**

**Created by Reagan Lundy**

> **Not medical advice.** This app is an educational demo. It is not a
> diagnostic tool and should not be used to make medical decisions. If you are
> concerned about diabetes risk, talk with a qualified healthcare provider.

## What this app does

This app estimates diabetes risk from non-invasive information such as BMI, age,
general health, blood-pressure history, and recent physical or mental health
days. It also explains how the estimate should be interpreted and where the
model may be less reliable.

The goal is not just to show a risk percentage. The app also helps users
understand:

- what influenced the estimate,
- how the decision threshold changes results,
- whether the risk percentage is well calibrated,
- how targeting higher-risk groups compares with random selection,
- whether the model performs differently across BMI or age groups,
- and why the output is educational only.

## Public app

Streamlit app:

`https://diabetes-risk-reagan-lundy.streamlit.app`

## Main features

- **Risk estimate.** Shows an estimated diabetes-risk percentage for the entered
  inputs.
- **Risk threshold.** Lets users choose when the app should label someone as a
  higher-risk band.
- **AI lifestyle coach.** Gives plain-English, education-only suggestions. When
  an OpenAI API key is configured, the app can use OpenAI; otherwise it uses a
  safe template response.
- **Explanation table.** Shows which inputs pushed the estimate up or down.
- **Calibration curve.** Checks whether the model's risk percentages are
  believable on test data.
- **Lift and gain table.** Shows how many diabetes cases can be found by
  focusing on the highest-risk 10%, 20%, 30%, and so on.
- **Subgroup performance check.** Compares performance across BMI and age groups
  so users can see where the model may be less reliable.
- **Model card.** Documents the model, data, limitations, and safety rules.

## Data sources

The default app uses CDC BRFSS public health survey data when it is reachable.
Because that dataset is large, the app samples cleaned rows so it can run
responsively in Streamlit.

If the CDC file cannot load, the app uses synthetic survey-shaped backup data so
the demo still works. A smaller PIMA clinical-style dataset is also included as
a comparison option. PIMA is not representative of everyone, so it should not be
treated as a general medical dataset.

## Important terms

- **AUC** measures how well the model separates people with diabetes from people
  without diabetes. Higher is better.
- **Sensitivity** means the model catches true diabetes cases.
- **Specificity** means the model correctly avoids flagging people who do not
  have diabetes.
- **Threshold** is the cut-off for labeling an estimate as a higher-risk band.
- **Calibration** checks whether the risk percentage is believable. For example,
  if many people receive a 70% estimate, about 70% should actually have diabetes
  in the test data.
- **Lift** shows how much better model-based targeting is than random selection.

## Run locally

From this folder:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app/streamlit_app.py
```

Run tests:

```bash
python3 -m pytest tests/ -q
```

## Optional OpenAI coach setup

The app works without an API key. To enable the live OpenAI coach in Streamlit
Cloud, add these secrets in the Streamlit app settings:

```toml
OPENAI_API_KEY = "your-api-key-here"
OPENAI_MODEL = "gpt-4.1-mini"
```

Do not commit API keys to GitHub.

## Safety rules

- The app always includes a "not medical advice" disclaimer.
- The AI coach must not prescribe medications, doses, or calorie targets.
- The app does not store personal health information.
- The model is not clinically validated.
- The output should be used only for learning and demonstration.

## Final project status

The five-week project is complete:

- The notebook is fully executed and ready for a final walkthrough.
- The public app includes the model, explanations, threshold analysis,
  calibration, lift/gain, subgroup checks, and safety notices.
- The model card uses plain language and clearly states the data sources,
  verified baseline numbers, intended use, and limitations.
- Automated tests check probability bounds, threshold behavior, lift math,
  subgroup-gap math, coach safety wording, and both data schemas.
- A five-minute presentation script is included in `FINAL_DEMO_GUIDE.md`.

## Version note

This README describes the final Week 5 submission. Any future changes should be
tested and then documented here before the public app is updated.
