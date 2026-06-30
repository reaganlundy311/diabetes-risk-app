# Model Card: Diabetes Risk Educational Demo

**SciEncephalon AI · Summer Intern Series 2026**  
**Created by Reagan Lundy**

> **Not medical advice.** This is an educational project, not a medical device.
> It has not been clinically validated and must not be used to diagnose anyone
> or make medical decisions.

## What the model does

The model estimates how closely a person's answers match patterns associated
with diabetes in the training data. It returns a number between 0% and 100%.
That number is an estimate, not a diagnosis.

The main app uses simple survey-style inputs such as BMI, age group, general
health, and high-blood-pressure history. A separate PIMA mode uses clinical-style
measurements for comparison.

## Model and data

- **Model:** gradient boosting with probability calibration.
- **Main data:** a reproducible sample of the CDC BRFSS 2022 public survey when
  that file is available.
- **Backup data:** synthetic BRFSS-shaped data when the CDC file cannot load.
- **Comparison data:** the PIMA Indians Diabetes dataset, with impossible zero
  measurements replaced by the median of each affected column.
- **Testing:** 25% of each dataset is held out and not used for training.
- **Repeatability:** data splits and generated data use random seed 42.

## How to read the output

- **Risk estimate:** `0.63` means an estimated 63% probability based on the
  model's learned patterns. It does not mean the person has diabetes.
- **Threshold:** the cut-off used to flag a higher-risk result. Lowering it
  catches more true cases but also creates more false alarms.
- **AUC:** how well the model ranks diabetes cases above non-diabetes cases.
  `0.50` is no better than random ranking; `1.00` is perfect ranking.
- **Sensitivity:** the share of actual diabetes cases the model catches.
- **Specificity:** the share of non-diabetes cases the model correctly leaves
  unflagged.
- **Brier score:** the average squared error of the probabilities. Lower is
  better.
- **Calibration:** whether predicted percentages match observed rates.

The app calculates and displays these metrics from the current held-out test
set. It labels the active data source so synthetic results are never presented
as real CDC or PIMA results.

## Verified offline baseline

These values are a repeatable software check, not a claim of clinical quality.
They come from 10,000 synthetic BRFSS-shaped rows, seed 42, a 75/25 split, and a
0.50 decision threshold.

| Metric | Value |
|---|---:|
| AUC | 0.720 |
| Accuracy | 0.681 |
| Sensitivity | 0.372 |
| Specificity | 0.865 |
| Brier score | 0.202 |

## Intended use

Use this project to learn about risk prediction, calibration, thresholds,
lift/gain, feature explanations, and subgroup evaluation. It may be used for a
classroom or portfolio demonstration.

Do not use it for diagnosis, treatment, screening decisions, insurance,
employment, or any decision that affects a real person.

## Important limitations

- **Not clinically validated:** no clinician or regulator has approved this
  model for patient care.
- **A probability is not a diagnosis:** only a qualified healthcare provider
  using appropriate testing can diagnose diabetes.
- **Survey limitations:** BRFSS answers are self-reported and can be incomplete
  or inaccurate. A sample also does not preserve every property of the full
  survey.
- **Synthetic fallback:** backup data is computer-generated and is not evidence
  about real patients.
- **PIMA population limits:** PIMA contains only Pima women age 21 and older.
  Results are not validated for men, younger people, other communities, or the
  general population.
- **Uneven subgroup performance:** sensitivity and specificity can differ by
  age or BMI group. The app reports these gaps, but small groups can make the
  numbers unstable.
- **Limited inputs:** the model does not use all relevant medical history,
  laboratory tests, medications, pregnancy status, access to care, or social
  factors.
- **Explanation is not causation:** a feature attribution describes the model's
  behavior. It does not prove that a feature caused diabetes.
- **Threshold choice is contextual:** 0.50 is a demonstration default, not a
  medically recommended cut-off.

## Safety and privacy

- The app repeatedly states that it is **not medical advice**.
- The coach does not diagnose or provide medications, doses, prescriptions, or
  calorie targets.
- The app does not intentionally save entered health information.
- A live OpenAI coach is used only when an API key is configured; otherwise the
  app uses a fixed safety-reviewed template.
- Anyone concerned about diabetes should speak with a qualified healthcare
  provider.

## Fairness check

The app compares AUC, sensitivity, and specificity across age and BMI groups.
Its displayed sensitivity gap is calculated as:

`highest subgroup sensitivity - lowest subgroup sensitivity`

For example, `0.75 - 0.25 = 0.50`, a 50-percentage-point gap. The lower-performing
group has more true diabetes cases missed at that threshold. This check reveals
uneven performance; it does not prove that the model is fair or unfair by itself.

## Public demonstration

<https://diabetes-risk-reagan-lundy.streamlit.app>
