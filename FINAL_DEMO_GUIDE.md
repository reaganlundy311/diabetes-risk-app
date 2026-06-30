# Five-Minute Final Demo Guide

## 0:00-0:30: Purpose and safety

**Point to:** the title, your name, and the warning at the top.

**Say:** "I built an educational diabetes-risk demo for the SciEncephalon AI
Summer Intern Series. It estimates risk from user inputs, but it is not medical
advice, not a diagnosis, and not clinically validated."

## 0:30-1:15: Inputs and prediction

**Point to:** the data-source selector, patient inputs, threshold, and Estimate
risk button.

**Say:** "The default mode uses CDC BRFSS survey-style inputs. PIMA is available
as a smaller clinical-style comparison. The model returns a calibrated
probability. For example, 0.63 means an estimated 63% probability from the
model's patterns, not a 63% certainty that a person has diabetes."

## 1:15-2:00: Threshold and explanation

**Point to:** the result, threshold table, and feature-attribution table.

**Say:** "The threshold changes the label, not the probability. Lower thresholds
catch more true cases, increasing sensitivity, but create more false alarms and
usually reduce specificity. The feature table shows what pushed this prediction
up or down; it explains model behavior and does not prove medical cause."

## 2:00-2:35: AI coach

**Point to:** the AI Lifestyle Coach label and response.

**Say:** "When an OpenAI key is configured, the coach turns the result into
plain-English educational suggestions. Without a key, a fixed safe template is
used. Both versions avoid diagnoses, prescriptions, medication doses, and
calorie targets, and both include a medical disclaimer."

## 2:35-3:15: Calibration and verified numbers

**Point to:** the calibration curve and dataset comparison.

**Say:** "Calibration asks whether predicted percentages match observed rates.
The metrics are calculated on a held-out 25% test set, and the app identifies
the active data source so synthetic backup results cannot be mistaken for real
CDC results. AUC measures ranking, while Brier score measures probability error."

## 3:15-3:50: Lift and gain

**Point to:** the 10-decile lift/gain table and chart.

**Say:** "This addresses mentor feedback about targeting. People are ranked by
risk in ten groups. Gain shows the percent of all diabetes cases captured by
targeting the highest-risk 10%, 20%, or 30%. Lift divides that gain by the
population targeted, so 2.0 times lift means twice the capture expected from
random selection."

## 3:50-4:30: Subgroup performance

**Point to:** the BMI and age tables and the gap explanation.

**Say:** "I compare AUC, sensitivity, and specificity across BMI and age groups.
The sensitivity gap is the highest subgroup sensitivity minus the lowest. A
0.50 gap means a 50-percentage-point difference in how often true cases are
caught. A large gap warns users that the model may miss more cases in one group."

## 4:30-5:00: Limitations and close

**Point to:** Important model limitations in the sidebar and What this model
cannot tell you at the bottom.

**Say:** "The model has important limits: BRFSS is self-reported, synthetic data
is not real patient data, PIMA only represents Pima women age 21 and older, and
subgroup results can be unstable. This final version emphasizes those limits in
plain language. The project demonstrates responsible evaluation, not a medical
product."
