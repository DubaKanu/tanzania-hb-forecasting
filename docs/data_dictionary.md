# Data dictionary

What every column and file in this repository means. The raw file's own columns and where they come from are in `docs/data_provenance.md` (section 4). This page describes what the code makes from them. Counts are for the 2,685 eligible women.

## 1. Prepared cohort tables

`model_ready.csv` (all 2,685 eligible women, no partition label) and `train.csv`, `validation.csv`, `test.csv` (1,884, 402 and 399 women) all have the same 13 columns.

| Column | Meaning | Unit | Role | Missing | Range seen |
|---|---|---|---|---|---|
| `hb_visit1_g_dl` | Haemoglobin at the first recorded visit | g/dL | input | 0 | 3.9 to 18.0 |
| `gestation_visit1_weeks` | Gestational week at the first visit | weeks | input | 0 | 1 to 37 |
| `maternal_age_years` | Age | years | input | 0 | 11 to 54 |
| `pregnancy_count` | Number of pregnancies, as recorded | count | input | 0 | 1 to 12 |
| `height_cm` | Height | cm | input | 0 | 0 to 180 |
| `weight_visit1_kg` | Weight at the first visit | kg | input | 0 | 16 to 159 |
| `pulse_visit1_bpm` | Pulse at the first visit | beats per minute | input | 132 (4.9%) | 61 to 123 |
| `respiratory_rate_visit1` | Respiratory rate at the first visit | breaths per minute | input | 643 (23.9%) | 12 to 103 |
| `systolic_bp_visit1` | Systolic blood pressure, from the text "120/80" | mmHg | input | 18 (0.7%) | 80 to 165 |
| `diastolic_bp_visit1` | Diastolic blood pressure, from the same text | mmHg | input | 18 (0.7%) | 16 to 99 |
| `target_hb_visit2_g_dl` | Haemoglobin at the next recorded visit | g/dL | target | 0 | 4.1 to 17.2 |
| `gestation_visit2_weeks` | Gestational week at that visit | weeks | eligibility and follow-up gap only, never a model input | 0 | 8 to 42 |
| `_source_row` | The release's own row number (its `SN` column), used as the record ID | none | identifier | 0 | 1 to 8811 |

Notes
- The first ten rows of the table are the model inputs, in the fixed order `FEATURE_COLUMNS` in `src/pipeline.py`.
- `_source_row` is a row number, not a patient identifier. The release has no verified patient identifier.
- Values that look wrong are kept, not repaired: 2 women have `height_cm` 0 and 2 more are under 100 cm, and 3 weigh under 30 kg. The eligibility screen only covers haemoglobin, gestational weeks and their order. The API has stricter limits than these tables (see section 5).
- Missing inputs are filled with the median of the training women, and a 0 or 1 flag column records which were missing (20 columns in all go into the forests). The filling is fitted on training rows only, and inside each cross-validation fold on that fold's training rows only.

## 2. Partition and fold files

| File | Columns | Meaning |
|---|---|---|
| `split_manifest.csv` | `_source_row`, `partition` | Which partition (`train`, `dev` or `test`) every eligible woman is in. 2,685 rows. |
| `cv_folds.csv` | `_source_row`, `cv_fold` | The cross-validation fold (0, 1 or 2) of every training woman. 1,884 rows. Written by the fixed rule `assign_cv_folds`. |
| `exclusion_log.txt` | text | How many of the 8,817 rows were excluded for each reason, written by `src/audit.py`. |

## 3. Result tables

`reports/oof_stage1.csv` (out-of-fold table, one row per training woman, the table Stage 2 learns from)

| Column | Meaning |
|---|---|
| `run_id` | Short ID of the run that made the table (built from the data checksums, the folds and the fixed settings) |
| `record_id` | `_source_row` of the woman |
| `fold` | The fold she was held out in |
| `observed_hb` | Her real haemoglobin at the next recorded visit (g/dL) |
| `oof_forecast` | Stage 1 forecast from a forest that never saw her (g/dL) |
| `tree_disagreement` | Standard deviation of the individual trees' forecasts for her (g/dL) |
| `abs_error` | `abs(observed_hb - oof_forecast)`, the target Stage 2 learns (g/dL) |

`reports/stage2_dev_table.csv` (one row per development woman; the locked test makes `locked_test_table.csv` with the same columns)

| Column | Meaning |
|---|---|
| `record_id` | `_source_row` of the woman |
| `observed_hb`, `forecast`, `abs_error`, `signed_error` | Real value, Stage 1 forecast, absolute error, and forecast minus real value (g/dL) |
| `tree_disagreement` | As above |
| `stage2_score` | Stage 2's estimate of `abs_error` (g/dL). Lower means a more reliable forecast. Not a confidence interval. |
| `n_missing_inputs` | How many of the ten inputs were missing (0 to 10) |
| `first_hb` | The first haemoglobin (g/dL), used by the simple rule and the audit |
| `gap_weeks` | Weeks between the two visits, used by the audit |
| `simple_rule_score` | The simple rule as one number: missing inputs, plus the distance of `first_hb` from the training median divided by 100 |

## 4. Saved models

Both files hold plain scikit-learn objects in a Python dictionary, so they load without any class from this code.

| File | Keys |
|---|---|
| `models/stage1_model.joblib` | `model` (the fitted random forest), `imputer` (the training-fitted median imputer), `name` (`random_forest`) |
| `models/stage2_model.joblib` | `model` (the Stage 2 forest), `run_id`, `config`, `input_columns` (the 12 Stage 2 inputs), `train_hb_median`, `demo_cutoff`, `demo_cutoff_rule`, `versions` |

## 5. Demonstration API

Inputs of `POST /forecast`. NaN and Infinity are refused. To leave an optional value out, omit it or send `null`.

| Field | Required | Accepted range |
|---|---|---|
| `hb_visit1_g_dl` | yes | 3 to 20 |
| `gestation_visit1_weeks` | yes | 1 to 42 |
| `maternal_age_years` | yes | 10 to 60 |
| `pregnancy_count` | yes | 0 to 25 |
| `height_cm` | yes | 100 to 200 |
| `weight_visit1_kg` | yes | 30 to 150 |
| `pulse_visit1_bpm` | no | 20 to 250 |
| `respiratory_rate_visit1` | no | 4 to 120 |
| `systolic_bp_visit1` | no | 40 to 300 |
| `diastolic_bp_visit1` | no | 10 to 200 |

The limits for haemoglobin and gestational week are the eligibility screen's. The limits for age, height and weight are plausible values, and they are stricter than the tables (see the notes in section 1). The limits for the pregnancy count and the four optional fields are wide plausibility limits, not clinical thresholds: every woman in the training and development tables is inside them.

Output of `POST /forecast`

| Field | Meaning |
|---|---|
| `predicted_hb_visit2_g_dl` | Stage 1 forecast, rounded to 2 decimals (g/dL) |
| `estimated_abs_error_g_dl` | Stage 2 estimate of how far the forecast is likely to be from the real value, rounded to 3 decimals (g/dL) |
| `reliability_flag` | `kept` if the estimate is at or below the cutoff, otherwise `set_aside` |
| `cutoff_g_dl` | The demonstration cutoff: the 80th percentile of the development women's Stage 2 scores. It is fitted on development data, so it is a different method from the exact-coverage ranking used in the evaluation. |
| `reliability_note` | The same result in a sentence |
| `disclaimer` | Research demonstration output only. Not a diagnosis. Does not replace clinical blood testing. |

Errors: `422` for an input outside the ranges, a missing required input or a malformed request (the message names the field and never repeats the value sent); `503` if the model files do not exist yet.
