# Frozen protocol for the locked test

Project: Predicting Follow-up Haemoglobin and Identifying Unreliable Predictions in a Tanzanian Antenatal Cohort
Author: Josephine Duba Kanu. Supervisor: Emmanuel Adjei.

**Status: draft until `python src/freeze_protocol.py` has been run.** That command records a hash of this file, the code, the data files and the saved models in `reports/protocol_freeze.json`. After that, none of them may change before the locked test is run, and `src/run_locked_test.py` refuses to run if any of them has changed, or if numpy, pandas or scikit-learn are not the versions recorded at the freeze. Items marked *proposed* have not yet been approved by the supervisor.

## 1. What is tested

Whether a learned second-stage score picks out reliable haemoglobin forecasts better than tree disagreement, a simple rule and random selection, on women that none of the models has seen. A null or inconclusive result is a valid finding and will be reported as it is.

## 2. Data and partitions

- Pinned release: Zenodo 10.5281/zenodo.15309733, Version 1, file `maternal_dataset_csv.csv`, MD5 `0253d2d385dd80d6c9bb82df6b85fcc8`. The file is not stored in the repository. Retrieval, both checksums and the 12 of 683 columns that are loaded are in `docs/data_provenance.md`.
- 2,685 eligible women: 1,884 training, 402 development, 399 locked test. Seed 42. Women with an identical ten-input vector stay in one partition. `split_manifest.csv` lists the partition of every woman.
- The locked test is opened once, by `src/run_locked_test.py`, after this protocol is frozen. Results are kept whatever they are. Seeds are not changed because a result is disappointing.
- Before the split, cohort-level descriptives were computed on all 2,685 eligible women, so they include the 399: the exclusion counts, the missing values, the median first haemoglobin, the follow-up gap, and a chart of first against second haemoglobin (`notebooks/model_notebook.ipynb`). No model was fitted, tuned, compared or evaluated on the 399, and no model has made a forecast for them. `src/run_stage2.py` never opens `test.csv`, and a test checks that.

## 3. Stage 1 (frozen)

- Inputs: the ten first-visit values only (haemoglobin, gestational week, age, number of pregnancies, height, weight, pulse, respiratory rate, systolic and diastolic blood pressure). Never second-visit data, visit timing, delivery fields or the dataset's own risk label.
- Model: random forest, 200 trees, depth 6, minimum leaf size 5, seed 42, fitted once on all 1,884 training women, with training-fitted median imputation and missing-value flags. It was chosen by the lowest training cross-validation MAE among five fixed candidates (median, carry-forward, two-input linear, ridge, forest). Nothing is tuned.
- This one frozen model supplies the forecast and the tree disagreement for every selector.

## 4. Stage 2 (frozen)

- Out-of-fold forecasts for the 1,884 training women come from the same 3-fold cross-validation, with folds from the fixed rule in `src/pipeline.py` (`assign_cv_folds`). They are saved in `reports/oof_stage1.csv` with a run ID.
- A separate random forest regressor is trained on the out-of-fold absolute errors, with the same settings as Stage 1 (*proposed*). Its inputs are the ten first-visit values, the out-of-fold forecast and the tree disagreement. It never sees in-sample residuals.
- For development and test women, the frozen Stage 1 forest supplies the forecast and the tree disagreement.

## 5. Selectors (all use the same Stage 1 forecasts)

A lower score means a more reliable forecast.

- Learned: the Stage 2 estimated absolute error.
- Tree disagreement: the standard deviation of the individual trees' forecasts.
- Simple rule (*proposed*): the number of missing inputs, ties broken by the distance of the first haemoglobin from the training median, then by record ID.
- Random selection: 200 repeats with fixed seeds (seed 42), reporting the mean and the 2.5th to 97.5th percentiles (*proposed*).

## 6. Primary comparison

- Coverage levels are 100%, 90%, 80% and 70%. The 80% level is primary.
- Exact-coverage ranking: keep the n lowest-scoring forecasts, with n = round(coverage x N), halves rounded up. For 80% of the 399 test women, n = 319. Ties are broken by record ID, ascending.
- Metric: mean absolute error (MAE) of the kept forecasts, in g/dL. RMSE, mean signed error, kept counts and achieved coverage are reported too.
- Primary result: learned minus tree disagreement at 80% coverage. A negative value favours the learned score. The 95% interval is a percentile interval from a paired bootstrap of 5,000 replicates (seed 42; the minimum is 1,000). Each replicate resamples identical-record groups and re-applies the exact-coverage rule to the resampled women (*proposed* detail).
- Reading the result: an interval that crosses zero shows neither superiority nor equivalence. The study then reports the estimated difference and its uncertainty and does not establish equivalence.

## 7. Secondary and descriptive analyses

- The other paired differences at every coverage level: learned and tree disagreement against the simple rule, and each against keeping every forecast.
- Who is kept at 80% coverage, by first haemoglobin (below 10, 10 to below 11, 11 and above), complete versus incomplete inputs, follow-up gap (up to 2, 3 to 4, 5 to 8, over 8 weeks; *proposed* bands) and observed follow-up haemoglobin. Descriptive only. These groups are never used to choose forecasts.
- Complete-case sensitivity: the whole pipeline repeated on women with all ten inputs.
- Stage 1: all five candidates on the test women, and the forest against each simpler model with a paired bootstrap.
- Nothing is refitted after the development comparison. The locked test loads the saved Stage 1 and Stage 2 models. Two refits happen, both on the training women only and with the same fixed settings: the complete-case sensitivity analysis above, which is first checked against the saved development result, and the five Stage 1 candidates, whose forest is first checked against the frozen forest.

## 8. Demonstration cutoff

The demonstration API needs a cutoff because one woman cannot be ranked. It is the 80th percentile of the Stage 2 scores of the development women (*proposed*). It is fitted on development data, so it is a different method from the exact-coverage ranking above and is never mixed with it.

## 9. Limits that apply to every result

Internal validation on one public dataset from five Tanzanian districts, not nationally representative. No verified patient identifier, so identical-vector grouping is only a proxy for the same woman. The target is haemoglobin at the next recorded visit, with gaps of 1 to 28 weeks (median 4). Eligibility requires a repeat test. The Stage 2 score ranks forecasts and is not a calibrated confidence interval. "Set aside" means a research output is not counted in this evaluation. It does not mean withholding care or testing, and clinical usefulness has not been verified.
