# Tanzania Antenatal Haemoglobin Forecasting (Initial Demo)

## Status: run against the real data
This has been executed end-to-end against the actual Zenodo file
(`maternal_dataset_csv.csv`, md5 `0253d2d385dd80d6c9bb82df6b85fcc8`, the
version pinned in the supervisor's guidance). Real results, not placeholders:

- **Eligible cohort:** 2,685 women — matches the supervisor's audited figure exactly.
- **Stage 1 training CV MAE (g/dL):** random_forest 0.760 < ridge 0.802 < carry_forward 0.879 < median 1.134
- **Development-set MAE (g/dL):** random_forest 0.793 < ridge 0.807 < carry_forward 0.926 < median 1.080
- Full output, including the four generated charts, is embedded in `notebooks/model_notebook.ipynb`.
- The reserved test set (n=399) has **not** been touched, per the frozen protocol; that happens once, in Week 7.
- A trained Stage 1 model is saved at `models/stage1_model.joblib` and served by the demo API.

## Description
A two-stage machine learning pipeline that (1) forecasts a pregnant woman's
next antenatal haemoglobin reading from her first-contact data, and (2)
estimates how reliable that individual forecast is, so a forecast can be
flagged for direct re-measurement rather than trusted blindly. This is the
initial-demo checkpoint of the ALU Software Engineering capstone
*"Predicting Follow-up Haemoglobin and Identifying Unreliable Predictions in
a Tanzanian Antenatal Cohort"* (Supervisor: Emmanuel Adjei).

This is a **retrospective research demonstration**, not a clinical tool. It
does not replace blood testing and does not assess referral effectiveness.

## Repository link
`[paste your GitHub repo URL here after you push]`

## About the prepared-data folder name
The supervisor's guidance names a folder, `Josephine_African_Data_Review/
Tanzania_Followup_Haemoglobin`, holding `model_ready.csv`, `train.csv`,
`validation.csv`, and `test.csv`. Those files did not exist anywhere public;
they are the supervisor's own already-processed version of this same raw
Zenodo file. This repo builds an independent version of that exact folder
under `data/Josephine_African_Data_Review/Tanzania_Followup_Haemoglobin/`,
derived from the raw file via `src/data_adapter.py` (field mapping) and
`src/pipeline.py` (eligibility rules) — and its eligible count (2,685)
matches the supervisor's own figure exactly. If the supervisor shares his
exact files directly, drop them into that same folder; either version is
usable, but using his exact split lets you reproduce his own preliminary
numbers precisely, which is worth doing as a cross-check.

## What's in this repo
```
src/data_adapter.py       Maps the raw 683-column Zenodo export onto the model's 10-feature contract
src/pipeline.py           Core pipeline: eligibility rules, fixed split, Stage 1 models, metrics
notebooks/model_notebook.ipynb   Data viz, model architecture, initial performance metrics (already run)
app/main.py               FastAPI demo (Swagger UI) serving the real trained Stage 1 model
models/stage1_model.joblib  The trained, frozen Stage 1 model (random forest)
tests/test_pipeline.py    Unit tests for the pipeline logic (5 tests, all passing)
data/raw/                       Place maternal_dataset_csv.csv here (not committed to git, 15 MB)
data/Josephine_African_Data_Review/Tanzania_Followup_Haemoglobin/
                                 Derived model_ready.csv / train.csv / validation.csv / test.csv
                                 + exclusion_log.txt (the audit trail)
requirements.txt
deployment_plan.md
```

## How to set up the environment and run the project

1. **Clone and install dependencies**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Get the data.** Download the pinned Tanzania MHRS release (Zenodo DOI
   `10.5281/zenodo.15309733`), file `maternal_dataset_csv.csv`, and place it
   in `data/raw/`. Verify its MD5 checksum matches
   `0253d2d385dd80d6c9bb82df6b85fcc8` before use.

3. **Run the tests** (sanity-checks the pipeline code itself, not the real results):
   ```bash
   pytest tests/ -v
   ```

4. **Run the notebook** to regenerate the derived data, the visualizations, the
   Stage 1 model comparison, and the metrics table:
   ```bash
   jupyter notebook notebooks/model_notebook.ipynb
   ```
   Run all cells top to bottom. With the fixed seed (42), this reproduces the
   same 2,685-row cohort and the same splits reported above. **Do not open
   the reserved test partition again until the final Week 7 evaluation.**

5. **Run the demo API**:
   ```bash
   uvicorn app.main:app --reload
   ```
   Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI and try
   the `/forecast` endpoint — it's backed by the real trained model in
   `models/stage1_model.joblib`, already included in this repo.

## Designs
See `notebooks/fig_*.png`: missingness by feature, first- vs second-contact
haemoglobin, follow-up gap distribution, and the Stage 1 model comparison
chart, all generated from the real cohort. The Swagger UI at `/docs` serves
as the interface design for the deployment mockup, since this demo is an API
rather than a mobile/web app.

## Deployment plan
See `deployment_plan.md`.
