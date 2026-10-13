# Tanzania Antenatal Haemoglobin Forecasting

Capstone project for my Software Engineering degree at ALU.
Supervisor: Emmanuel Adjei

GitHub repo: https://github.com/DubaKanu/tanzania-hb-forecasting

Video demo: https://drive.google.com/file/d/1hme7EPQWwJKcGfEDZ_sLhvhIecOXRt67/view?usp=sharing

## What this does

Predicts a pregnant woman's haemoglobin level at her next antenatal visit, using her first-visit data (age, pregnancy history, blood pressure, pulse, and so on). The second stage, which will estimate how reliable each prediction is so that a low-confidence one can be flagged for a repeat blood test, is the next piece of work and is not built yet.

This is a research demo, not a medical tool. It doesn't replace blood testing.

## Data

Real data from the Tanzania Maternal Health Risks Stratification dataset (Zenodo, DOI 10.5281/zenodo.15309733, Version 1, CC BY 4.0). After cleaning and keeping only women with two recorded visits, 2,685 women are used. They are split into 1,884 for training, 402 for development and 399 reserved for a final test that has not been used yet. Women with an identical set of first-visit values are kept in the same group, though without a patient ID this cannot prove the groups are independent.

`src/audit.py` reproduces the 2,685 count, every exclusion step, the missing values and the split sizes from the raw file, and checks the file's checksum.

## Results so far

Average error in g/dL on the 402 development women (lower is better):

| Model | Error (g/dL) |
|---|---|
| Random Forest | 0.794 |
| Ridge regression | 0.807 |
| First haemoglobin and gestational week only | 0.811 |
| Carry previous reading forward | 0.926 |
| Just guess the average | 1.080 |

Random Forest performs best, but the model that uses only first haemoglobin and gestational week is close, so the other inputs add little so far. Full charts and results are in `notebooks/model_notebook.ipynb`.

## How to run it

1. `pip install -r requirements.txt`
2. Download `maternal_dataset_csv.csv` from the Zenodo link above and put it in `data/raw/`
3. `python src/audit.py` checks the data and reproduces the counts
4. `python src/run_stage1.py` prepares the data, compares the models and saves the chosen one
5. Open and run `notebooks/model_notebook.ipynb` to see the charts
6. To try the demo API: `python3 -m uvicorn app.main:app --reload`, then open `http://127.0.0.1:8000/docs`

## Designs

Four charts are generated from the real data and saved in `notebooks/`: missingness per feature, first-visit vs. second-visit haemoglobin, the gap in weeks between visits, and the model comparison chart. A screenshot of the Swagger interface is included in this repo, showing a sample request and its returned prediction. It is the closest thing to an interface screenshot for this project.

## Deployment plan

Right now: a small local API (`app/main.py`) that takes a woman's first-visit numbers and returns a predicted haemoglobin reading. This is the MVP, just enough to show input going in and a real output coming out. No app, no SMS, no hospital system, and that is intentional at this stage. If this continued past the capstone, the next step would be wrapping this same model behind a simple web or mobile form so a health worker could use it directly, deployed somewhere like Render.

## What's in here

- `src/`: data cleaning, model code, and the scripts `run_stage1.py` and `audit.py`
- `notebooks/`: the main notebook with charts and results
- `app/`: a small API that serves predictions
- `tests/`: unit tests
- `models/`: the saved Stage 1 model
- `reports/`: the audit report and the model results
- `data/`: the prepared data files and split manifest (the raw dataset is not included because it is too large)
