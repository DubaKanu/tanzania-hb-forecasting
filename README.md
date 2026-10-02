# Tanzania Antenatal Haemoglobin Forecasting

Capstone project for my Software Engineering degree at ALU.
Supervisor: Emmanuel Adjei

GitHub repo: PASTE_YOUR_REPO_URL_HERE

## What this does

Predicts a pregnant woman's haemoglobin level at her next antenatal visit, using her first-visit data (age, pregnancy history, blood pressure, pulse, and so on). A second model flags how reliable each prediction is, so a low-confidence one can be flagged for a repeat blood test instead of being trusted blindly.

This is a research demo, not a medical tool. It doesn't replace blood testing.

## Data

Real data from the Tanzania Maternal Health Risks Stratification dataset (Zenodo, DOI 10.5281/zenodo.15309733). After cleaning and keeping only women with two recorded visits, 2,685 women are used.

## Results so far

| Model | Error (g/dL) |
|---|---|
| Random Forest | 0.793 |
| Ridge regression | 0.807 |
| Carry previous reading forward | 0.926 |
| Just guess the average | 1.080 |

Random Forest performs best. Full charts and results are in `notebooks/model_notebook.ipynb`.

## How to run it

1. `pip install -r requirements.txt`
2. Download `maternal_dataset_csv.csv` from the Zenodo link above and put it in `data/raw/`
3. Open and run `notebooks/model_notebook.ipynb`
4. To try the demo API: `uvicorn app.main:app --reload`, then open `http://127.0.0.1:8000/docs`

## Designs

Four charts are generated from the real data and saved in `notebooks/`: missingness per feature, first-visit vs. second-visit haemoglobin, the gap in weeks between visits, and the model comparison chart above. These stand in for interface mockups since the deliverable here is a prediction model, not an app screen. The API's Swagger page at `/docs` is the closest thing to an interface, it's what a screenshot of the app in action would show.

## Deployment plan

Right now: a small local API (`app/main.py`) that takes a woman's first-visit numbers and returns a predicted haemoglobin reading, this is the MVP, just enough to show input going in and a real output coming out. No app, no SMS, no hospital system, that's intentional at this stage. If this continued past the capstone, the next step would be wrapping this same model behind a simple web or mobile form so a health worker could use it directly, deployed somewhere like Render.

## What's in here

- `src/` — data cleaning and model code
- `notebooks/` — the main notebook with charts and results
- `app/` — a small API that serves predictions
- `tests/` — unit tests
- `data/` — where the dataset goes (not included here, too large)
