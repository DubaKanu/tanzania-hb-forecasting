# Deployment Plan

## Current checkpoint (Initial Software Demonstration)
- **Form:** local FastAPI service (`app/main.py`) with interactive Swagger UI
  at `/docs`. No mobile app, SMS, or hospital integration, per the
  supervisor-approved scope: this is a retrospective computational study, not
  a field deployment.
- **How to run:** `uvicorn app.main:app --reload`, then open
  `http://127.0.0.1:8000/docs`.
- **What it demonstrates:** submitting a first-contact record and receiving a
  Stage 1 haemoglobin forecast, matching the "simple local demonstration"
  described in the capstone's Week 8 deliverable.

## Near-term (Weeks 5-7 of the capstone timeline)
- Add the Stage 2 error estimator to the API response, so each forecast
  returns both a predicted value and a reliability flag (trustworthy /
  flag-for-repeat-test), consistent with the two-stage design in the
  proposal.
- Persist the frozen Stage 1 model (`models/stage1_model.joblib`) produced by
  the notebook, so the API loads a real trained model rather than requiring
  retraining on every restart.

## What is explicitly out of scope for this capstone
- A production mobile application, SMS alerting, or live hospital
  integration. These were removed from the core deliverable during proposal
  review, since an eight-week capstone cannot responsibly support a field
  pilot on top of a rigorous modelling study.
- Any deployment claim about performance outside this specific Tanzanian
  cohort. The API's disclaimer field reflects this on every response.

## Longer-term, if this work continues past the capstone
- Containerize the API (Dockerfile) for easier grading/demo reproducibility.
- Add authentication and request logging before any real health-worker-facing
  pilot would be considered, alongside a formal ethics and data-governance
  review, which is out of scope for this academic capstone.
