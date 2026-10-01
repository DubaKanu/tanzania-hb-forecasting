"""
Local demonstration API for the Tanzania haemoglobin forecasting pipeline.

This is the "simple local demonstration" described in the capstone timeline
(Week 8): it shows a predicted next-contact haemoglobin value alongside an
estimated reliability signal for that prediction. It is a research
demonstration, not a clinical tool, and does not replace blood testing.

Run with:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

import sys
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import FEATURE_COLUMNS, _prep_matrix  # noqa: E402

app = FastAPI(
    title="Tanzania Antenatal Hb Forecast — Research Demo",
    description=(
        "Retrospective research demonstration only. Predicts a woman's next "
        "antenatal haemoglobin reading and flags how reliable that forecast "
        "is estimated to be. Does not replace clinical blood testing."
    ),
    version="0.1.0",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "stage1_model.joblib")
_model_bundle = None  # loaded lazily so the API still starts before training is finished


class VisitOneFeatures(BaseModel):
    hb_visit1_g_dl: float = Field(..., ge=3, le=20, description="Haemoglobin at first contact (g/dL)")
    gestation_visit1_weeks: float = Field(..., ge=1, le=42)
    maternal_age_years: float = Field(..., ge=10, le=60)
    pregnancy_count: float = Field(..., ge=0)
    height_cm: float = Field(..., ge=100, le=200)
    weight_visit1_kg: float = Field(..., ge=30, le=150)
    pulse_visit1_bpm: float | None = Field(None, description="Optional; imputed if missing")
    respiratory_rate_visit1: float | None = Field(None, description="Optional; imputed if missing")
    systolic_bp_visit1: float | None = Field(None, description="Optional; imputed if missing")
    diastolic_bp_visit1: float | None = Field(None, description="Optional; imputed if missing")


class ForecastResponse(BaseModel):
    predicted_hb_visit2_g_dl: float
    reliability_note: str
    disclaimer: str = (
        "Research demonstration output only. Not a diagnosis. "
        "Does not replace clinical blood testing."
    )


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_bundle is not None}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(features: VisitOneFeatures):
    """Predict next-contact haemoglobin from first-contact features.

    Placeholder response until the frozen Stage 1 model (trained on the real
    Tanzania cohort) is saved to models/stage1_model.joblib -- see the
    model notebook for training that file.
    """
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=(
                    "No trained model found yet. Run notebooks/model_notebook.ipynb "
                    "against the real Tanzania MHRS data to produce "
                    "models/stage1_model.joblib, then restart this API."
                ),
            )
        _model_bundle = joblib.load(MODEL_PATH)

    import pandas as pd

    row = pd.DataFrame([features.dict()])[FEATURE_COLUMNS]
    X, _ = _prep_matrix(row, imputer=_model_bundle["imputer"], fit=False)
    pred = float(_model_bundle["model"].predict(X)[0])

    return ForecastResponse(
        predicted_hb_visit2_g_dl=round(pred, 2),
        reliability_note=(
            "Stage 2 reliability scoring is trained separately per the capstone "
            "timeline (Weeks 5-7); this endpoint currently returns the Stage 1 "
            "point forecast only."
        ),
    )
