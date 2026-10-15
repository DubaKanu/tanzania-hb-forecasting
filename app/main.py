"""
Local demonstration API for the Tanzania haemoglobin forecasting pipeline.

Given a woman's first-visit values, it returns
  1. a forecast of her haemoglobin at her next recorded visit (Stage 1),
  2. an estimate of how far that forecast is likely to be from the real value (Stage 2), and
  3. whether the forecast would be kept or set aside by the research cutoff.

This is a research demonstration, not a clinical tool, and does not replace blood
testing. It does not store anything that is sent to it.

Run with:
    python3 -m uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

import os
import sys
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import FEATURE_COLUMNS  # noqa: E402
from stage2 import forecast_with_reliability  # noqa: E402

app = FastAPI(
    title="Tanzania Antenatal Hb Forecast: Research Demo",
    description=(
        "Retrospective research demonstration only. Forecasts a woman's haemoglobin at her next "
        "recorded antenatal visit, estimates how large that forecast's error is likely to be, and "
        "says whether the forecast would be kept or set aside by the research cutoff. Not a "
        "diagnosis and does not replace clinical blood testing. Nothing sent to this service is stored."
    ),
    version="0.2.0",
)

HERE = os.path.dirname(__file__)
STAGE1_PATH = os.path.join(HERE, "..", "models", "stage1_model.joblib")
STAGE2_PATH = os.path.join(HERE, "..", "models", "stage2_model.joblib")
_models = None  # loaded on first use, so the API still starts before the models exist


class VisitOneFeatures(BaseModel):
    hb_visit1_g_dl: float = Field(..., ge=3, le=20, description="Haemoglobin at first contact (g/dL)")
    gestation_visit1_weeks: float = Field(..., ge=1, le=42)
    maternal_age_years: float = Field(..., ge=10, le=60)
    pregnancy_count: float = Field(..., ge=0)
    height_cm: float = Field(..., ge=100, le=200)
    weight_visit1_kg: float = Field(..., ge=30, le=150)
    pulse_visit1_bpm: Optional[float] = Field(None, description="Optional; imputed if missing")
    respiratory_rate_visit1: Optional[float] = Field(None, description="Optional; imputed if missing")
    systolic_bp_visit1: Optional[float] = Field(None, description="Optional; imputed if missing")
    diastolic_bp_visit1: Optional[float] = Field(None, description="Optional; imputed if missing")


class ForecastResponse(BaseModel):
    predicted_hb_visit2_g_dl: float = Field(
        ..., description="Stage 1 forecast of haemoglobin at the next recorded visit (g/dL)")
    estimated_abs_error_g_dl: float = Field(
        ..., description="Stage 2 estimate of how far the forecast is likely to be from the real value "
                         "(g/dL). A score for ranking forecasts, not a confidence interval.")
    reliability_flag: str = Field(
        ..., description="'kept' if the estimated error is at or below the cutoff, otherwise 'set_aside'")
    cutoff_g_dl: float = Field(..., description="The research cutoff for the estimated error (g/dL)")
    reliability_note: str
    disclaimer: str = (
        "Research demonstration output only. Not a diagnosis. "
        "Does not replace clinical blood testing."
    )


def _load_models():
    global _models
    if _models is None:
        if not (os.path.exists(STAGE1_PATH) and os.path.exists(STAGE2_PATH)):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model files not found. From the repository root run "
                    "'python src/run_stage1.py' and then 'python src/run_stage2.py', "
                    "and restart this API."
                ),
            )
        _models = (joblib.load(STAGE1_PATH), joblib.load(STAGE2_PATH))
    return _models


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _models is not None}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(features: VisitOneFeatures):
    """Forecast next-visit haemoglobin, estimate the forecast's error and say whether it is kept.

    The cutoff is the 80th percentile of the Stage 2 scores of the development women. It is
    fitted on development data, so it is a different method from the exact-coverage ranking
    used in the evaluation. One woman cannot be ranked, which is why a cutoff is needed here.
    """
    bundle1, bundle2 = _load_models()
    row = pd.DataFrame([features.model_dump()])[FEATURE_COLUMNS].astype(float)
    pred, _, estimate = forecast_with_reliability(bundle1, bundle2["model"], row)

    est = float(estimate[0])
    cutoff = float(bundle2["demo_cutoff"])
    kept = est <= cutoff
    verdict = ("at or below the research cutoff, so this forecast would be kept"
               if kept else "above the research cutoff, so this forecast would be set aside as less reliable")
    return ForecastResponse(
        predicted_hb_visit2_g_dl=round(float(pred[0]), 2),
        estimated_abs_error_g_dl=round(est, 3),
        reliability_flag="kept" if kept else "set_aside",
        cutoff_g_dl=round(cutoff, 3),
        reliability_note=(
            f"The estimated error of {est:.2f} g/dL is {verdict} in the research evaluation. The cutoff is "
            "the 80th percentile of the development women's scores. It is fitted on development data and "
            "is different from the exact-coverage rule used in the evaluation."
        ),
    )
