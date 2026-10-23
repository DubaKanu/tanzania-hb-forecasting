"""
Local demonstration API for the Tanzania haemoglobin forecasting pipeline.

Given a woman's first-visit values, it returns
  1. a forecast of her haemoglobin at her next recorded visit (Stage 1),
  2. an estimate of how far that forecast is likely to be from the real value (Stage 2), and
  3. whether the forecast would be kept or set aside by the research cutoff.

This is a research demonstration, not a clinical tool, and does not replace blood
testing. It does not store anything that is sent to it.

This file only holds the routes. The checks on the input are in schemas.py, the error
response in errors.py and the forecasting in service.py.

Run with:
    python3 -m uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError

from app.errors import validation_error_handler
from app.schemas import ForecastResponse, VisitOneFeatures
from app.service import ForecastService, ModelsNotFoundError, load_forecaster_from_disk

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

app.add_exception_handler(RequestValidationError, validation_error_handler)

_service = ForecastService(load_forecaster_from_disk)  # models are read on the first request


def get_service() -> ForecastService:
    """The one service the routes share. Tests replace it with app.dependency_overrides."""
    return _service


@app.get("/")
def root():
    """Say that the service is up and where the interactive documentation is."""
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health(service: ForecastService = Depends(get_service)):
    """Say that the service is up and whether the two model files have been loaded yet."""
    return {"status": "ok", "model_loaded": service.is_loaded}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(features: VisitOneFeatures, service: ForecastService = Depends(get_service)):
    """Forecast next-visit haemoglobin, estimate the forecast's error and say whether it is kept.

    The cutoff is the 80th percentile of the Stage 2 scores of the development women. It is
    fitted on development data, so it is a different method from the exact-coverage ranking
    used in the evaluation. One woman cannot be ranked, which is why a cutoff is needed here.
    """
    try:
        verdict = service.forecast(features.model_dump())
    except ModelsNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return ForecastResponse.from_verdict(verdict)
