"""
What the demonstration API accepts and returns.

Pydantic checks every request before any model is touched, so a value outside the allowed
range is rejected with a 422 and never reaches the forecaster. This module has no model code.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.service import Verdict


class VisitOneFeatures(BaseModel):
    """The ten first-visit values. The first six are required; the last four may be left out.

    Every value has a lower and an upper limit, and NaN and Infinity are refused. The limits for
    the four optional values and for the number of pregnancies are wide plausibility limits, not
    clinical thresholds (every woman in the training and development data is inside them). They
    exist so that an impossible number gets a 422 instead of reaching the model. To leave an
    optional value out, omit it or send null.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    hb_visit1_g_dl: float = Field(..., ge=3, le=20, description="Haemoglobin at first contact (g/dL)")
    gestation_visit1_weeks: float = Field(..., ge=1, le=42)
    maternal_age_years: float = Field(..., ge=10, le=60)
    pregnancy_count: float = Field(..., ge=0, le=25)
    height_cm: float = Field(..., ge=100, le=200)
    weight_visit1_kg: float = Field(..., ge=30, le=150)
    pulse_visit1_bpm: Optional[float] = Field(None, ge=20, le=250, description="Optional; imputed if missing")
    respiratory_rate_visit1: Optional[float] = Field(None, ge=4, le=120, description="Optional; imputed if missing")
    systolic_bp_visit1: Optional[float] = Field(None, ge=40, le=300, description="Optional; imputed if missing")
    diastolic_bp_visit1: Optional[float] = Field(None, ge=10, le=200, description="Optional; imputed if missing")


class ForecastResponse(BaseModel):
    """The forecast, how far off it is likely to be, and whether the research cutoff would keep it."""

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

    @classmethod
    def from_verdict(cls, verdict: Verdict) -> "ForecastResponse":
        """Turn the service's result into the response: rounding and the plain-language note live here."""
        verdict_text = ("at or below the research cutoff, so this forecast would be kept"
                        if verdict.kept else
                        "above the research cutoff, so this forecast would be set aside as less reliable")
        return cls(
            predicted_hb_visit2_g_dl=round(verdict.forecast_g_dl, 2),
            estimated_abs_error_g_dl=round(verdict.estimated_abs_error_g_dl, 3),
            reliability_flag="kept" if verdict.kept else "set_aside",
            cutoff_g_dl=round(verdict.cutoff_g_dl, 3),
            reliability_note=(
                f"The estimated error of {verdict.estimated_abs_error_g_dl:.2f} g/dL is {verdict_text} in the "
                "research evaluation. The cutoff is the 80th percentile of the development women's scores. "
                "It is fitted on development data and is different from the exact-coverage rule used in the "
                "evaluation."
            ),
        )
