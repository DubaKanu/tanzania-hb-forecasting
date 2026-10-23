"""
The forecasting logic behind the API, with no web code in it.

ForecastService receives a loader (a function that returns a TwoStageForecaster) instead of
reading files itself, so the web layer and the model files can each change, or be replaced by
a fake in a test, without touching this class (dependency inversion). The models are loaded
on first use, so the API starts even when the model files do not exist yet, and loading is
guarded by a lock so two requests arriving together load them once.

Nothing here writes a file, logs an input or keeps a request: the service holds only the
loaded models.
"""

import os
import sys
import threading
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "src"))  # the modelling code lives in src/
from forecaster import TwoStageForecaster  # noqa: E402
from pipeline import FEATURE_COLUMNS  # noqa: E402

STAGE1_PATH = os.path.join(HERE, "..", "models", "stage1_model.joblib")
STAGE2_PATH = os.path.join(HERE, "..", "models", "stage2_model.joblib")

MODELS_MISSING_MESSAGE = (
    "Model files not found. From the repository root run "
    "'python src/run_stage1.py' and then 'python src/run_stage2.py', "
    "and restart this API."
)


class ModelsNotFoundError(RuntimeError):
    """The saved model files are not there yet."""


@dataclass(frozen=True)
class Verdict:
    """The service's answer for one woman, before any rounding or wording."""

    forecast_g_dl: float
    estimated_abs_error_g_dl: float
    cutoff_g_dl: float
    kept: bool


def load_forecaster_from_disk(stage1_path: str = STAGE1_PATH, stage2_path: str = STAGE2_PATH) -> TwoStageForecaster:
    """The default loader: the two saved model files in models/."""
    if not (os.path.exists(stage1_path) and os.path.exists(stage2_path)):
        raise ModelsNotFoundError(MODELS_MISSING_MESSAGE)
    return TwoStageForecaster.from_files(stage1_path, stage2_path)


class ForecastService:
    """Forecast one woman's next-visit haemoglobin and say whether the research cutoff would keep it."""

    def __init__(self, loader: Callable[[], TwoStageForecaster]):
        """`loader` returns the forecaster. It is called at most once, on the first forecast."""
        self._loader = loader
        self._forecaster: Optional[TwoStageForecaster] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """True once the models have been read from disk."""
        return self._forecaster is not None

    def _get_forecaster(self) -> TwoStageForecaster:
        """The forecaster, loaded on the first call. The lock stops two requests loading it at once."""
        if self._forecaster is None:
            with self._lock:
                if self._forecaster is None:  # another request may have loaded it while this one waited
                    self._forecaster = self._loader()
        return self._forecaster

    def forecast(self, features: Mapping[str, Optional[float]]) -> Verdict:
        """Forecast for one woman given her ten first-visit values (the optional ones may be None).

        Every one of the ten keys must be present; a key left out is an error, not a blank.
        """
        missing = [c for c in FEATURE_COLUMNS if c not in features]
        if missing:
            raise ValueError(f"missing input values: {missing}")
        forecaster = self._get_forecaster()
        row = pd.DataFrame([dict(features)])[FEATURE_COLUMNS].astype(float)
        batch = forecaster.forecast(row)
        estimate = float(batch.estimated_abs_error[0])
        return Verdict(
            forecast_g_dl=float(batch.forecast[0]),
            estimated_abs_error_g_dl=estimate,
            cutoff_g_dl=float(forecaster.demo_cutoff),
            kept=forecaster.is_kept(estimate),
        )
