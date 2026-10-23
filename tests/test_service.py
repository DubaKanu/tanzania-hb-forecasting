"""
Tests for app/service.py: the forecasting logic behind the API, with no web code involved.

The service receives its loader from outside, so these tests give it a fake forecaster and check
the lazy loading, the single load under concurrent first requests, the cutoff decision and the
handling of missing optional inputs.
"""
import threading
import time

import numpy as np
import pytest

from app.service import (
    MODELS_MISSING_MESSAGE, ForecastService, ModelsNotFoundError, Verdict, load_forecaster_from_disk,
)
from forecaster import ForecastBatch
from pipeline import FEATURE_COLUMNS

SAMPLE = {
    "hb_visit1_g_dl": 10.8, "gestation_visit1_weeks": 20, "maternal_age_years": 28, "pregnancy_count": 2,
    "height_cm": 160, "weight_visit1_kg": 62, "pulse_visit1_bpm": 80, "respiratory_rate_visit1": 18,
    "systolic_bp_visit1": 118, "diastolic_bp_visit1": 76,
}


class StubForecaster:
    """Stands in for TwoStageForecaster with fixed numbers, and remembers the rows it was given."""

    def __init__(self, forecast=11.2, estimate=0.4, cutoff=0.5):
        self._forecast, self._estimate, self.demo_cutoff = forecast, estimate, cutoff
        self.rows = []

    def forecast(self, df):
        self.rows.append(df)
        return ForecastBatch(np.array([self._forecast]), np.array([0.1]), np.array([self._estimate]))

    def is_kept(self, estimate):
        return estimate <= self.demo_cutoff


def _counting_loader(forecaster, delay=0.0):
    calls = []

    def loader():
        calls.append(1)
        time.sleep(delay)
        return forecaster
    return loader, calls


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------
def test_nothing_is_loaded_until_the_first_request():
    loader, calls = _counting_loader(StubForecaster())
    service = ForecastService(loader)
    assert service.is_loaded is False and calls == []
    service.forecast(SAMPLE)
    assert service.is_loaded is True and len(calls) == 1


def test_the_models_are_loaded_once_however_many_requests_follow():
    loader, calls = _counting_loader(StubForecaster())
    service = ForecastService(loader)
    for _ in range(25):
        service.forecast(SAMPLE)
    assert len(calls) == 1


def test_two_requests_arriving_together_load_the_models_once():
    loader, calls = _counting_loader(StubForecaster(), delay=0.05)
    service = ForecastService(loader)
    results = []
    threads = [threading.Thread(target=lambda: results.append(service.forecast(SAMPLE))) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and len(results) == 16
    assert len({r.forecast_g_dl for r in results}) == 1


def test_a_failed_load_is_not_remembered_so_the_next_request_can_succeed():
    state = {"files_exist": False}

    def loader():
        if not state["files_exist"]:
            raise ModelsNotFoundError(MODELS_MISSING_MESSAGE)
        return StubForecaster()

    service = ForecastService(loader)
    with pytest.raises(ModelsNotFoundError):
        service.forecast(SAMPLE)
    assert service.is_loaded is False
    state["files_exist"] = True
    assert service.forecast(SAMPLE).forecast_g_dl == pytest.approx(11.2) and service.is_loaded


def test_the_default_loader_names_the_missing_files_and_how_to_make_them(tmp_path):
    with pytest.raises(ModelsNotFoundError) as error:
        load_forecaster_from_disk(str(tmp_path / "no1.joblib"), str(tmp_path / "no2.joblib"))
    assert "python src/run_stage1.py" in str(error.value) and "python src/run_stage2.py" in str(error.value)


# ---------------------------------------------------------------------------
# What the service decides
# ---------------------------------------------------------------------------
def test_the_verdict_carries_the_forecast_the_estimate_and_the_cutoff():
    service = ForecastService(lambda: StubForecaster(forecast=12.34, estimate=0.25, cutoff=0.5))
    verdict = service.forecast(SAMPLE)
    assert verdict == Verdict(forecast_g_dl=12.34, estimated_abs_error_g_dl=0.25, cutoff_g_dl=0.5, kept=True)


@pytest.mark.parametrize("estimate, kept", [(0.49, True), (0.5, True), (0.5000001, False), (2.0, False)])
def test_a_forecast_is_kept_at_or_below_the_cutoff_and_set_aside_above_it(estimate, kept):
    service = ForecastService(lambda: StubForecaster(estimate=estimate, cutoff=0.5))
    assert service.forecast(SAMPLE).kept is kept


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def test_the_row_given_to_the_forecaster_has_the_ten_columns_in_the_fixed_order_as_floats():
    stub = StubForecaster()
    ForecastService(lambda: stub).forecast(dict(reversed(list(SAMPLE.items()))))   # keys in another order
    row = stub.rows[0]
    assert list(row.columns) == FEATURE_COLUMNS and len(row) == 1
    assert all(np.issubdtype(kind, np.floating) for kind in row.dtypes)


def test_none_inputs_reach_the_forecaster_as_missing_values():
    stub = StubForecaster()
    optional = ["pulse_visit1_bpm", "respiratory_rate_visit1", "systolic_bp_visit1", "diastolic_bp_visit1"]
    ForecastService(lambda: stub).forecast({**SAMPLE, **{c: None for c in optional}})
    row = stub.rows[0]
    assert row[optional].isna().to_numpy().all()
    assert row.drop(columns=optional).notna().to_numpy().all()


def test_a_missing_key_is_an_error_and_not_a_silent_blank():
    features = {k: v for k, v in SAMPLE.items() if k != "hb_visit1_g_dl"}
    with pytest.raises(ValueError, match="hb_visit1_g_dl"):
        ForecastService(lambda: StubForecaster()).forecast(features)


def test_the_inputs_are_not_changed_or_kept_by_the_service():
    stub = StubForecaster()
    features = dict(SAMPLE)
    service = ForecastService(lambda: stub)
    service.forecast(features)
    assert features == SAMPLE
    assert not any(isinstance(v, dict) and "hb_visit1_g_dl" in v for v in vars(service).values())
