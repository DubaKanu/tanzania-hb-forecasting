"""
API behaviour that does not need the saved models: the 503 when they are missing, the health
route before and after the first request, and a bad request never reaching the models.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_service
from app.service import MODELS_MISSING_MESSAGE, ForecastService, ModelsNotFoundError
from forecaster import ForecastBatch

SAMPLE = {
    "hb_visit1_g_dl": 10.8, "gestation_visit1_weeks": 20, "maternal_age_years": 28, "pregnancy_count": 2,
    "height_cm": 160, "weight_visit1_kg": 62,
}


class StubForecaster:
    demo_cutoff = 0.6

    def forecast(self, df):
        return ForecastBatch(np.array([11.0]), np.array([0.1]), np.array([0.4]))

    def is_kept(self, estimate):
        return estimate <= self.demo_cutoff


@pytest.fixture
def client_with():
    """A client whose service is replaced for one test and restored afterwards."""
    def make(service):
        app.dependency_overrides[get_service] = lambda: service
        return TestClient(app, raise_server_exceptions=False)
    yield make
    app.dependency_overrides.clear()


def _missing_models_loader():
    raise ModelsNotFoundError(MODELS_MISSING_MESSAGE)


def test_missing_model_files_give_503_with_instructions(client_with):
    client = client_with(ForecastService(_missing_models_loader))
    r = client.post("/forecast", json=SAMPLE)
    assert r.status_code == 503
    assert "python src/run_stage1.py" in r.json()["detail"]


def test_the_api_still_starts_and_answers_health_when_the_models_are_missing(client_with):
    client = client_with(ForecastService(_missing_models_loader))
    assert client.get("/").json()["status"] == "ok"
    assert client.get("/health").json() == {"status": "ok", "model_loaded": False}


def test_health_shows_the_models_loaded_only_after_the_first_forecast(client_with):
    client = client_with(ForecastService(StubForecaster))
    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/forecast", json=SAMPLE).status_code == 200
    assert client.get("/health").json()["model_loaded"] is True


def test_a_bad_request_is_refused_before_the_models_are_touched(client_with):
    calls = []
    service = ForecastService(lambda: calls.append(1) or StubForecaster())
    client = client_with(service)
    assert client.post("/forecast", json=dict(SAMPLE, hb_visit1_g_dl=99)).status_code == 422
    assert calls == [] and service.is_loaded is False


def test_the_response_is_built_from_the_service_verdict(client_with):
    client = client_with(ForecastService(StubForecaster))
    body = client.post("/forecast", json=SAMPLE).json()
    assert body["predicted_hb_visit2_g_dl"] == 11.0 and body["estimated_abs_error_g_dl"] == 0.4
    assert body["reliability_flag"] == "kept" and body["cutoff_g_dl"] == 0.6
    assert "Not a diagnosis" in body["disclaimer"]


def test_nan_in_a_request_is_a_422_even_when_no_models_exist(client_with):
    client = client_with(ForecastService(_missing_models_loader))
    text = '{"hb_visit1_g_dl": NaN, "gestation_visit1_weeks": 20, "maternal_age_years": 28, "pregnancy_count": 2, ' \
           '"height_cm": 160, "weight_visit1_kg": 62}'
    r = client.post("/forecast", content=text, headers={"content-type": "application/json"})
    assert r.status_code == 422
