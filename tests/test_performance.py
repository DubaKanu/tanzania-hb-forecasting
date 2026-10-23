"""
Performance smoke tests. They are deliberately generous so that they pass on a slow machine and
fail only if something is badly wrong, such as the models being read from disk on every request
or the trees being asked one by one through scikit-learn's checked predict() again (about seven
times slower for one woman). The measured figures are in docs/performance.md, made with
scripts/benchmark_api.py. They need the saved models.
"""
import os
import statistics
import time

import joblib
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = [os.path.join(ROOT, "models", "stage1_model.joblib"), os.path.join(ROOT, "models", "stage2_model.joblib")]
DEV = os.path.join(ROOT, "data", "Josephine_African_Data_Review", "Tanzania_Followup_Haemoglobin", "validation.csv")
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in MODELS),
    reason="saved models not found: run python src/run_stage1.py and python src/run_stage2.py first")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, get_service  # noqa: E402
from app.service import ForecastService, load_forecaster_from_disk  # noqa: E402
from forecaster import TwoStageForecaster  # noqa: E402

SAMPLE = {
    "hb_visit1_g_dl": 10.8, "gestation_visit1_weeks": 20, "maternal_age_years": 28, "pregnancy_count": 2,
    "height_cm": 160, "weight_visit1_kg": 62, "pulse_visit1_bpm": 80, "respiratory_rate_visit1": 18,
    "systolic_bp_visit1": 118, "diastolic_bp_visit1": 76,
}


@pytest.fixture
def fresh_client():
    service = ForecastService(load_forecaster_from_disk)     # a new service, so the first request really loads
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_the_models_are_read_from_disk_once_not_once_per_request(fresh_client, monkeypatch):
    reads = []
    real_load = joblib.load
    monkeypatch.setattr(joblib, "load", lambda path, *a, **k: reads.append(path) or real_load(path, *a, **k))
    for _ in range(10):
        assert fresh_client.post("/forecast", json=SAMPLE).status_code == 200
    assert len(reads) == 2        # the two model files, on the first request only


def test_single_requests_are_fast_enough_for_a_demonstration(fresh_client):
    for _ in range(3):
        fresh_client.post("/forecast", json=SAMPLE)               # warm-up
    times = []
    for _ in range(40):
        start = time.perf_counter()
        assert fresh_client.post("/forecast", json=SAMPLE).status_code == 200
        times.append(time.perf_counter() - start)
    times.sort()
    assert statistics.median(times) < 0.1
    assert times[int(0.95 * len(times))] < 0.3


def test_the_whole_development_set_is_forecast_in_about_a_second():
    if not os.path.exists(DEV):
        pytest.skip("development file not found")
    dev = pd.read_csv(DEV)
    forecaster = TwoStageForecaster.from_files(*MODELS)
    start = time.perf_counter()
    batch = forecaster.forecast(dev)
    elapsed = time.perf_counter() - start
    assert len(batch.forecast) == len(dev) == 402
    assert elapsed < 2
