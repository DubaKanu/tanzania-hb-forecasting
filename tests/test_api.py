"""
Acceptance checks for the demonstration API (app/main.py).

Each test is one of the eight checks agreed for the researcher demonstration. They need the
saved models, so run these first from the repository root:

    python src/run_stage1.py
    python src/run_stage2.py
"""
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

MODELS = [os.path.join(ROOT, "models", "stage1_model.joblib"),
          os.path.join(ROOT, "models", "stage2_model.joblib")]
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in MODELS),
    reason="saved models not found: run python src/run_stage1.py and python src/run_stage2.py first",
)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from pipeline import FEATURE_COLUMNS  # noqa: E402
from stage2 import forecast_with_reliability  # noqa: E402

client = TestClient(app)

SAMPLE = {
    "hb_visit1_g_dl": 10.8, "gestation_visit1_weeks": 20, "maternal_age_years": 28, "pregnancy_count": 2,
    "height_cm": 160, "weight_visit1_kg": 62, "pulse_visit1_bpm": 80, "respiratory_rate_visit1": 18,
    "systolic_bp_visit1": 118, "diastolic_bp_visit1": 76,
}
OPTIONAL = ["pulse_visit1_bpm", "respiratory_rate_visit1", "systolic_bp_visit1", "diastolic_bp_visit1"]


def test_check1_valid_input_gives_200_and_a_value_between_3_and_20():
    r = client.post("/forecast", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 3 <= body["predicted_hb_visit2_g_dl"] <= 20
    assert body["reliability_flag"] in ("kept", "set_aside")
    assert body["estimated_abs_error_g_dl"] >= 0


@pytest.mark.parametrize("change", [
    {"hb_visit1_g_dl": 25},            # above the 3 to 20 range
    {"hb_visit1_g_dl": 1},             # below it
    {"gestation_visit1_weeks": 60},
    {"maternal_age_years": 5},
    {"pregnancy_count": -1},
    {"height_cm": 20},
    {"weight_visit1_kg": 500},
    {"hb_visit1_g_dl": "not a number"},
])
def test_check2_out_of_range_input_is_rejected_with_422(change):
    r = client.post("/forecast", json=dict(SAMPLE, **change))
    assert r.status_code == 422


@pytest.mark.parametrize("missing", ["hb_visit1_g_dl", "gestation_visit1_weeks", "maternal_age_years",
                                     "pregnancy_count", "height_cm", "weight_visit1_kg"])
def test_check2_missing_required_input_is_rejected_with_422(missing):
    payload = {k: v for k, v in SAMPLE.items() if k != missing}
    assert client.post("/forecast", json=payload).status_code == 422


def test_check3_the_same_input_always_gives_the_same_output():
    first = client.post("/forecast", json=SAMPLE).json()
    for _ in range(3):
        assert client.post("/forecast", json=SAMPLE).json() == first


def test_check4_output_equals_the_offline_prediction_for_the_same_row():
    bundle1, bundle2 = joblib.load(MODELS[0]), joblib.load(MODELS[1])
    row = pd.DataFrame([SAMPLE])[FEATURE_COLUMNS].astype(float)
    forecast, _, estimate = forecast_with_reliability(bundle1, bundle2["model"], row)
    body = client.post("/forecast", json=SAMPLE).json()
    assert abs(body["predicted_hb_visit2_g_dl"] - float(forecast[0])) <= 0.0051
    assert abs(body["estimated_abs_error_g_dl"] - float(estimate[0])) <= 0.0006


def test_check4_output_matches_the_saved_development_results_for_real_women():
    data = os.path.join(ROOT, "data", "Josephine_African_Data_Review", "Tanzania_Followup_Haemoglobin",
                        "validation.csv")
    saved = os.path.join(ROOT, "reports", "stage2_dev_table.csv")
    if not (os.path.exists(data) and os.path.exists(saved)):
        pytest.skip("development files not found")
    dev = pd.read_csv(data)
    table = pd.read_csv(saved).set_index("record_id")
    for idx in np.linspace(0, len(dev) - 1, 8).astype(int):
        woman = dev.iloc[idx]
        payload = {c: float(woman[c]) for c in FEATURE_COLUMNS if not pd.isna(woman[c])}
        body = client.post("/forecast", json=payload).json()
        offline = table.loc[woman["_source_row"]]
        assert abs(body["predicted_hb_visit2_g_dl"] - offline["forecast"]) <= 0.0051
        assert abs(body["estimated_abs_error_g_dl"] - offline["stage2_score"]) <= 0.0006


def test_check5_leaving_out_the_optional_inputs_still_gives_a_value():
    payload = {k: v for k, v in SAMPLE.items() if k not in OPTIONAL}
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200
    assert 3 <= r.json()["predicted_hb_visit2_g_dl"] <= 20


def test_check6_every_response_carries_the_disclaimer():
    for payload in (SAMPLE, {k: v for k, v in SAMPLE.items() if k not in OPTIONAL},
                    dict(SAMPLE, hb_visit1_g_dl=6.0), dict(SAMPLE, hb_visit1_g_dl=16.0)):
        body = client.post("/forecast", json=payload).json()
        assert "Research demonstration output only" in body["disclaimer"]
        assert "Does not replace clinical blood testing" in body["disclaimer"]


def _snapshot(root):
    state = {}
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache", ".git", "venv", ".ipynb_checkpoints")]
        for name in files:
            path = os.path.join(folder, name)
            stat = os.stat(path)
            state[os.path.relpath(path, root)] = (stat.st_size, stat.st_mtime_ns)
    return state


def test_check7_the_service_stores_no_inputs(caplog):
    before = _snapshot(ROOT)
    odd_value = 13.37
    with caplog.at_level(logging.DEBUG):
        r = client.post("/forecast", json=dict(SAMPLE, hb_visit1_g_dl=odd_value))
    assert r.status_code == 200
    assert _snapshot(ROOT) == before               # nothing was written anywhere in the repository
    assert str(odd_value) not in caplog.text       # and the input was not logged
    with open(os.path.join(ROOT, "app", "main.py"), encoding="utf8") as f:
        source = f.read()
    for writer in ("open(", ".write(", "to_csv", "joblib.dump", "pickle.dump", "sqlite", "logging"):
        assert writer not in source


def test_health_and_root_respond():
    assert client.get("/").json()["status"] == "ok"
    assert client.get("/health").json()["status"] == "ok"
