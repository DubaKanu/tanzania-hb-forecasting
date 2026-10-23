"""
The demonstration API tested with many different inputs: boundary values, values just outside
them, NaN and Infinity, missing and null values, malformed requests, and every pattern of left-out
optional inputs. They need the saved models (python src/run_stage1.py, then python src/run_stage2.py).

Several of these cases returned HTTP 500 in the first version of the API (NaN, Infinity or a
huge number reached the model or the error response). They are kept as regression tests.
"""
import itertools
import os

import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = [os.path.join(ROOT, "models", "stage1_model.joblib"), os.path.join(ROOT, "models", "stage2_model.joblib")]
DATA = os.path.join(ROOT, "data", "Josephine_African_Data_Review", "Tanzania_Followup_Haemoglobin")
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in MODELS),
    reason="saved models not found: run python src/run_stage1.py and python src/run_stage2.py first")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.schemas import VisitOneFeatures  # noqa: E402

# raise_server_exceptions=False so that a crash inside the app shows up as a 500 status, as it would for a user
client = TestClient(app, raise_server_exceptions=False)

SAMPLE = {
    "hb_visit1_g_dl": 10.8, "gestation_visit1_weeks": 20, "maternal_age_years": 28, "pregnancy_count": 2,
    "height_cm": 160, "weight_visit1_kg": 62, "pulse_visit1_bpm": 80, "respiratory_rate_visit1": 18,
    "systolic_bp_visit1": 118, "diastolic_bp_visit1": 76,
}
OPTIONAL = ["pulse_visit1_bpm", "respiratory_rate_visit1", "systolic_bp_visit1", "diastolic_bp_visit1"]
REQUIRED = [k for k in SAMPLE if k not in OPTIONAL]
# (lowest accepted, highest accepted) for every input
LIMITS = {
    "hb_visit1_g_dl": (3, 20), "gestation_visit1_weeks": (1, 42), "maternal_age_years": (10, 60),
    "pregnancy_count": (0, 25), "height_cm": (100, 200), "weight_visit1_kg": (30, 150),
    "pulse_visit1_bpm": (20, 250), "respiratory_rate_visit1": (4, 120),
    "systolic_bp_visit1": (40, 300), "diastolic_bp_visit1": (10, 200),
}
RESPONSE_FIELDS = {"predicted_hb_visit2_g_dl", "estimated_abs_error_g_dl", "reliability_flag", "cutoff_g_dl",
                   "reliability_note", "disclaimer"}


def post_raw(text, content_type="application/json"):
    return client.post("/forecast", content=text, headers={"content-type": content_type})


def raw_body(**changes):
    """A JSON body written by hand, so that NaN and Infinity (which json.dumps refuses by default) can be sent."""
    values = {**SAMPLE, **changes}
    return "{" + ",".join(f'"{k}":{v}' for k, v in values.items()) + "}"


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field, edge", [(f, i) for f in LIMITS for i in (0, 1)])
def test_the_lowest_and_highest_allowed_value_of_every_input_are_accepted(field, edge):
    r = client.post("/forecast", json=dict(SAMPLE, **{field: LIMITS[field][edge]}))
    assert r.status_code == 200, r.text
    assert 3 <= r.json()["predicted_hb_visit2_g_dl"] <= 20


@pytest.mark.parametrize("field, edge, step", [(f, i, s) for f in LIMITS for i, s in ((0, -0.01), (1, 0.01))])
def test_a_value_just_outside_the_limits_of_any_input_is_refused(field, edge, step):
    r = client.post("/forecast", json=dict(SAMPLE, **{field: LIMITS[field][edge] + step}))
    assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# Values that are not ordinary numbers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", list(SAMPLE))
@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_are_refused_with_422_for_every_input(field, token):
    r = post_raw(raw_body(**{field: token}))
    assert r.status_code == 422, f"{field}={token} gave {r.status_code}"
    assert r.json()["detail"][0]["loc"] == ["body", field]


@pytest.mark.parametrize("field", list(SAMPLE))
@pytest.mark.parametrize("huge", ["1e39", "1e300", "-1e300"])
def test_numbers_too_big_for_the_model_are_refused_not_crashed_on(field, huge):
    assert post_raw(raw_body(**{field: huge})).status_code == 422


@pytest.mark.parametrize("value", ['"abc"', "[10]", "{}", "true", '"10,8"', '""'])
def test_values_of_the_wrong_type_are_refused(value):
    assert post_raw(raw_body(hb_visit1_g_dl=value)).status_code == 422


def test_error_messages_do_not_repeat_the_value_that_was_sent():
    r = client.post("/forecast", json=dict(SAMPLE, hb_visit1_g_dl=1234.5678))
    assert r.status_code == 422 and "1234.5678" not in r.text
    assert set(r.json()["detail"][0]) == {"loc", "msg", "type"}


# ---------------------------------------------------------------------------
# Missing and null values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", REQUIRED)
def test_null_for_a_required_input_is_refused(field):
    assert client.post("/forecast", json=dict(SAMPLE, **{field: None})).status_code == 422


@pytest.mark.parametrize("field", OPTIONAL)
def test_null_for_an_optional_input_means_missing(field):
    r = client.post("/forecast", json=dict(SAMPLE, **{field: None}))
    assert r.status_code == 200 and 3 <= r.json()["predicted_hb_visit2_g_dl"] <= 20


@pytest.mark.parametrize("present", list(itertools.product([True, False], repeat=4)))
def test_every_pattern_of_left_out_optional_inputs_gives_a_forecast(present):
    payload = {k: v for k, v in SAMPLE.items() if k in REQUIRED or present[OPTIONAL.index(k)]}
    r = client.post("/forecast", json=payload)
    assert r.status_code == 200, r.text
    assert set(r.json()) == RESPONSE_FIELDS


def test_leaving_out_inputs_changes_the_estimate_but_never_the_status():
    complete = client.post("/forecast", json=SAMPLE).json()
    bare = client.post("/forecast", json={k: SAMPLE[k] for k in REQUIRED}).json()
    assert complete["predicted_hb_visit2_g_dl"] != bare["predicted_hb_visit2_g_dl"] or \
        complete["estimated_abs_error_g_dl"] != bare["estimated_abs_error_g_dl"]


# ---------------------------------------------------------------------------
# Malformed requests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, content_type", [
    ("{not json", "application/json"),
    ("", "application/json"),
    ("[1, 2, 3]", "application/json"),
    ("null", "application/json"),
    ("{}", "application/json"),
    ('"just a string"', "application/json"),
    ("hb_visit1_g_dl=10.8", "application/x-www-form-urlencoded"),
])
def test_malformed_requests_are_refused_with_422(text, content_type):
    assert post_raw(text, content_type).status_code == 422


def test_an_empty_object_names_all_six_required_inputs():
    detail = post_raw("{}").json()["detail"]
    assert sorted(d["loc"][1] for d in detail) == sorted(REQUIRED)


def test_unknown_extra_fields_are_ignored_and_change_nothing():
    plain = client.post("/forecast", json=SAMPLE).json()
    extra = client.post("/forecast", json=dict(SAMPLE, secret="x", patient_name="nobody")).json()
    assert plain == extra


def test_wrong_method_and_unknown_path():
    assert client.get("/forecast").status_code == 405
    assert client.put("/forecast", json=SAMPLE).status_code == 405
    assert client.delete("/forecast").status_code == 405
    assert client.get("/no-such-route").status_code == 404


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def test_the_response_has_exactly_the_documented_fields_and_rounding():
    body = client.post("/forecast", json=SAMPLE).json()
    assert set(body) == RESPONSE_FIELDS
    assert round(body["predicted_hb_visit2_g_dl"], 2) == body["predicted_hb_visit2_g_dl"]
    assert round(body["estimated_abs_error_g_dl"], 3) == body["estimated_abs_error_g_dl"]
    assert round(body["cutoff_g_dl"], 3) == body["cutoff_g_dl"]


def test_the_flag_agrees_with_the_cutoff_and_the_note_says_the_same():
    for hb in (5.0, 8.0, 10.8, 12.0, 15.0):
        body = client.post("/forecast", json=dict(SAMPLE, hb_visit1_g_dl=hb)).json()
        kept = body["estimated_abs_error_g_dl"] <= body["cutoff_g_dl"]
        assert body["reliability_flag"] == ("kept" if kept else "set_aside") or \
            abs(body["estimated_abs_error_g_dl"] - body["cutoff_g_dl"]) < 0.0006   # rounding at the boundary
        assert ("would be kept" in body["reliability_note"]) == (body["reliability_flag"] == "kept")


def test_openapi_describes_both_routes_and_the_limits():
    spec = client.get("/openapi.json").json()
    assert "/forecast" in spec["paths"] and "/health" in spec["paths"]
    hb = spec["components"]["schemas"]["VisitOneFeatures"]["properties"]["hb_visit1_g_dl"]
    assert hb["minimum"] == 3 and hb["maximum"] == 20
    assert client.get("/docs").status_code == 200


# ---------------------------------------------------------------------------
# The limits against the real data
# ---------------------------------------------------------------------------
def test_the_wide_limits_accept_every_woman_in_the_training_and_development_data():
    frames = [pd.read_csv(os.path.join(DATA, f)) for f in ("train.csv", "validation.csv")
              if os.path.exists(os.path.join(DATA, f))]
    if not frames:
        pytest.skip("training and development files not found")
    data = pd.concat(frames)
    for field in OPTIONAL + ["pregnancy_count"]:
        low, high = LIMITS[field]
        seen = data[field].dropna()
        assert seen.min() >= low and seen.max() <= high, f"{field}: {seen.min()} to {seen.max()}"
    # and the schema itself accepts a typical woman with those values
    VisitOneFeatures(**SAMPLE)
