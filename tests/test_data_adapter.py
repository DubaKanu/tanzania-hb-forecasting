"""
Tests for src/data_adapter.py: reading only the 12 columns the project uses and cleaning them.
A small made-up raw file is written for each test; the real file is never needed.
"""
import numpy as np
import pandas as pd
import pytest

from data_adapter import RAW_COLUMNS, _clean_numeric, _split_bp, load_and_map


def _raw_frame(**overrides):
    base = {
        "SN": [101, 102, 103],
        "hemoglobin_check_result_v1": ["10.5", "9", "not_checked"],
        "pregnant_week_number_v1": [12, 20, 8],
        "age": [25, 31, 19],
        "no_pregnancy": [1, 3, 2],
        "height_cm": [158, 162, 150],
        "weight_kg_v1": [60, 71.5, 55],
        "pulse_rate_v1": ["80", "not_checked", "72"],
        "respiratory_rate_v1": ["18", "", "20"],
        "blood_pressure_v1": ["119/65", "bad", "110/70"],
        "hemoglobin_check_result_v2": ["11.0", "9.5", "10"],
        "pregnant_week_number_v2": [24, 30, 20],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_numeric_turns_sentinels_and_blanks_into_missing():
    series = pd.Series(["12.5", " 7 ", "not_checked", "NaN", "", "None", "abc", 3])
    out = _clean_numeric(series)
    assert out.iloc[0] == 12.5 and out.iloc[1] == 7 and out.iloc[7] == 3
    assert out.iloc[2:7].isna().all()


def test_split_bp_reads_systolic_and_diastolic_and_blanks_what_it_cannot_read():
    sys_, dia = _split_bp(pd.Series(["119/65", "bad", "120/", "/80", "130/85/9"]))
    assert list(sys_.iloc[:1]) == [119.0] and list(dia.iloc[:1]) == [65.0]
    assert np.isnan(sys_.iloc[1]) and np.isnan(dia.iloc[1])
    assert sys_.iloc[2] == 120.0 and np.isnan(dia.iloc[2])
    assert np.isnan(sys_.iloc[3]) and dia.iloc[3] == 80.0
    assert np.isnan(dia.iloc[4])      # "85/9" is not a number: treated as missing, not guessed


def test_load_and_map_renames_to_the_ten_inputs_the_target_and_the_record_id(tmp_path):
    path = tmp_path / "raw.csv"
    _raw_frame().to_csv(path, index=False)
    out = load_and_map(str(path))
    assert list(out.columns) == [
        "hb_visit1_g_dl", "gestation_visit1_weeks", "maternal_age_years", "pregnancy_count", "height_cm",
        "weight_visit1_kg", "pulse_visit1_bpm", "respiratory_rate_visit1", "systolic_bp_visit1",
        "diastolic_bp_visit1", "target_hb_visit2_g_dl", "gestation_visit2_weeks", "_source_row"]
    first = out.iloc[0]
    assert (first["hb_visit1_g_dl"], first["systolic_bp_visit1"], first["diastolic_bp_visit1"]) == (10.5, 119, 65)
    assert first["_source_row"] == 101 and first["target_hb_visit2_g_dl"] == 11.0
    assert np.isnan(out.loc[2, "hb_visit1_g_dl"]) and np.isnan(out.loc[1, "pulse_visit1_bpm"])


def test_load_and_map_never_reads_the_other_columns_of_the_file(tmp_path):
    path = tmp_path / "raw.csv"
    frame = _raw_frame()
    frame["hiv_test_result"] = ["positive", "negative", "negative"]    # sensitive: must never be loaded
    frame["risk_label"] = ["high", "low", "low"]                       # the dataset's own label: must never be loaded
    frame["free_text_notes"] = ["a", "b", "c"]
    frame.to_csv(path, index=False)
    out = load_and_map(str(path))
    assert out.shape[1] == 13
    assert not any(s in " ".join(out.columns) for s in ("hiv", "risk", "notes"))


def test_load_and_map_stops_if_a_needed_column_is_missing(tmp_path):
    path = tmp_path / "raw.csv"
    _raw_frame().drop(columns=["weight_kg_v1"]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_and_map(str(path))


def test_exactly_twelve_raw_columns_are_used():
    assert len(RAW_COLUMNS) == 12 and len(set(RAW_COLUMNS)) == 12
