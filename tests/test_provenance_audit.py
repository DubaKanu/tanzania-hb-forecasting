"""
Tests for src/provenance.py (checksums and versions) and the pure parts of src/audit.py
(the exclusion waterfall and the exclusion log text). Small made-up tables only.
"""
import hashlib

import numpy as np
import pandas as pd
import sklearn

from audit import RAW_MD5, exclusion_log_text, exclusion_steps, report_lines
from pipeline import GESTATION_VISIT2_COLUMN, TARGET_COLUMN, apply_eligibility, missingness_report
from provenance import library_versions, md5_of_file, md5_of_frame


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def test_md5_of_file_matches_hashlib_including_for_files_larger_than_one_chunk(tmp_path):
    small, big = tmp_path / "small.bin", tmp_path / "big.bin"
    small.write_bytes(b"haemoglobin")
    big.write_bytes(np.random.default_rng(0).bytes((1 << 20) * 2 + 123))
    assert md5_of_file(str(small)) == hashlib.md5(b"haemoglobin").hexdigest()
    assert md5_of_file(str(big)) == hashlib.md5(big.read_bytes()).hexdigest()


def test_md5_of_frame_follows_the_csv_text_and_changes_with_the_data():
    df = pd.DataFrame({"a": [1, 2], "b": [0.5, 1.5]})
    assert md5_of_frame(df) == hashlib.md5(b"a,b\n1,0.5\n2,1.5\n").hexdigest()
    assert md5_of_frame(df) == md5_of_frame(df.copy())
    changed = df.copy()
    changed.loc[0, "a"] = 9
    assert md5_of_frame(changed) != md5_of_frame(df)


def test_library_versions_name_python_and_the_three_libraries():
    versions = library_versions()
    assert set(versions) == {"python", "pandas", "numpy", "scikit-learn"}
    assert versions["pandas"] == pd.__version__ and versions["numpy"] == np.__version__
    assert versions["scikit-learn"] == sklearn.__version__


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def _raw_rows():
    """Seven made-up rows, each meant to fall at a different step of the waterfall."""
    return pd.DataFrame({
        "hb_visit1_g_dl":         [11.0, np.nan, 25.0, 10.0, 10.0, 9.0, 12.0],
        TARGET_COLUMN:            [11.5, 11.0,   11.0, 10.5, 10.5, 9.5, 12.5],
        "gestation_visit1_weeks": [10,   10,     10,   0,    20,   20,  15],
        GESTATION_VISIT2_COLUMN:  [20,   20,     20,   10,   20,   30,  30],
    })
    # row 0 eligible, row 1 no first Hb, row 2 Hb 25, row 3 first week 0, row 4 week not later, rows 5 and 6 eligible


def test_the_exclusion_waterfall_counts_each_row_once_and_ends_at_the_eligible_cohort():
    rows = _raw_rows()
    steps = dict(exclusion_steps(rows))
    assert steps["All rows in the release"] == 7
    assert steps["Missing a first- or second-contact haemoglobin"] == 1
    assert steps["Haemoglobin outside 3 to 20 g/dL"] == 1
    assert steps["Gestational week outside 1 to 42"] == 1
    assert steps["Second contact not later than first"] == 1
    assert steps["Eligible paired records"] == len(apply_eligibility(rows)) == 3
    removed = sum(n for name, n in steps.items() if name not in ("All rows in the release", "Eligible paired records"))
    assert removed + steps["Eligible paired records"] == steps["All rows in the release"]


def test_the_exclusion_log_text_counts_every_reason_and_says_where_it_came_from():
    rows = _raw_rows()
    text = exclusion_log_text(rows, n_eligible=3, md5="abc123")
    assert text.startswith("Derived from data/raw/maternal_dataset_csv.csv (md5:abc123)")
    assert "eligible" in text and "missing_hb_reading" in text and "hb_visit1_out_of_range" in text
    assert text.endswith("total_rows: 7\neligible: 3\n")


def test_the_audit_report_names_the_checksum_result_and_the_partition_sizes():
    rows = _raw_rows()
    eligible = apply_eligibility(rows)
    miss = missingness_report(eligible.assign(**{c: 1.0 for c in [
        "maternal_age_years", "pregnancy_count", "height_cm", "weight_visit1_kg", "pulse_visit1_bpm",
        "respiratory_rate_visit1", "systolic_bp_visit1", "diastolic_bp_visit1"]}))
    parts = (eligible.head(2), eligible.tail(1), eligible.tail(0))
    text = "\n".join(report_lines(RAW_MD5, exclusion_steps(rows), miss, eligible.assign(
        **{c: 1.0 for c in miss.index if c not in eligible}), parts, 0, 0))
    assert "MATCH" in text and "MISMATCH" not in text
    assert "Train / development / test: 2 / 1 / 0" in text
    assert "MISMATCH" in "\n".join(report_lines("0" * 32, exclusion_steps(rows), miss, eligible.assign(
        **{c: 1.0 for c in miss.index if c not in eligible}), parts, 0, 0))
