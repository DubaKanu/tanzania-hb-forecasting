"""
Maps the raw Zenodo MHRS export (683 wide columns, one row per woman, visit
data spread across _v1.._v8 suffixes) onto the ten-feature / one-target
contract used by src/pipeline.py.

This performs ONLY field selection, renaming, and light type-cleaning
(parsing "120/80" style BP strings, converting "not_checked" sentinels to
missing). It does not impute, does not drop rows, and does not touch the
eligibility screen itself -- that stays in pipeline.apply_eligibility so the
audit rule is defined in exactly one place.
"""
import pandas as pd
import numpy as np


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a string column with 'not_checked' / blank sentinels to float."""
    s = series.astype(str).str.strip().str.lower()
    s = s.replace({"not_checked": np.nan, "nan": np.nan, "": np.nan, "none": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _split_bp(series: pd.Series):
    """'119/65' -> (119.0, 65.0); anything unparseable -> (NaN, NaN)."""
    parts = series.astype(str).str.split("/", n=1, expand=True)
    if parts.shape[1] < 2:
        return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
    sys_ = pd.to_numeric(parts[0], errors="coerce")
    dia_ = pd.to_numeric(parts[1], errors="coerce")
    return sys_, dia_


def load_and_map(raw_path: str) -> pd.DataFrame:
    raw = pd.read_csv(raw_path, low_memory=False)

    sys1, dia1 = _split_bp(raw["blood_pressure_v1"])

    out = pd.DataFrame({
        "hb_visit1_g_dl": pd.to_numeric(raw["hemoglobin_check_result_v1"], errors="coerce"),
        "gestation_visit1_weeks": pd.to_numeric(raw["pregnant_week_number_v1"], errors="coerce"),
        "maternal_age_years": pd.to_numeric(raw["age"], errors="coerce"),
        "pregnancy_count": pd.to_numeric(raw["no_pregnancy"], errors="coerce"),
        "height_cm": pd.to_numeric(raw["height_cm"], errors="coerce"),
        "weight_visit1_kg": pd.to_numeric(raw["weight_kg_v1"], errors="coerce"),
        "pulse_visit1_bpm": _clean_numeric(raw["pulse_rate_v1"]),
        "respiratory_rate_visit1": _clean_numeric(raw["respiratory_rate_v1"]),
        "systolic_bp_visit1": sys1,
        "diastolic_bp_visit1": dia1,
        # target + the second gestation reading (needed for eligibility, not a model input)
        "target_hb_visit2_g_dl": _clean_numeric(raw["hemoglobin_check_result_v2"]),
        "gestation_visit2_weeks": pd.to_numeric(raw["pregnant_week_number_v2"], errors="coerce"),
    })

    # keep original row id and the (unused-as-a-feature) original classification
    # label for reference/reporting only -- never fed to the models.
    out["_source_row"] = raw["SN"]
    out["_original_risk_label"] = raw["Risk"]
    return out
