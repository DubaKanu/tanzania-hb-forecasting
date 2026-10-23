"""
Reproducibility check: rebuild every derived file from the raw file and compare it with the one in this repository.

    python scripts/check_reproducibility.py

It needs the raw file at data/raw/maternal_dataset_csv.csv (see data/raw/README.md). Nothing in the
repository is changed: the repository is copied to a temporary folder, the 19 derived files are
deleted there, and src/audit.py, src/run_stage1.py and src/run_stage2.py make them again from the raw
file. It takes about a minute.

Each file is reported as one of
    identical                             the same bytes
    identical except the Python version  the saved Python version string differs (the rebuild ran on
                                          another Python); every number in the file is equal
    DIFFERENT or MISSING                  something changed; the exit code is 1
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent
DATA = "data/Josephine_African_Data_Review/Tanzania_Followup_Haemoglobin"

# Every file the three scripts write.
DERIVED_FILES = [f"{DATA}/{name}" for name in (
    "cv_folds.csv", "exclusion_log.txt", "model_ready.csv", "split_manifest.csv", "test.csv", "train.csv",
    "validation.csv")] + [
    "models/stage1_model.joblib", "models/stage2_model.joblib",
    "reports/audit_report.md", "reports/fig_error_vs_coverage_dev.png", "reports/fig_forecast_vs_observed_dev.png",
    "reports/fig_stage2_check_dev.png", "reports/oof_run.json", "reports/oof_stage1.csv",
    "reports/stage1_results.json", "reports/stage2_dev_results.json", "reports/stage2_dev_results.md",
    "reports/stage2_dev_table.csv",
]
STEPS = ["src/audit.py", "src/run_stage1.py", "src/run_stage2.py"]
RAW_FILE = "data/raw/maternal_dataset_csv.csv"
IGNORED_WHEN_COPYING = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "venv", ".venv")

IDENTICAL = "identical"
SAME_EXCEPT_PYTHON = "identical except the Python version"
DIFFERENT = "DIFFERENT"
MISSING = "MISSING"
PYTHON_TOKEN = re.compile(r"Python \d+\.\d+\.\d+")


# ---------------------------------------------------------------------------
# Comparing one file
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_python_version(value):
    """The same JSON-like value with the saved Python version (the "python" entry of "versions") blanked out."""
    if isinstance(value, dict):
        return {key: _blank_python(item) if key == "versions" else without_python_version(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [without_python_version(item) for item in value]
    return value


def _blank_python(versions):
    if isinstance(versions, dict) and "python" in versions:
        return {**versions, "python": "<python>"}
    return versions


def same_forest(a: RandomForestRegressor, b: RandomForestRegressor) -> bool:
    """Every tree has the same splits, thresholds and leaf values."""
    if len(a.estimators_) != len(b.estimators_):
        return False
    return all(x.tree_.node_count == y.tree_.node_count
               and np.array_equal(x.tree_.feature, y.tree_.feature)
               and np.array_equal(x.tree_.threshold, y.tree_.threshold)
               and np.array_equal(x.tree_.value, y.tree_.value)
               for x, y in zip(a.estimators_, b.estimators_))


def _same_imputer(a: SimpleImputer, b: SimpleImputer) -> bool:
    return bool(np.array_equal(a.statistics_, b.statistics_))


def _same_mapping(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(same_object(a[key], b[key]) for key in a)


def _same_sequence(a, b) -> bool:
    return len(a) == len(b) and all(same_object(x, y) for x, y in zip(a, b))


# How to compare two objects of each kind found in a saved model. To look inside another kind of object,
# add one line here.
COMPARERS = [
    (RandomForestRegressor, same_forest),
    (SimpleImputer, _same_imputer),
    (dict, _same_mapping),
    ((list, tuple), _same_sequence),
]


def same_object(a, b) -> bool:
    """Equality that looks inside the scikit-learn objects and dictionaries of a saved model."""
    for kind, compare in COMPARERS:
        if isinstance(a, kind) and isinstance(b, kind):
            return compare(a, b)
    return bool(a == b)


def same_apart_from_python_version(committed: Path, rebuilt: Path) -> bool:
    """True if the two files hold the same content once the saved Python version is set aside."""
    suffix = committed.suffix
    if suffix == ".json":
        return (without_python_version(json.loads(committed.read_text()))
                == without_python_version(json.loads(rebuilt.read_text())))
    if suffix == ".md":
        return (PYTHON_TOKEN.sub("Python <version>", committed.read_text())
                == PYTHON_TOKEN.sub("Python <version>", rebuilt.read_text()))
    if suffix == ".joblib":
        return same_object(without_python_version(joblib.load(committed)),
                           without_python_version(joblib.load(rebuilt)))
    return False


def classify(committed: Path, rebuilt: Path) -> str:
    """IDENTICAL, SAME_EXCEPT_PYTHON, DIFFERENT or MISSING for one pair of files."""
    if not committed.exists() or not rebuilt.exists():
        return MISSING
    if sha256(committed) == sha256(rebuilt):
        return IDENTICAL
    return SAME_EXCEPT_PYTHON if same_apart_from_python_version(committed, rebuilt) else DIFFERENT


# ---------------------------------------------------------------------------
# Rebuilding in a copy
# ---------------------------------------------------------------------------
def make_copy_without_derived_files(destination: Path) -> Path:
    """Copy the repository to `destination` and delete the derived files there."""
    copy = destination / "repo"
    shutil.copytree(ROOT, copy, ignore=IGNORED_WHEN_COPYING)
    for name in DERIVED_FILES:
        (copy / name).unlink(missing_ok=True)
    return copy


def run_step(copy: Path, step: str) -> float:
    """Run one script inside the copy and return how many seconds it took. Stops with its error text if it fails."""
    started = time.perf_counter()
    result = subprocess.run([sys.executable, step], cwd=copy, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"{step} failed:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
    return time.perf_counter() - started


def compare_all(copy: Path) -> List[Tuple[str, str]]:
    return [(name, classify(ROOT / name, copy / name)) for name in DERIVED_FILES]


def print_table(results: List[Tuple[str, str]]) -> None:
    width = max(len(name) for name, _ in results)
    for name, verdict in results:
        print(f"  {name.ljust(width)}  {verdict}")
    counts = {verdict: sum(1 for _, v in results if v == verdict) for verdict in (IDENTICAL, SAME_EXCEPT_PYTHON)}
    bad = [name for name, verdict in results if verdict in (DIFFERENT, MISSING)]
    print(f"\n{len(results)} derived files: {counts[IDENTICAL]} identical, "
          f"{counts[SAME_EXCEPT_PYTHON]} identical except the Python version, {len(bad)} different or missing")


def main() -> int:
    if not (ROOT / RAW_FILE).exists():
        print(f"The raw file is not at {RAW_FILE}. Download it as described in data/raw/README.md first.")
        return 2
    with tempfile.TemporaryDirectory() as temporary:
        copy = make_copy_without_derived_files(Path(temporary))
        print(f"Rebuilding {len(DERIVED_FILES)} files from the raw file in a temporary copy (about a minute)...")
        for step in STEPS:
            print(f"  {step}: {run_step(copy, step):.0f} s")
        print()
        results = compare_all(copy)
    print_table(results)
    return 0 if all(verdict in (IDENTICAL, SAME_EXCEPT_PYTHON) for _, verdict in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
