"""
Tests for scripts/check_reproducibility.py: how it decides that a rebuilt file is the same as the
committed one. The rebuild itself needs the raw file and is not run here; it is run by hand
(see docs/performance.md and the technical report).
"""
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from check_reproducibility import (
    DERIVED_FILES, DIFFERENT, IDENTICAL, MISSING, ROOT, SAME_EXCEPT_PYTHON, classify, without_python_version,
)


def write_json(path, python="3.9.25", numpy="2.0.2", mae=0.794):
    path.write_text(json.dumps({"versions": {"python": python, "numpy": numpy}, "mae": mae}))
    return path


def small_forest(seed):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    return RandomForestRegressor(n_estimators=5, max_depth=3, random_state=seed).fit(X, X[:, 0] + rng.normal(size=60))


def test_the_same_bytes_are_identical(tmp_path):
    a, b = write_json(tmp_path / "a.json"), write_json(tmp_path / "b.json")
    assert classify(a, b) == IDENTICAL


def test_a_different_python_version_alone_is_not_a_difference(tmp_path):
    a = write_json(tmp_path / "a.json", python="3.9.25")
    b = write_json(tmp_path / "b.json", python="3.12.13")
    assert classify(a, b) == SAME_EXCEPT_PYTHON


def test_a_changed_number_is_a_difference_even_if_the_python_version_also_changed(tmp_path):
    a = write_json(tmp_path / "a.json", python="3.9.25", mae=0.794)
    b = write_json(tmp_path / "b.json", python="3.12.13", mae=0.795)
    assert classify(a, b) == DIFFERENT


def test_a_changed_library_version_is_a_difference(tmp_path):
    a = write_json(tmp_path / "a.json", numpy="2.0.2")
    b = write_json(tmp_path / "b.json", numpy="2.1.0")
    assert classify(a, b) == DIFFERENT


def test_a_missing_file_is_reported_not_skipped(tmp_path):
    a = write_json(tmp_path / "a.json")
    assert classify(a, tmp_path / "nothing.json") == MISSING
    assert classify(tmp_path / "nothing.json", a) == MISSING


def test_the_python_version_line_of_a_markdown_report_is_set_aside(tmp_path):
    a, b, c = (tmp_path / name for name in ("a.md", "b.md", "c.md"))
    a.write_text("- Python 3.9.25, scikit-learn 1.6.1\nMAE 0.794\n")
    b.write_text("- Python 3.11.17, scikit-learn 1.6.1\nMAE 0.794\n")
    c.write_text("- Python 3.11.17, scikit-learn 1.6.1\nMAE 0.795\n")
    assert classify(a, b) == SAME_EXCEPT_PYTHON
    assert classify(a, c) == DIFFERENT


def test_other_files_must_match_byte_for_byte(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"\x89PNG one")
    b.write_bytes(b"\x89PNG two")
    assert classify(a, b) == DIFFERENT


def test_only_the_python_entry_of_versions_is_blanked():
    value = {"versions": {"python": "3.9.25", "numpy": "2.0.2"}, "python": "kept",
             "rows": [{"versions": {"python": "x"}}]}
    blanked = without_python_version(value)
    assert blanked["versions"] == {"python": "<python>", "numpy": "2.0.2"}
    assert blanked["python"] == "kept"
    assert value["versions"]["python"] == "3.9.25"       # the original is not changed


def test_saved_models_with_the_same_trees_match_apart_from_the_python_version(tmp_path):
    a, b = tmp_path / "a.joblib", tmp_path / "b.joblib"
    joblib.dump({"model": small_forest(1), "versions": {"python": "3.9.25"}}, a)
    joblib.dump({"model": small_forest(1), "versions": {"python": "3.12.13"}}, b)
    assert classify(a, b) == SAME_EXCEPT_PYTHON


def test_saved_models_with_different_trees_are_different(tmp_path):
    a, b = tmp_path / "a.joblib", tmp_path / "b.joblib"
    joblib.dump({"model": small_forest(1), "versions": {"python": "3.9.25"}}, a)
    joblib.dump({"model": small_forest(2), "versions": {"python": "3.9.25"}}, b)
    assert classify(a, b) == DIFFERENT


def test_the_list_of_derived_files_names_nineteen_files_that_exist_in_the_repository():
    assert len(DERIVED_FILES) == len(set(DERIVED_FILES)) == 19
    missing = [name for name in DERIVED_FILES if not os.path.exists(os.path.join(ROOT, name))]
    assert missing == []
