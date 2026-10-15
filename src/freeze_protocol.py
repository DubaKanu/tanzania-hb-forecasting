"""
Freeze the protocol before the locked test. Run from the repository root, once,
after run_stage1.py and run_stage2.py:

    python src/freeze_protocol.py

It records a SHA-256 hash of every file the locked test depends on (this protocol,
the code, the data files, the saved models and the development results) in
reports/protocol_freeze.json, with the time and the library versions.

After that, run_locked_test.py checks every hash and refuses to run if anything has
changed. This script only reads files to hash them. It does not evaluate the test
women, and it refuses to run if the locked test has already been run.
"""
import datetime
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import sklearn

DATA = "data/Josephine_African_Data_Review/Tanzania_Followup_Haemoglobin"
FREEZE_PATH = "reports/protocol_freeze.json"
LOCKED_RESULTS_PATH = "reports/locked_test_results.json"

FROZEN_FILES = [
    "docs/frozen_protocol.md",
    "src/pipeline.py",
    "src/stage2.py",
    "src/data_adapter.py",
    "src/run_stage1.py",
    "src/run_stage2.py",
    "src/freeze_protocol.py",
    "src/run_locked_test.py",
    f"{DATA}/split_manifest.csv",
    f"{DATA}/cv_folds.csv",
    f"{DATA}/train.csv",
    f"{DATA}/validation.csv",
    f"{DATA}/test.csv",
    "reports/stage1_results.json",
    "reports/oof_stage1.csv",
    "reports/oof_run.json",
    "reports/stage2_dev_results.json",
    "models/stage1_model.joblib",
    "models/stage2_model.joblib",
]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:  # bytes only: hashing does not evaluate the test women
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze(record: dict, root: str = ".") -> list:
    """Problems found when the current files are compared with the frozen record."""
    problems = []
    for rel, info in record["files"].items():
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            problems.append(f"missing: {rel}")
        elif sha256_of(path) != info["sha256"]:
            problems.append(f"changed since the freeze: {rel}")
    return problems


def main():
    if os.path.exists(LOCKED_RESULTS_PATH):
        sys.exit("The locked test has already been run. Nothing can be frozen after it.")
    if os.path.exists(FREEZE_PATH):
        with open(FREEZE_PATH) as f:
            when = json.load(f).get("frozen_at_utc", "an earlier time")
        sys.exit(f"The protocol was already frozen at {when}. To freeze again, delete {FREEZE_PATH} "
                 "yourself and write down why.")
    missing = [p for p in FROZEN_FILES if not os.path.exists(p)]
    if missing:
        sys.exit("These files are missing: " + ", ".join(missing) +
                 "\nRun python src/run_stage1.py and python src/run_stage2.py first.")

    # The Stage 2 pieces must all belong to the same run.
    with open("reports/oof_run.json") as f:
        oof_run = json.load(f)
    with open("reports/stage2_dev_results.json") as f:
        dev_run = json.load(f)
    if oof_run["run_id"] != dev_run["run_id"]:
        sys.exit("oof_run.json and stage2_dev_results.json come from different runs. Rerun run_stage2.py.")

    record = {
        "frozen_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": oof_run["run_id"],
        "note": ("Hashes of everything the locked test depends on. run_locked_test.py refuses to run if "
                 "any of these files has changed."),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "files": {p: {"sha256": sha256_of(p), "bytes": os.path.getsize(p)} for p in FROZEN_FILES},
    }
    os.makedirs("reports", exist_ok=True)
    with open(FREEZE_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Protocol frozen at {record['frozen_at_utc']} UTC. Run ID {record['run_id']}.")
    print(f"{len(FROZEN_FILES)} files hashed into {FREEZE_PATH}. Do not edit any of them now.")
    print("Next: python src/run_locked_test.py  (runs once)")


if __name__ == "__main__":
    main()
