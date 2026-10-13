"""
Executable audit of the data preparation.

Reproduces, from the raw release, the numbers the proposal reports: row counts,
the exclusion steps down to the eligible cohort, missingness among the ten
inputs, the partition sizes, and the identical-vector integrity check.

Run from the repository root:

    python src/audit.py

Writes reports/audit_report.md and prints the same text.
"""
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_adapter import load_and_map  # noqa: E402
from pipeline import (  # noqa: E402
    FEATURE_COLUMNS, GESTATION_VISIT2_COLUMN, TARGET_COLUMN, apply_eligibility,
    fixed_split, identical_vector_groups, missingness_report, split_group_violations,
    N_TRAIN, N_DEV, N_TEST,
)

RAW = "data/raw/maternal_dataset_csv.csv"
RAW_MD5 = "0253d2d385dd80d6c9bb82df6b85fcc8"


def exclusion_steps(m: pd.DataFrame) -> list:
    """Row-count waterfall from all rows to the eligible cohort (each step removes rows not removed earlier)."""
    steps = [("All rows in the release", len(m))]
    have_hb = m["hb_visit1_g_dl"].notna() & m[TARGET_COLUMN].notna()
    steps.append(("Missing a first- or second-contact haemoglobin", int((~have_hb).sum())))
    hb_ok = have_hb & m["hb_visit1_g_dl"].between(3, 20) & m[TARGET_COLUMN].between(3, 20)
    steps.append(("Haemoglobin outside 3 to 20 g/dL", int(have_hb.sum() - hb_ok.sum())))
    g_ok = hb_ok & m["gestation_visit1_weeks"].between(1, 42) & m[GESTATION_VISIT2_COLUMN].between(1, 42)
    steps.append(("Gestational week outside 1 to 42", int(hb_ok.sum() - g_ok.sum())))
    order_ok = g_ok & (m[GESTATION_VISIT2_COLUMN] > m["gestation_visit1_weeks"])
    steps.append(("Second contact not later than first", int(g_ok.sum() - order_ok.sum())))
    steps.append(("Eligible paired records", int(order_ok.sum())))
    return steps


def main():
    h = hashlib.md5()
    with open(RAW, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    md5 = h.hexdigest()

    mapped = load_and_map(RAW)
    eligible = apply_eligibility(mapped)
    steps = exclusion_steps(mapped)
    assert steps[-1][1] == len(eligible), "Waterfall does not match apply_eligibility"
    miss = missingness_report(eligible)
    complete = int(eligible[FEATURE_COLUMNS].notna().all(axis=1).sum())
    groups = identical_vector_groups(eligible)
    dup_rows = int((pd.Series(groups).map(pd.Series(groups).value_counts()) > 1).sum())
    train, dev, test = fixed_split(eligible, seed=42)

    lines = ["# Data preparation audit", "",
             f"- Raw file: maternal_dataset_csv.csv, Zenodo 10.5281/zenodo.15309733, Version 1",
             f"- MD5 computed: {md5}",
             f"- MD5 expected: {RAW_MD5}  ->  {'MATCH' if md5 == RAW_MD5 else 'MISMATCH'}", "",
             "## Exclusions", "", "| Step | Rows |", "| --- | --- |"]
    lines += [f"| {name} | {n:,} |" for name, n in steps]
    lines += ["", "## Missingness among the ten inputs (eligible records)", "",
              "| Input | Missing | % |", "| --- | --- | --- |"]
    lines += [f"| {idx} | {int(r.n_missing)} | {r.pct_missing} |" for idx, r in miss.iterrows()]
    lines += ["", f"Complete on all ten inputs: {complete:,} of {len(eligible):,}",
              f"Median first haemoglobin (eligible): {eligible['hb_visit1_g_dl'].median():.1f} g/dL", "",
              "## Partitions", "",
              f"- Train / development / test: {len(train):,} / {len(dev):,} / {len(test):,} "
              f"(expected {N_TRAIN:,} / {N_DEV:,} / {N_TEST:,})",
              f"- Rows sharing an identical ten-input vector with another row: {dup_rows}",
              f"- Identical-vector groups split across partitions: {split_group_violations(train, dev, test)}",
              "", "Grouping by identical vectors is a proxy for the same woman. Without a verified "
              "patient identifier it cannot guarantee that partitions are independent."]
    text = "\n".join(lines)
    os.makedirs("reports", exist_ok=True)
    with open("reports/audit_report.md", "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
