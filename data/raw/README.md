# Raw data is not stored here

The raw file is deliberately not in this repository. It is an open dataset on Zenodo, and the code needs it only to rebuild the prepared tables and to rerun the audit.

To rerun `python src/audit.py` or `python src/run_stage1.py`:

1. Download `maternal_dataset_csv.csv` (Version 1) from https://zenodo.org/records/15309733/files/maternal_dataset_csv.csv?download=1 and save it in this folder, so that its path is `data/raw/maternal_dataset_csv.csv`.
2. Check it. Zenodo lists the MD5 `0253d2d385dd80d6c9bb82df6b85fcc8` for this file, and its size is 15,074,677 bytes.
3. Run `python src/audit.py`. It checks the MD5 again and reproduces the counts in `reports/audit_report.md`.

The other checks, the SHA-256, the citation and what the file contains are in `docs/data_provenance.md`. Please do not commit the file to this repository.
