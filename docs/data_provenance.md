# Data provenance

Project: Predicting Follow-up Haemoglobin and Identifying Unreliable Predictions in a Tanzanian Antenatal Cohort
Author: Josephine Duba Kanu. Supervisor: Emmanuel Adjei.
Written on 20 October 2026. It describes the repository as it stands after the raw file was removed. The covering email gives the exact commit.

## 1. Source

| | |
|---|---|
| Dataset | *Dataset for Maternal Health Risks Stratification (MHRS) in Tanzania* |
| Version | 1, published 30 April 2025. The record notes that a newer version exists. This project uses Version 1 only and has not examined the newer one |
| Record | https://zenodo.org/records/15309733 |
| DOI | 10.5281/zenodo.15309733 |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Hosting institution | The University of Dodoma |
| Creators as listed on the record | Rukia Mwifunyi, Nyaura Kibinda, Noe Elisa, Florence Upendo Rashidi, Kilavo Hassan, Mustafa Mohsini, Theresia Masoi, Lilian Mnabwiru, Yahaya Hussein, Saturini Manangwa |
| Funders listed on the record | The Wellcome Trust and the Lacuna Fund |
| What the record says about collection | Data from 8,817 pregnant women attending maternity primary health facilities, over more than nine months (20 May 2024 to 30 March 2025), in five districts of Tanzania: Chamwino (Dodoma), Mkuranga (Pwani), Meatu (Simiyu), Mbulu (Manyara) and Kilolo (Iringa) |
| Related item listed on the record | DOI 10.17632/p5w98dvbbk.1 (relation "Describes", resource type dataset). This project has not examined it |

Citation (APA 7): Mwifunyi, R., Kibinda, N., Elisa, N., Rashidi, F. U., Hassan, K., Mohsini, M., Masoi, T., Mnabwiru, L., Hussein, Y., & Manangwa, S. (2025). *Dataset for Maternal Health Risks Stratification (MHRS) in Tanzania* (Version 1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15309733

## 2. Files on the record

| File | Size | MD5 listed on Zenodo | SHA-256 |
|---|---|---|---|
| `maternal_dataset_csv.csv` | 15,074,677 bytes (shown as 15.1 MB on the record) | `0253d2d385dd80d6c9bb82df6b85fcc8` | `ab95366ef22dde5d13df93f10723942f5faffda8fbf0007e8efe85e4f25c1126` |
| `metadata.pdf` | shown as 114.0 kB on the record | `853526fe45528323965f4bb59479c47a` | not computed |

Zenodo lists the MD5 only. For the data file, the exact size and the SHA-256 were computed for this project on a copy whose MD5 matches the listed value, as a second check. `metadata.pdf` is documentation. No code reads it, so no SHA-256 is given for it.

## 3. How to get the raw file and check it

The raw file is not stored in this repository.

1. Open the record, https://zenodo.org/records/15309733, and download `maternal_dataset_csv.csv` from the Files list. Save it as `data/raw/maternal_dataset_csv.csv`. The same file can be fetched on the command line: `curl -L -o data/raw/maternal_dataset_csv.csv "https://zenodo.org/records/15309733/files/maternal_dataset_csv.csv?download=1"`
2. Check the MD5 against the value in section 2, and the SHA-256 if you wish.
   - macOS: `md5 data/raw/maternal_dataset_csv.csv` and `shasum -a 256 data/raw/maternal_dataset_csv.csv`
   - Linux: `md5sum data/raw/maternal_dataset_csv.csv` and `sha256sum data/raw/maternal_dataset_csv.csv`
   - Windows PowerShell: `Get-FileHash data\raw\maternal_dataset_csv.csv -Algorithm MD5` (and `-Algorithm SHA256`)
3. Run `python src/audit.py`. It computes the MD5 again, prints MATCH or MISMATCH, and reproduces the counts in `reports/audit_report.md`. `python src/run_stage1.py` stops if the MD5 differs.

## 4. What the code reads from the file

The file has 8,817 rows and 683 columns. The code loads 12 of them. `RAW_COLUMNS` in `src/data_adapter.py` lists the 12 names and passes them to `pandas.read_csv(usecols=...)`, so the other 671 columns are never loaded. If a name is missing from the file, loading stops with an error. The prepared tables have 13 columns.

| Column in the file | Name in the prepared tables | Role |
|---|---|---|
| `hemoglobin_check_result_v1` | `hb_visit1_g_dl` | input |
| `pregnant_week_number_v1` | `gestation_visit1_weeks` | input |
| `age` | `maternal_age_years` | input |
| `no_pregnancy` | `pregnancy_count` | input |
| `height_cm` | `height_cm` | input |
| `weight_kg_v1` | `weight_visit1_kg` | input |
| `pulse_rate_v1` | `pulse_visit1_bpm` | input |
| `respiratory_rate_v1` | `respiratory_rate_visit1` | input |
| `blood_pressure_v1` (text such as "120/80", split in two) | `systolic_bp_visit1`, `diastolic_bp_visit1` | inputs |
| `hemoglobin_check_result_v2` | `target_hb_visit2_g_dl` | target: haemoglobin at the next recorded visit |
| `pregnant_week_number_v2` | `gestation_visit2_weeks` | eligibility check and follow-up gap only, never a model input |
| `SN` | `_source_row` | the file's own row number, used to trace every table back to a row |

Not loaded: the columns for HIV, syphilis and malaria tests, urine and blood tests, substance use, intimate partner violence, the calendar dates (for example the last menstrual period and the expected date of delivery), the free-text "specify" fields (for example current pregnancy problems, medication and allergies), visits 3 to 8, the four postnatal contacts, the delivery and newborn fields, and the dataset's own risk label.

The file has no name, phone, address, facility or district columns. `SN` is only a row number (1 to 8,817), so there is no verified patient identifier. Of the 8,817 rows, 315 record an age of 16 or under, and 63 of the 2,685 women used do.

Until this commit, the prepared tables carried one more column, the dataset's own risk label (high or low), copied for reference only. No model, selector or report used it. This commit removes it from the code and from the four prepared tables, and limits the loader to the 12 columns above. Checked on 20 October 2026: the tables regenerated by the code equal the old tables without that column, and every result is unchanged.

## 5. What the record and `metadata.pdf` say about ethics, consent and de-identification

- The Zenodo record page has no statement about ethics approval, consent, de-identification or anonymisation. It shows no contact for questions. Its only statement about collection is summarised in section 1.
- `metadata.pdf` has 4 pages. It is a data dictionary of 36 features, each with a description, type and unit. It has no statement about ethics approval, consent, de-identification or anonymisation, or about how the data were collected. It does list sensitive clinical features (for example HIV, syphilis and malaria results), so the file holds sensitive health information.

Not available in either document, and so unknown to this project:

1. Whether an ethics committee reviewed or approved the original study, and under which reference.
2. How consent was obtained, including for girls aged 16 or under, and whether it covered public release of the data.
3. Which de-identification or anonymisation steps were applied, and whether the risk of re-identification was assessed.
4. How the data were collected and checked (instruments, training of staff, quality control).
5. A column-by-column dictionary for the file. `metadata.pdf` describes 36 features and the file has 683 columns, so the meaning and unit of each column used here are taken from its name and from the 36-feature dictionary.
6. Whom to ask about consent or data protection. The record gives no contact.

Not yet examined: the related item listed on the record (DOI 10.17632/p5w98dvbbk.1) and any article by the creators. Either may state the ethics approval and the consent procedure.

What this project does about it:

- It claims no ethics approval, consent procedure or de-identification method for the data.
- It is a secondary analysis of an existing open dataset. No participant was recruited or contacted and no new data were collected. It recruits no participants and collects no usability feedback.
- It loads only the 12 columns in section 4. The demonstration API takes made-up values only and stores nothing it receives.
- The raw file is not stored in this repository (section 3).
- An ALU Research Ethics Application Form for this study is being prepared for the supervisor's review. Public availability of the data and the absence of new participants are not treated as an exemption. The institutional determination of the route and timeline is still needed, and no clearance has been granted at the time of writing.

## 6. Cohort, exclusions and partitions

Reproduced by `python src/audit.py` from the pinned file. The output is in `reports/audit_report.md`, and the reason for every excluded row is in `data/Josephine_African_Data_Review/Tanzania_Followup_Haemoglobin/exclusion_log.txt`.

| Step | Rows |
|---|---|
| All rows in the release | 8,817 |
| Missing a first- or second-contact haemoglobin | 6,036 |
| Haemoglobin outside 3 to 20 g/dL | 11 |
| Gestational week outside 1 to 42 | 5 |
| Second contact not later than first | 80 |
| Eligible paired records | 2,685 |

- Complete on all ten inputs: 2,025 of 2,685. Respiratory rate has the most missing values (643, 23.9%).
- Partitions: 1,884 training, 402 development, 399 locked test, seed 42. Women with an identical ten-input vector stay in one partition (2 rows share a vector, and no group is split). The partition of every woman is in `data/Josephine_African_Data_Review/Tanzania_Followup_Haemoglobin/split_manifest.csv`. Without a verified patient identifier this grouping is only a proxy for the same woman.

## 7. Code revision and reproduction

The preparation code is `src/data_adapter.py` (field mapping), `src/pipeline.py` (eligibility and split) and `src/run_stage1.py` (saves the prepared tables, manifest and folds). The revision is the commit named in the covering email, which was the newest commit of the branch `post-feedback-revision` when the email was sent.

Checked on 20 October 2026 with the pinned libraries (`requirements.txt`) on Python 3.11, using a copy of the raw file whose MD5 matches section 2:

- `python src/audit.py` reproduces `reports/audit_report.md` exactly.
- `python src/run_stage1.py` and `python src/run_stage2.py` regenerate the prepared tables, the split manifest, the folds, the out-of-fold table and the development results with the same values. The saved reports differ only in the recorded Python version.
- `notebooks/model_notebook.ipynb`, run again on Python 3.9, gives the same outputs. Its saved outputs were regenerated so that no table preview shows the risk label.
- Without the raw file, all 55 tests pass, because the prepared tables and the saved models are in the repository.

## 8. Reproduced results and the earlier pilot

Pilot numbers in earlier versions of the proposal are not used as results. The numbers to quote are the ones this code regenerates from the pinned file: `reports/audit_report.md` (counts, exclusions, missingness, partitions), `reports/stage1_results.json` and `reports/stage2_dev_results.md` (development results). The proposal's Tables 4 and 5 will be replaced by them. The locked test has not been run.

## 9. History of the raw file in this repository (as of 20 October 2026)

Before this commit the raw file was stored in `data/raw/`. It was added once, in commit `a648f5c` (2 October 2026), and it is the same file in every place it appears. It remains in the `main` branch, in the earlier commits of the `post-feedback-revision` branch and in the release `initial-demo-submitted`. The repository has no forks, stars or watchers. Removing the file from the current version does not remove it from that history. Whether to rewrite the history is the supervisor's decision, and rewriting would change commit hashes, including the submitted release.
