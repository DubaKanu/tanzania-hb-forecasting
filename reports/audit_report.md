# Data preparation audit

- Raw file: maternal_dataset_csv.csv, Zenodo 10.5281/zenodo.15309733, Version 1
- MD5 computed: 0253d2d385dd80d6c9bb82df6b85fcc8
- MD5 expected: 0253d2d385dd80d6c9bb82df6b85fcc8  ->  MATCH

## Exclusions

| Step | Rows |
| --- | --- |
| All rows in the release | 8,817 |
| Missing a first- or second-contact haemoglobin | 6,036 |
| Haemoglobin outside 3 to 20 g/dL | 11 |
| Gestational week outside 1 to 42 | 5 |
| Second contact not later than first | 80 |
| Eligible paired records | 2,685 |

## Missingness among the ten inputs (eligible records)

| Input | Missing | % |
| --- | --- | --- |
| hb_visit1_g_dl | 0 | 0.0 |
| gestation_visit1_weeks | 0 | 0.0 |
| maternal_age_years | 0 | 0.0 |
| pregnancy_count | 0 | 0.0 |
| height_cm | 0 | 0.0 |
| weight_visit1_kg | 0 | 0.0 |
| pulse_visit1_bpm | 132 | 4.9 |
| respiratory_rate_visit1 | 643 | 23.9 |
| systolic_bp_visit1 | 18 | 0.7 |
| diastolic_bp_visit1 | 18 | 0.7 |

Complete on all ten inputs: 2,025 of 2,685
Median first haemoglobin (eligible): 11.0 g/dL

## Partitions

- Train / development / test: 1,884 / 402 / 399 (expected 1,884 / 402 / 399)
- Rows sharing an identical ten-input vector with another row: 2
- Identical-vector groups split across partitions: 0

Grouping by identical vectors is a proxy for the same woman. Without a verified patient identifier it cannot guarantee that partitions are independent.
