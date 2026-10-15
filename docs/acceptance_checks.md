# Acceptance checks for the demonstration API

These are the eight checks agreed for the demo. Checks 1 to 7 are automated in `tests/test_api.py`. Check 8 is done by building a new environment from the README alone.

The API tests need the saved models. From the repository root, run `python src/run_stage1.py` and `python src/run_stage2.py` first, then `pytest tests/`.

| # | Check | How it is tested |
|---|---|---|
| 1 | A valid input returns HTTP 200 and a value between 3 and 20 g/dL | `test_check1_valid_input_gives_200_and_a_value_between_3_and_20` |
| 2 | An out-of-range or missing required input is rejected with HTTP 422 | `test_check2_out_of_range_input_is_rejected_with_422` (8 cases: Hb 25 and 1, week 60, age 5, pregnancy count -1, height 20, weight 500, Hb as text) and `test_check2_missing_required_input_is_rejected_with_422` (each of the 6 required inputs left out) |
| 3 | The same input always gives the same output | `test_check3_the_same_input_always_gives_the_same_output` (4 identical requests, identical responses) |
| 4 | The output equals the offline prediction for the same row, within rounding | `test_check4_output_equals_the_offline_prediction_for_the_same_row` (forecast within 0.0051 g/dL, estimated error within 0.0006 g/dL) and `test_check4_output_matches_the_saved_development_results_for_real_women` (8 real development women against `reports/stage2_dev_table.csv`) |
| 5 | Leaving out the optional inputs still returns a value | `test_check5_leaving_out_the_optional_inputs_still_gives_a_value` |
| 6 | Every response includes the disclaimer | `test_check6_every_response_carries_the_disclaimer` (4 different requests) |
| 7 | The service stores no inputs | `test_check7_the_service_stores_no_inputs`: nothing in the repository is created or changed by a request, the submitted value is not in the logs, and `app/main.py` contains no code that writes files or logs |
| 8 | A fresh machine can run it using only the README commands | A new virtual environment on a clean copy of the repository, then the README steps in order (`pip install -r requirements.txt`, `python src/audit.py`, `python src/run_stage1.py`, `python src/run_stage2.py`, `pytest tests/`, then the API) |

`tests/test_api.py` also checks that `/` and `/health` respond.

## Last verification (check 8)

Run on 14 October 2026 on a clean copy of the repository, in a new virtual environment built from `requirements.txt`, using only the README commands.

- Python 3.9.25: all 55 tests passed. The regenerated reports, figures and models were identical to the saved ones. The notebook ran top to bottom without errors and gave the same outputs. The API answered the sample request with forecast 10.56, estimated error 0.492, kept, and refused an Hb of 25 with a 422 error.
- With the saved models, the 55 tests also passed on Python 3.10, 3.11 and 3.12. Retraining on those versions gave the same results as on 3.9.
- Not tested: macOS or Windows, and a hosted copy on Render. The hosted copy is checked once it exists (see `deployment_plan.md`).

## What the checks do not cover

They show that the service behaves as described and matches the offline code. They do not show that a forecast is clinically useful. The research results are in `reports/stage2_dev_results.md`.
