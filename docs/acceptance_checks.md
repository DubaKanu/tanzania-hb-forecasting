# Acceptance checks for the demonstration API

These are the eight checks agreed for the demo. Checks 1 to 7 are automated in `tests/test_api.py`. Check 8 is done by building a new environment from the README alone.

Updated on 23 October 2026: the API was split into small modules (`app/main.py`, `app/schemas.py`, `app/service.py`, `app/errors.py`) and tested much more heavily. The eight checks below are unchanged and still pass. The extra tests are described in "More input testing" below, and the whole suite is now 316 tests in 16 files.

The API tests use the saved models and the prepared tables, which are in the repository, so `pytest tests/` runs without the raw data file. To regenerate everything from scratch, first download the raw file as described in `data/raw/README.md`, then run `python src/audit.py`, `python src/run_stage1.py` and `python src/run_stage2.py`.

| # | Check | How it is tested |
|---|---|---|
| 1 | A valid input returns HTTP 200 and a value between 3 and 20 g/dL | `test_check1_valid_input_gives_200_and_a_value_between_3_and_20` |
| 2 | An out-of-range or missing required input is rejected with HTTP 422 | `test_check2_out_of_range_input_is_rejected_with_422` (8 cases: Hb 25 and 1, week 60, age 5, pregnancy count -1, height 20, weight 500, Hb as text) and `test_check2_missing_required_input_is_rejected_with_422` (each of the 6 required inputs left out). The same refusal is also tested for every input at and just beyond its limits, for NaN, Infinity and huge numbers, and for malformed requests (see "More input testing") |
| 3 | The same input always gives the same output | `test_check3_the_same_input_always_gives_the_same_output` (4 identical requests, identical responses) |
| 4 | The output equals the offline prediction for the same row, within rounding | `test_check4_output_equals_the_offline_prediction_for_the_same_row` (forecast within 0.0051 g/dL, estimated error within 0.0006 g/dL) and `test_check4_output_matches_the_saved_development_results_for_real_women` (8 real development women against `reports/stage2_dev_table.csv`) |
| 5 | Leaving out the optional inputs still returns a value | `test_check5_leaving_out_the_optional_inputs_still_gives_a_value` |
| 6 | Every response includes the disclaimer | `test_check6_every_response_carries_the_disclaimer` (4 different requests) |
| 7 | The service stores no inputs | `test_check7_the_service_stores_no_inputs`: nothing in the repository is created or changed by a request, the submitted value is not in the logs, and `app/main.py` contains no code that writes files or logs |
| 8 | A fresh machine can run it using only the README commands | A new virtual environment on a clean copy of the repository, then the README steps in order (`pip install -r requirements.txt`, download the raw file and check its checksum, `python src/audit.py`, `python src/run_stage1.py`, `python src/run_stage2.py`, `pytest tests/`, then the API) |

`tests/test_api.py` also checks that `/` and `/health` respond.

## More input testing (added 23 October 2026)

`tests/test_api_inputs.py` (148 tests) sends the API a wide range of inputs. `tests/test_api_errors.py` (6 tests) covers a missing model file and the order in which the service checks things, and `tests/test_service.py` (14 tests) tests the forecasting logic without the web layer.

| What is sent | What must happen | Cases |
|---|---|---|
| The lowest and the highest allowed value of each of the 10 inputs | 200, and a forecast between 3 and 20 | 20 |
| A value 0.01 below the lowest or above the highest of each input | 422 | 20 |
| `NaN`, `Infinity` and `-Infinity` in each input | 422 that names the input | 30 |
| `1e39`, `1e300` and `-1e300` in each input | 422, no crash | 30 |
| Text, a list, an object, `true`, `"10,8"` or an empty string as the haemoglobin | 422 | 6 |
| `null` for each of the 6 required inputs | 422 | 6 |
| `null` for each of the 4 optional inputs | Treated as left out, 200 | 4 |
| All 16 patterns of left-out optional inputs | 200 and exactly the six documented response fields | 16 |
| Broken or wrong-shaped bodies: invalid JSON, nothing, a list, `null`, `{}`, a bare string, form data | 422 | 7 |
| Extra fields the API does not know | Ignored, the answer does not change | 1 |
| `GET`, `PUT` or `DELETE` on `/forecast`, and an unknown path | 405 and 404 | 1 |
| Models missing from `models/` | `/forecast` answers 503 with instructions, `/health` still answers, and a bad request is still a 422 | 3 |

The same file also checks that error messages never repeat the value that was sent, that the response has exactly the documented fields and rounding, that the flag agrees with the cutoff and with the sentence in the note, that the generated OpenAPI description lists both routes and the limits, and that the limits accept every woman in the training and development files.

**A real defect was found.** NaN, Infinity or a huge number in eight of the inputs made the API answer with HTTP 500 and not a 422. In some cases the value reached the model, and in others the error response tried to write the value back as JSON, which is impossible for NaN. The first submitted version did the same; this was checked by running it. Every input now has a lower and an upper limit, `allow_inf_nan=False` refuses NaN and Infinity, and the 422 message leaves out the value sent (`app/errors.py`). These cases are kept as regression tests.

## Last verification (check 8)

Run on 14 October 2026 on a clean copy of the repository, in a new virtual environment built from `requirements.txt`, using only the README commands.

- Python 3.9.25: all 55 tests passed. The regenerated reports, figures and models were identical to the saved ones. The notebook ran top to bottom without errors and gave the same outputs. The API answered the sample request with forecast 10.56, estimated error 0.492, kept, and refused an Hb of 25 with a 422 error.
- With the saved models, the 55 tests also passed on Python 3.10, 3.11 and 3.12. Retraining on those versions gave the same results as on 3.9.
- Not tested: macOS or Windows. The hosted copy on Render was checked separately (next section).

Re-checked on 20 October 2026, after the raw file was taken out of the repository, with the pinned libraries on Python 3.11 and a copy of the raw file whose MD5 matches the Zenodo listing: the audit, Stage 1 and Stage 2 gave the same tables and numbers as before, and the notebook gave the same outputs on Python 3.9. Without the raw file, all 55 tests pass. The details are in `docs/data_provenance.md`, section 7.

Re-checked on 23 October 2026, on the finished final version (code organised into classes, API inputs limited, forecast path faster), with the pinned libraries on Linux:

- All 316 tests pass on Python 3.9.25, 3.10.20, 3.11.17 and 3.12.13, with no raw file present.
- `ruff check .` passes.
- `python scripts/check_reproducibility.py` rebuilt the 19 derived files from the raw file (checksum matching the Zenodo listing). On Python 3.9.25, 19 are identical byte for byte. On 3.10.20, 3.11.17 and 3.12.13, 14 are identical and 5 are identical except for the Python version saved inside them.
- The acceptance checks 1 to 7 are part of the 316 tests. Check 8 and the hosted copy were not repeated by hand on 23 October. The hosted copy answered `GET /health` with `"status": "ok"` that day.

## Hosted copy on Render (checked 18 October 2026)

The demo is live at https://tanzania-hb-forecast-demo.onrender.com. These checks were run by hand on 18 October 2026, in a browser and in the Swagger page at `/docs`, using the sample request in the README. At the time, the live copy was built from commit `807acd6`.

| Check on the live copy | Result |
|---|---|
| `GET /health` answers with `"status": "ok"` | Passed. It showed `"model_loaded": false` before the first forecast, as designed |
| The sample request returns 200 with the same numbers as the local service | Passed. Forecast 10.56, estimated error 0.492, kept, cutoff 0.915 |
| An Hb of 25 is refused (acceptance check 2) | Passed. Status 422, "Input should be less than or equal to 20" |
| `GET /health` after the first forecast | Passed. It showed `"model_loaded": true` |

The other checks (same input gives the same output, optional inputs left out, the disclaimer on every response, nothing stored) were not repeated by hand on the live copy. They run in `tests/test_api.py` on the same code. `deployment_plan.md` explains why `model_loaded` starts as `false`.

## What the checks do not cover

They show that the service behaves as described and matches the offline code. They do not show that a forecast is clinically useful. The research results are in `reports/stage2_dev_results.md`.
