# Performance and compatibility

What was measured, on what, and what was not. The raw numbers are in `reports/performance_*.json`. Everything below comes from one Linux machine unless it says otherwise.

## Where it was measured

| | |
|---|---|
| Machine | Cloud Linux container: 2 vCPU (Intel Xeon @ 2.80 GHz), 8 GB RAM, Linux 6.18 |
| Software | Python 3.11.17 with the pinned `requirements.txt`: numpy 2.0.2, scikit-learn 1.6.1, FastAPI 0.115.6, uvicorn 0.34.0 |
| Server | One `uvicorn app.main:app` process, started by the benchmark script. The client runs on the same machine |
| Requests | 300 requests one at a time after 10 warm-up requests, then 1, 4 and 16 clients at once. The values are real first-visit values from the development file |

## Speed of the demo API

"Before" is the code as first reorganised into classes. "After" adds the fix described below. Same machine, same script.

| | Before | After |
|---|---|---|
| One request at a time, median | 73.5 ms | 11.1 ms |
| 95th percentile | 107.4 ms | 17.2 ms |
| 99th percentile | 115.0 ms | 19.5 ms |
| Requests per second, 1 client | 12.6 | 68.8 |
| Requests per second, 4 clients | 8.2 | 38.7 |
| 16 clients at once, median wait | 2,014 ms | 480 ms |
| Start until `/health` answers | 1.6 s | 1.5 s |
| First forecast request (loads the two model files) | 192 ms | 82 ms |
| Memory of the server after the load test | 183 MB | 182 MB |
| 402 development women forecast in one call | 0.118 s | 0.022 s |

With the server on one CPU core (`taskset`) and the load generator on the other: median 10.5 ms, 95th percentile 16.8 ms, and about 80 requests per second with one client or four. In the two-core runs the load generator and the server share the two cores, so the many-clients figures understate what the server can do.

Memory is far below the 512 MB of Render's free plan (about 173 MB when the server has started, 179 MB after the two models load).

### What was found and fixed

A profile of one forecast, in-process (`TwoStageForecaster.forecast` on one woman):

| Step | Before | After |
|---|---|---|
| Stage 1: forecast and tree disagreement (includes preparing the inputs) | 48.0 ms | 4.7 ms |
| Stage 2: build its 12 inputs | 2.2 ms | 2.2 ms |
| Stage 2: estimate the error | 16.7 ms | 1.8 ms |
| One forecast, measured as a whole | 73.7 ms | 7.3 ms |

The parts were timed one by one and the whole separately, so they do not add up exactly.

Almost all of the time was in three passes over the 200 trees of a forest. For every tree, scikit-learn checked its input again, and for one woman that checking cost far more than the forecast. `tree_predictions` in `src/stage1.py` converts the inputs once to the float32 the trees use and asks each fitted tree directly. `forest_mean` adds the trees up one at a time in the order scikit-learn uses, so the forecast is the same to the last digit. One pass now gives both the forecast and the tree disagreement.

No result changed. The new route was compared with scikit-learn's own `predict` on all 1,884 training and 402 development women: the forecasts, the tree disagreement and the Stage 2 estimates were identical (`numpy.array_equal`, not "close"). `tests/test_stage1_models.py`, `tests/test_stage2.py` and `tests/test_forecaster.py` repeat that comparison, and `python scripts/check_reproducibility.py` rebuilds all 19 derived files from the raw file and finds them unchanged (see below).

## Compatibility

The pinned libraries were installed in a new environment for each Python version. The tests were run, and every derived file was rebuilt from the raw file and compared with the files in the repository (`scripts/check_reproducibility.py`).

| Python | Install | Tests | Rebuild from the raw file (19 files) |
|---|---|---|---|
| 3.9.25 | ok | all pass | 19 identical, byte for byte |
| 3.10.20 | ok | all pass | 14 identical, 5 differ only in the saved Python version |
| 3.11.17 | ok | all pass | 14 identical, 5 differ only in the saved Python version |
| 3.12.13 | ok | all pass | 14 identical, 5 differ only in the saved Python version |
| 3.13.17 | fails | not run | not run |

The files in the repository were built on Python 3.9.25 by the code as it was before it was reorganised into classes, so the 3.9.25 row is the strongest result: the reorganised code reproduces all 19 files byte for byte. On the other versions, five files hold the Python version that built them (`versions.python` in three JSON files and in `models/stage2_model.joblib`, and one line of `reports/stage2_dev_results.md`) and differ only in that text. The numbers in them, and every tree of both models, are equal.

Python 3.13 cannot install the pinned libraries: `scipy==1.13.1` has no ready-made build for 3.13, so pip tries to build it from source, which needs a Fortran compiler (`gfortran`) that the test machine does not have. The code itself was not tested on 3.13, and the README asks for Python 3.9 to 3.12.

The hosted demo runs on Render with Python 3.11.9 (see `deployment_plan.md`). A GitHub Actions workflow (`.github/workflows/tests.yml`) repeats the four-version test run and the style check on every upload.

An earlier finding from comparing environments: scikit-learn's `GroupKFold` changed its tie-breaking between versions and moved the cross-validation error by about 0.003 g/dL, so the code uses its own fixed fold rule (`assign_cv_folds` in `src/pipeline.py`).

## What was not measured

- **The hosted copy on Render's free plan.** It has a smaller share of a CPU than the machine above, so its requests will be slower, and the first request after 15 minutes of silence waits for the service to wake up. To measure it: `python scripts/benchmark_api.py --url https://tanzania-hb-forecast-demo.onrender.com --label render-free`.
- **macOS and Windows.** Everything here ran on Linux.
- **Other hardware.** The only machine is the one in the first table.

## Repeat it

```
python scripts/benchmark_api.py --start-local --label my-machine   # speed and memory of the API
python scripts/benchmark_api.py --url https://tanzania-hb-forecast-demo.onrender.com --label render-free
python -m pytest tests -q                                          # all tests
python scripts/check_reproducibility.py                            # needs the raw file, see data/raw/README.md
```

The benchmark writes `reports/performance_<label>.json` and prints a table. It sends only made-up or development-set values and stores nothing else.
