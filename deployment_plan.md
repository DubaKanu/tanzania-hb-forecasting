# Deployment Plan

## What exists now

- A FastAPI service (`app/main.py`) with an interactive Swagger page at `/docs`. It runs locally, and since 18 October 2026 it also runs on Render at https://tanzania-hb-forecast-demo.onrender.com.
- `POST /forecast` returns the Stage 1 haemoglobin forecast, the Stage 2 estimated error, a kept or set-aside flag, the cutoff used and a research-only disclaimer. The service stores nothing it receives and does not log the inputs.
- It loads the saved models in `models/` and does not retrain when it starts.
- It is checked by eight acceptance checks (`docs/acceptance_checks.md`). The live copy was also checked by hand on 18 October 2026 (see below).
- There is no mobile app, SMS or hospital integration. This is a retrospective computational study, and the supervisor-approved scope is a simple demonstration.

## Hosted on Render

The demo is hosted on Render's free plan at https://tanzania-hb-forecast-demo.onrender.com (the Swagger page is at `/docs`). It was created on 18 October 2026 as a Blueprint from `render.yaml`, using the `post-feedback-revision` branch of the GitHub repo. By default, each new commit on that branch starts a new build.

### How it was set up

1. The models were made with `python src/run_stage1.py` and `python src/run_stage2.py`, and `models/stage1_model.joblib` and `models/stage2_model.joblib` are committed in the repo. Render builds from the repo, and the API only loads these two files.
2. In Render: New, then Blueprint, then the GitHub repo and the branch `post-feedback-revision`. Render read `render.yaml` and created the web service `tanzania-hb-forecast-demo`.

### Settings (a web service on the free plan)

| Setting | Value |
|---|---|
| Language | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Instance type | Free |
| Environment variable | `PYTHON_VERSION` = `3.11.9` |
| Health check path | `/health` |

`render.yaml` in the repo holds the same settings. It names the free plan on purpose, because a Blueprint that leaves the plan out creates a paid instance.

### Why the Python version is set

Render's default Python depends on when a service was created. It was 3.14.3 for services created from 11 February 2026, and it is 3.14.8 for services created on or after 12 October 2026, which includes this one. The libraries in `requirements.txt` are pinned to the versions the results were produced with. numpy 2.0.2 and scikit-learn 1.6.1 have no ready-made builds for Python 3.14 (checked on 18 October 2026), so on the default Python, pip would have to build them from source. Setting `PYTHON_VERSION` avoids that. The full version must be given, for example `3.11.9`. The same code and models were checked on Python 3.9, 3.10, 3.11 and 3.12 and give the same results.

### What to expect on the free plan

- A free web service spins down after 15 minutes without requests and starts again on the next request. The first request after a quiet period is slow, so open the `/health` address a few minutes before a demonstration.
- The free instance has 512 MB of memory. The API measured about 196 MB in use after loading both models and answering 200 requests.
- `GET /health` returns `{"status": "ok", "model_loaded": false}` until the first valid forecast has been made. The two models are read from disk at that point, not when the service starts, and after that `/health` returns `"model_loaded": true`. It goes back to `false` after the service has slept and woken up. `false` is normal and is not a fault.
- A request that fails the input check (for example an Hb of 25) is refused with a 422 error before any model code runs, so it does not load the models.
- Each new commit on the branch starts a new build. While it builds, the address can be slow or unavailable, so do not change the branch just before a demonstration.

### Checks on the live copy

Run by hand on 18 October 2026 against https://tanzania-hb-forecast-demo.onrender.com, in a browser and in the Swagger page. At the time, the live copy was built from commit `807acd6`.

| # | Check | Result |
|---|---|---|
| 1 | `GET /health` returns `"status": "ok"` | Passed. It showed `"model_loaded": false` before the first forecast, as designed |
| 2 | `POST /forecast` with the sample request in the README returns the same numbers as the local service | Passed. Status 200, forecast 10.56, estimated error 0.492, kept, cutoff 0.915 |
| 3 | A request with Hb of 25 returns a 422 error | Passed. The message was "Input should be less than or equal to 20" |
| 4 | `GET /health` after the first forecast | Passed. It showed `"model_loaded": true` |
| 5 | The hosted address is in the README | Done |

### Using the hosted demo

Use made-up values only. The app does not store or print what it receives, and the demo is for showing how the system works, not for real women's data.

To try it, open `/docs`, choose `POST /forecast`, click Try it out, paste the sample request from the README and click Execute.

If the hosted demo is slow or unavailable during the defense, the same code and models run locally with `python3 -m uvicorn app.main:app`.

## What is explicitly out of scope for this capstone

- A production mobile application, SMS alerting, or live hospital integration. These were removed from the core deliverable during proposal review, since an eight-week capstone cannot responsibly support a field pilot on top of a rigorous modelling study.
- Any claim about performance outside this specific Tanzanian cohort, and any clinical use. The disclaimer field reflects this on every response.

## Longer-term, if this work continues past the capstone

- Containerize the API (Dockerfile) for easier reproducibility.
- Add authentication and request logging only after a formal ethics and data-governance review, before any health-worker-facing pilot is considered.
- Test the models on data from another time period or another site before any claim of usefulness.

## Sources

Read on 18 October 2026.

- Render, [Setting your Python version](https://render.com/docs/python-version)
- Render, [Blueprint YAML reference](https://render.com/docs/blueprint-spec)
- Render, [Your first Render deploy](https://render.com/docs/your-first-deploy)
