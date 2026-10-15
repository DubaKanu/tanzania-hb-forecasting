# Deployment Plan

## What exists now

- A local FastAPI service (`app/main.py`) with an interactive Swagger page at `/docs`.
- `POST /forecast` returns the Stage 1 haemoglobin forecast, the Stage 2 estimated error, a kept or set-aside flag, the cutoff used and a research-only disclaimer. The service stores nothing it receives and does not log the inputs.
- It loads the saved models in `models/` and does not retrain when it starts.
- It is checked by eight acceptance checks (`docs/acceptance_checks.md`).
- There is no mobile app, SMS or hospital integration. This is a retrospective computational study, and the supervisor-approved scope is a simple demonstration.

## Next: host the demo on Render (planned, not done yet)

The plan for the defense is to host the demo on Render, show it running, and walk through the code. No hosted address exists yet. Nothing below has been done on Render.

### Before creating the service

1. Run `python src/run_stage1.py`, `python src/run_stage2.py` and `pytest tests/` on the final code. All tests must pass.
2. Make sure `models/stage1_model.joblib` and `models/stage2_model.joblib` are committed. Render builds from the repo and the API only loads these files.

### Settings (a web service on the free plan)

| Setting | Value |
|---|---|
| Language | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Instance type | Free |
| Environment variable | `PYTHON_VERSION` = `3.11.9` |
| Health check path | `/health` |

`render.yaml` in the repo holds the same settings, for use as a Render Blueprint. It names the free plan on purpose, because a Blueprint that leaves the plan out creates a paid instance.

### Why the Python version is set

Render's default Python for services created on or after 11 February 2026 is 3.14.3. The libraries in `requirements.txt` are pinned to the versions the results were produced with, and some of them (numpy 2.0.2 and scikit-learn 1.6.1) have no builds for Python 3.14, so a service left on the default would fail to build. The full version must be given, for example `3.11.9`. The same code and models were checked on Python 3.9, 3.10, 3.11 and 3.12 and give the same results.

### What to expect on the free plan

- A free web service spins down after 15 minutes without requests and starts again on the next request. The first request after a quiet period is slow. Open the `/health` address a few minutes before a demonstration.
- The free instance has 512 MB of memory. The API measured about 196 MB in use after loading both models and answering 200 requests.

### Checks after it is hosted

1. `GET /health` returns `{"status": "ok", ...}`.
2. `POST /forecast` with the sample request from the README returns the same numbers as the local service (forecast 10.56, estimated error 0.492, kept).
3. A request with Hb of 25 returns a 422 error.
4. Add the hosted address to the README.

### Using the hosted demo

Use made-up values only. The app does not store or print what it receives, and the demo is for showing how the system works, not for real women's data.

If the hosted demo is slow or unavailable during the defense, the same code and models run locally with `python3 -m uvicorn app.main:app`.

## What is explicitly out of scope for this capstone

- A production mobile application, SMS alerting, or live hospital integration. These were removed from the core deliverable during proposal review, since an eight-week capstone cannot responsibly support a field pilot on top of a rigorous modelling study.
- Any claim about performance outside this specific Tanzanian cohort, and any clinical use. The disclaimer field reflects this on every response.

## Longer-term, if this work continues past the capstone

- Containerize the API (Dockerfile) for easier reproducibility.
- Add authentication and request logging only after a formal ethics and data-governance review, before any health-worker-facing pilot is considered.
- Test the models on data from another time period or another site before any claim of usefulness.

## Sources

- Render, [Setting your Python version](https://render.com/docs/python-version)
- Render, [Blueprint YAML reference](https://render.com/docs/blueprint-spec)
- Render, [Your first Render deploy](https://render.com/docs/your-first-deploy)
