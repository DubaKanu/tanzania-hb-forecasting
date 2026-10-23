"""
Speed and memory of the demonstration API. Run from the repository root.

Against a server this script starts itself (it needs the saved models in models/):

    python scripts/benchmark_api.py --start-local --label my-laptop

Against a server that is already running, for example the hosted one:

    python scripts/benchmark_api.py --url https://YOUR-SERVICE.onrender.com --label render-free

What it measures
  1. Start-up: seconds from starting the server until /health answers, and the first
     forecast request (which loads the two model files) against the ones after it.
  2. Warm latency of one request at a time: median, 95th and 99th percentile.
  3. Several clients at once (1, 4 and 16): requests per second and latency.
  4. Memory of the server process after start-up, after the models load and after the load test
     (only when the script starts the server itself).
  5. The offline forecaster on the 402 development women, if the data files are present.

It sends only made-up or development-set values and stores nothing except the numbers it
measures, which it prints and saves to reports/performance_<label>.json.
"""
import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import httpx

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEV_FILE = os.path.join(ROOT, "data", "Josephine_African_Data_Review", "Tanzania_Followup_Haemoglobin",
                        "validation.csv")
INPUTS = ["hb_visit1_g_dl", "gestation_visit1_weeks", "maternal_age_years", "pregnancy_count", "height_cm",
          "weight_visit1_kg", "pulse_visit1_bpm", "respiratory_rate_visit1", "systolic_bp_visit1",
          "diastolic_bp_visit1"]
SAMPLE = {"hb_visit1_g_dl": 10.8, "gestation_visit1_weeks": 20, "maternal_age_years": 28, "pregnancy_count": 2,
          "height_cm": 160, "weight_visit1_kg": 62, "pulse_visit1_bpm": 80, "respiratory_rate_visit1": 18,
          "systolic_bp_visit1": 118, "diastolic_bp_visit1": 76}


@dataclass
class LatencyStats:
    """Summary of a list of response times, in milliseconds."""

    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


def summarise(seconds: List[float]) -> LatencyStats:
    ms = sorted(s * 1000.0 for s in seconds)

    def percentile(q: float) -> float:
        return ms[min(len(ms) - 1, int(round(q * (len(ms) - 1))))]

    return LatencyStats(n=len(ms), mean_ms=round(statistics.fmean(ms), 1), p50_ms=round(percentile(0.50), 1),
                        p95_ms=round(percentile(0.95), 1), p99_ms=round(percentile(0.99), 1),
                        max_ms=round(ms[-1], 1))


def make_payloads(count: int = 50) -> List[dict]:
    """Different valid requests, so no response is simply repeated. Real development values if present."""
    payloads = [SAMPLE]
    if os.path.exists(DEV_FILE):
        import pandas as pd  # only needed here
        dev = pd.read_csv(DEV_FILE)
        for _, row in dev.head(count).iterrows():
            payload = {c: float(row[c]) for c in INPUTS if not pd.isna(row[c])}
            if 3 <= payload["hb_visit1_g_dl"] <= 20 and 100 <= payload["height_cm"] <= 200 \
                    and 30 <= payload["weight_visit1_kg"] <= 150:
                payloads.append(payload)
    return payloads


def post_once(client: httpx.Client, base: str, payload: dict) -> float:
    start = time.perf_counter()
    response = client.post(f"{base}/forecast", json=payload, timeout=120)
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return elapsed


def rss_mb(pid: int) -> Optional[float]:
    """Resident memory of a process in MB, from `ps` (works on macOS and Linux)."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True)
        return round(int(out.stdout.strip()) / 1024.0, 1)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalServer:
    """Starts uvicorn on a free port for the length of a `with` block."""

    def __init__(self, server_core: Optional[int] = None):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.server_core = server_core
        self.process: Optional[subprocess.Popen] = None

    def command(self) -> List[str]:
        """The command that starts the server, optionally pinned to one CPU core (Linux only)."""
        command = [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port",
                   str(self.port), "--log-level", "warning"]
        if self.server_core is not None:
            command = ["taskset", "-c", str(self.server_core)] + command
        return command

    def __enter__(self):
        self.started_at = time.perf_counter()
        self.process = subprocess.Popen(self.command(), cwd=ROOT, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
        return self

    def wait_until_ready(self, limit_s: float = 60.0) -> float:
        """Seconds from starting the process until /health answers."""
        while time.perf_counter() - self.started_at < limit_s:
            try:
                if httpx.get(f"{self.base}/health", timeout=1).status_code == 200:
                    return time.perf_counter() - self.started_at
            except httpx.HTTPError:
                time.sleep(0.05)
        raise RuntimeError("the server did not start in time")

    def __exit__(self, *exc):
        self.process.terminate()
        self.process.wait(timeout=10)


def measure_sequential(base: str, payloads: List[dict], count: int) -> LatencyStats:
    with httpx.Client() as client:
        times = [post_once(client, base, payloads[i % len(payloads)]) for i in range(count)]
    return summarise(times)


def measure_concurrent(base: str, payloads: List[dict], workers: int, per_worker: int) -> dict:
    """`workers` clients, each sending `per_worker` requests as fast as the server answers."""
    all_times: List[float] = []
    lock = threading.Lock()

    def client_loop(offset: int):
        with httpx.Client() as client:
            mine = [post_once(client, base, payloads[(offset + i) % len(payloads)]) for i in range(per_worker)]
        with lock:
            all_times.extend(mine)

    threads = [threading.Thread(target=client_loop, args=(w * 7,)) for w in range(workers)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - start
    return {"clients": workers, "requests": len(all_times), "requests_per_second": round(len(all_times) / wall, 1),
            **asdict(summarise(all_times))}


def measure_offline_batch() -> Optional[dict]:
    """The forecaster on the 402 development women, without the web layer."""
    sys.path.insert(0, os.path.join(ROOT, "src"))
    models = [os.path.join(ROOT, "models", f) for f in ("stage1_model.joblib", "stage2_model.joblib")]
    if not (os.path.exists(DEV_FILE) and all(os.path.exists(m) for m in models)):
        return None
    import pandas as pd
    from forecaster import TwoStageForecaster
    dev = pd.read_csv(DEV_FILE)
    forecaster = TwoStageForecaster.from_files(*models)
    forecaster.forecast(dev.head(5))                       # warm up
    start = time.perf_counter()
    forecaster.forecast(dev)
    seconds = time.perf_counter() - start
    return {"women": len(dev), "seconds": round(seconds, 3), "women_per_second": round(len(dev) / seconds, 1)}


def environment() -> Dict[str, object]:
    import numpy
    import sklearn
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "cpu_count": os.cpu_count(), "numpy": numpy.__version__, "scikit-learn": sklearn.__version__,
            "httpx": httpx.__version__}


def run_against(base: str, args, server: Optional[LocalServer], ready_s: Optional[float]) -> dict:
    payloads = make_payloads()
    result: dict = {"base_url": base if server is None else "local uvicorn"}
    if ready_s is not None:
        result["seconds_until_health_answers"] = round(ready_s, 2)
    if server is not None:
        result["rss_mb_after_start"] = rss_mb(server.process.pid)
    with httpx.Client() as client:
        result["first_request_ms"] = round(post_once(client, base, payloads[0]) * 1000, 1)
    if server is not None:
        result["rss_mb_after_models_loaded"] = rss_mb(server.process.pid)
    measure_sequential(base, payloads, 10)                  # warm-up
    result["sequential_warm"] = asdict(measure_sequential(base, payloads, args.requests))
    result["concurrent"] = [measure_concurrent(base, payloads, w, max(5, args.requests // (4 * w)))
                            for w in (1, 4, 16)]
    if server is not None:
        result["rss_mb_after_load_test"] = rss_mb(server.process.pid)
    return result


def print_table(label: str, result: dict) -> None:
    print(f"\n## {label}\n")
    for key in ("seconds_until_health_answers", "first_request_ms", "rss_mb_after_start",
                "rss_mb_after_models_loaded", "rss_mb_after_load_test"):
        if key in result:
            print(f"- {key.replace('_', ' ')}: {result[key]}")
    seq = result["sequential_warm"]
    print(f"- one request at a time ({seq['n']} requests): median {seq['p50_ms']} ms, 95th {seq['p95_ms']} ms, "
          f"99th {seq['p99_ms']} ms, max {seq['max_ms']} ms")
    print("\n| clients | requests | requests per second | median ms | 95th ms | 99th ms |")
    print("|---|---|---|---|---|---|")
    for row in result["concurrent"]:
        print(f"| {row['clients']} | {row['requests']} | {row['requests_per_second']} | {row['p50_ms']} | "
              f"{row['p95_ms']} | {row['p99_ms']} |")
    if result.get("offline_batch"):
        batch = result["offline_batch"]
        print(f"\nOffline forecaster, {batch['women']} development women: {batch['seconds']} s "
              f"({batch['women_per_second']} women per second)")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start-local", action="store_true", help="start uvicorn here and measure it")
    parser.add_argument("--url", help="measure a server that is already running (no memory figures)")
    parser.add_argument("--requests", type=int, default=200, help="requests for the one-at-a-time test")
    parser.add_argument("--label", default="run", help="name used in the output file")
    parser.add_argument("--server-core", type=int, default=None,
                        help="Linux only: pin the server to this CPU core (taskset), to see one-core behaviour")
    args = parser.parse_args(argv)
    if bool(args.start_local) == bool(args.url):
        parser.error("give exactly one of --start-local or --url")

    report: dict = {"label": args.label, "environment": environment(), "server_core": args.server_core}
    if args.start_local:
        with LocalServer(args.server_core) as server:
            ready = server.wait_until_ready()
            report["api"] = run_against(server.base, args, server, ready)
    else:
        report["api"] = run_against(args.url.rstrip("/"), args, None, None)
    report["api"]["offline_batch"] = measure_offline_batch() if args.start_local else None

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out_path = os.path.join(ROOT, "reports", f"performance_{args.label}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["environment"]))
    print_table(args.label, report["api"])
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
