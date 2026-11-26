#!/usr/bin/env python3
"""Measure how fast recurrence plots and their RQA are computed, window by window.

    python tools/benchmark_engine.py                       # sizes 1280..10240, 1 and all cores
    python tools/benchmark_engine.py --sizes 1280 2560 --jobs 1 8 38 --out bench.csv
    python tools/benchmark_engine.py --quick               # a smoke run, seconds

Every number in the decision record about speed comes from this script, so it
measures what an analysis does: many short windows of a three-dimensional
phase-amplitude trajectory, a threshold fixed per window at a 5% rate, the
default line floors, the full RQA. Two measurements per window size:

* **one window**, best of three: the time to estimate the threshold, and the
  time to build the plot and accumulate its line histograms, which is where
  the RQA spends its effort;
* **throughput**: ``batch_windowed_recurrence(...).metrics(rqa=True,
  n_jobs=k)`` over several records, as a study would call it, in windows per
  second for every ``k`` asked for.

The trajectories are synthetic, with fixed seeds, so runs on two machines
measure the machines and not the data. Results go to a CSV file (one row per
measurement, with the versions and the machine) and the printed table to a
``.txt`` file next to it.

Thread counts: with several jobs, each worker should use one BLAS thread, or
the workers fight over the cores. Set ``OPENBLAS_NUM_THREADS=1`` (and
``MKL_NUM_THREADS=1`` with MKL) before running; the CSV records the BLAS
thread count each run saw.
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import platform
import sys
import time
import warnings

import numpy as np

import recurra as rc

FIELDS = ["recurra", "engine", "device", "measurement", "n_points", "dim",
          "records", "windows", "n_jobs", "seconds", "windows_per_s",
          "threshold_s", "plot_and_histogram_s", "python", "numpy",
          "cpu_count", "blas_threads", "platform"]


def trajectory(n: int, seed: int) -> np.ndarray:
    """A phase-amplitude trajectory, ``(cos phi, sin phi, A)``, offset from 0."""
    g = np.random.default_rng(seed)
    t = np.arange(n) / 128.0
    phi = 2 * np.pi * 6 * t + 0.3 * np.cumsum(g.standard_normal(n)) / 30
    amp = 1 + 0.5 * np.cos(phi) + 0.2 * g.standard_normal(n)
    return np.column_stack([np.cos(phi), np.sin(phi), amp]) + 3.0


def blas_threads() -> str:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return os.environ.get("OPENBLAS_NUM_THREADS", "unknown")
    counts = {str(p.get("num_threads")) for p in threadpool_info()
              if p.get("user_api") == "blas"}
    return "/".join(sorted(counts)) or "unknown"


def best_of(fn, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def one_window(n: int) -> tuple[float, float]:
    X = trajectory(n, seed=0)
    th = rc.threshold(X, target_rr=0.05, rng=0)
    t_threshold = best_of(lambda: rc.threshold(X, target_rr=0.05, rng=0))

    def plot_and_histogram():
        rm = rc.recurrence_plot(X, th, store="stream")
        rc.rqa(rm)

    return t_threshold, best_of(plot_and_histogram)


def throughput(n: int, records: int, windows: int, n_jobs: int) -> float:
    sources = {f"r{k:02d}": trajectory(n * windows, seed=100 + k)
               for k in range(records)}
    spec = rc.WindowSpec(size=n, step=n, unit="samples")
    t0 = time.perf_counter()
    batch = rc.batch_windowed_recurrence(sources, spec, target_rr=0.05, rng=0,
                                         n_jobs=n_jobs)
    table = batch.metrics(rqa=True, n_jobs=n_jobs)
    seconds = time.perf_counter() - t0
    if len(table) != records * windows:
        raise RuntimeError(f"expected {records * windows} windows, got {len(table)}")
    return seconds


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=[1280, 2560, 5120, 10240],
                   help="points per window")
    p.add_argument("--jobs", type=int, nargs="+", default=None,
                   help="values of n_jobs for the throughput runs (default: 1 and all cores)")
    p.add_argument("--records", type=int, default=None,
                   help="records per throughput run (default: the largest n_jobs)")
    p.add_argument("--windows", type=int, default=4, help="windows per record")
    p.add_argument("--out", default="benchmark_engine.csv", help="CSV file to write")
    p.add_argument("--quick", action="store_true",
                   help="two small sizes and one job, to check the script runs")
    a = p.parse_args(argv)

    warnings.simplefilter("ignore", rc.RecurraWarning)
    cores = os.cpu_count() or 1
    if a.quick:
        a.sizes, a.jobs, a.windows = [300, 600], [1], 2
    jobs = a.jobs or sorted({1, cores})
    records = a.records or max(jobs)
    common = {"recurra": rc.__version__, "engine": "numpy", "device": "cpu",
              "dim": 3, "python": platform.python_version(),
              "numpy": np.__version__, "cpu_count": cores,
              "blas_threads": blas_threads(), "platform": platform.platform()}

    rows, lines = [], []

    def say(text: str) -> None:
        print(text, flush=True)
        lines.append(text)

    say(f"recurra {rc.__version__}, numpy {np.__version__}, {cores} cores, "
        f"BLAS threads {common['blas_threads']}")
    for n in a.sizes:
        t_th, t_plot = one_window(n)
        rows.append({**common, "measurement": "one_window", "n_points": n,
                     "records": 1, "windows": 1, "n_jobs": 1,
                     "seconds": t_th + t_plot, "windows_per_s": 1.0 / (t_th + t_plot),
                     "threshold_s": t_th, "plot_and_histogram_s": t_plot})
        say(f"N={n:6d}  one window: threshold {t_th:.4f} s, plot + histograms "
            f"{t_plot:.4f} s")
        for k in jobs:
            seconds = throughput(n, records, a.windows, k)
            total = records * a.windows
            rows.append({**common, "measurement": "throughput", "n_points": n,
                         "records": records, "windows": total, "n_jobs": k,
                         "seconds": seconds, "windows_per_s": total / seconds,
                         "threshold_s": "", "plot_and_histogram_s": ""})
            say(f"N={n:6d}  {total} windows, n_jobs={k:3d}: {seconds:8.2f} s, "
                f"{total / seconds:8.2f} windows/s")

    out = pathlib.Path(a.out)
    say(f"written: {out} and {out.with_suffix('.txt')}")
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
