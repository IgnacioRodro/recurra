#!/usr/bin/env python3
"""Measure how fast recurrence plots and their RQA are computed, window by window.

    python tools/benchmark_engine.py                         # CPU: 1 core and all physical cores
    python tools/benchmark_engine.py --jobs 1 8 20 --out bench.csv
    python tools/benchmark_engine.py --engines numpy torch --devices cuda:0
    python tools/benchmark_engine.py --quick                 # a smoke run, seconds
    python tools/benchmark_engine.py --engines torch --profile   # where the GPU time goes

Every number in the decision records about speed comes from this script, so it
measures what an analysis does: many windows of a three-dimensional
phase-amplitude trajectory, a threshold fixed per window at a 5% rate, the
default line floors, the full RQA. For each window size:

* **one window** on the reference engine, best of three: the time to estimate
  the threshold, and the time to build the plot and accumulate its line
  histograms;
* **batch_rqa** on every engine asked for: the numpy engine with each
  ``n_jobs`` (worker processes), the torch engine on each device and in each
  precision, in windows per second;
* **windowed** (numpy engine only): ``batch_windowed_recurrence(...)``
  ``.metrics(rqa=True)`` over several records, the other way a study calls it.

Two GPUs are measured by running the script twice at once, one device each
(``--devices cuda:0`` and ``--devices cuda:1``), as a study would run them.

The trajectories are synthetic, with fixed seeds. Each result is appended to
the CSV file as soon as it is measured (with the versions and the machine),
and the printed lines go to a ``.txt`` file next to it, so an interrupted run
keeps what it measured.

``--profile`` also times each stage of the torch engine (thresholds, the
decisions, each kind of line, copying back), waiting for the device at every
stage boundary, which slows it a little; the seconds go to the ``stages``
column and to the printed lines.

Thread counts: set ``OPENBLAS_NUM_THREADS=1`` (and ``MKL_NUM_THREADS=1`` with
MKL) before running, or the worker processes fight over the cores.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import pathlib
import platform
import sys
import time
import warnings

import numpy as np

import recurra as rc

FIELDS = ["recurra", "engine", "device", "precision", "measurement", "n_points",
          "dim", "windows", "n_jobs", "seconds", "windows_per_s", "threshold_s",
          "plot_and_histogram_s", "peak_device_memory_gb", "batch_size", "stages", "python",
          "numpy", "torch", "cpu_count",
          "physical_cores", "blas_threads", "platform"]


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
        rc.rqa(rc.recurrence_plot(X, th, store="stream"))

    return t_threshold, best_of(plot_and_histogram)


def windows_of(n: int, count: int) -> np.ndarray:
    return trajectory(n * count, seed=100).reshape(count, n, 3).copy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sizes", type=int, nargs="+", default=[1280, 2560, 5120, 10240],
                   help="points per window")
    p.add_argument("--engines", nargs="+", default=["numpy"], choices=["numpy", "torch"])
    p.add_argument("--jobs", type=int, nargs="+", default=None,
                   help="n_jobs for the numpy engine (default: 1 and all physical cores)")
    p.add_argument("--devices", nargs="+", default=["cuda:0"], help="torch devices")
    p.add_argument("--precisions", nargs="+", default=["double", "mixed"],
                   choices=["double", "single", "mixed"], help="torch precisions")
    p.add_argument("--windows", type=int, default=None,
                   help="windows per batch_rqa run (default: 10 x the largest n_jobs, "
                        "at least 20, and 200 on a GPU)")
    p.add_argument("--no-windowed", action="store_true",
                   help="skip the batch_windowed_recurrence measurement")
    p.add_argument("--out", default="benchmark_engine.csv", help="CSV file to write")
    p.add_argument("--profile", action="store_true",
                   help="time each stage of the torch engine (a little slower)")
    p.add_argument("--quick", action="store_true",
                   help="two small sizes and one job, to check the script runs")
    a = p.parse_args(argv)

    warnings.simplefilter("ignore", rc.RecurraWarning)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")       # for worker processes
    from recurra.engine import physical_cores

    cores = os.cpu_count() or 1
    physical = physical_cores()
    if a.quick:
        a.sizes, a.jobs = [300, 600], [1]
    jobs = a.jobs or sorted({1, physical})
    if max(jobs) > 1 and importlib.util.find_spec("joblib") is None:
        p.error('n_jobs > 1 needs joblib: pip install "recurra[parallel]"')
    if "torch" in a.engines and importlib.util.find_spec("torch") is None:
        p.error('the torch engine needs PyTorch: pip install "recurra[gpu]"')
    torch_version = ""
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch_version = torch.__version__
    if a.profile:
        import recurra.engine.torch_rqa as torch_rqa

        torch_rqa.PROFILE = True

    out = pathlib.Path(a.out)
    txt = out.with_suffix(".txt")
    fh = out.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader()
    tx = txt.open("w", encoding="utf-8")
    common = {"recurra": rc.__version__, "dim": 3, "python": platform.python_version(),
              "numpy": np.__version__, "torch": torch_version, "cpu_count": cores,
              "physical_cores": physical, "blas_threads": blas_threads(),
              "platform": platform.platform()}

    def say(text: str) -> None:
        print(text, flush=True)
        tx.write(text + "\n")
        tx.flush()

    def record(**row) -> None:
        writer.writerow({**common, **row})
        fh.flush()

    say(f"recurra {rc.__version__}, numpy {np.__version__}, torch {torch_version or '-'}, "
        f"{cores} logical cores ({physical} physical), BLAS threads {common['blas_threads']}")
    for n in a.sizes:
        t_th, t_plot = one_window(n)
        record(engine="numpy", device="cpu", precision="double",
               measurement="one_window", n_points=n, windows=1, n_jobs=1,
               seconds=t_th + t_plot, windows_per_s=1.0 / (t_th + t_plot),
               threshold_s=t_th, plot_and_histogram_s=t_plot)
        say(f"N={n:6d}  one window: threshold {t_th:.4f} s, plot + histograms {t_plot:.4f} s")

        if "numpy" in a.engines:
            for k in jobs:
                count = a.windows or (8 if a.quick else max(20, 10 * max(jobs)))
                X = windows_of(n, count)
                t0 = time.perf_counter()
                rc.batch_rqa(X, rng=0, engine="numpy", n_jobs=k)
                s = time.perf_counter() - t0
                record(engine="numpy", device="cpu", precision="double",
                       measurement="batch_rqa", n_points=n, windows=count, n_jobs=k,
                       seconds=s, windows_per_s=count / s)
                say(f"N={n:6d}  batch_rqa numpy, n_jobs={k:3d}: {count} windows in "
                    f"{s:8.2f} s, {count / s:9.2f} windows/s")
        if "torch" in a.engines:
            for dev in a.devices:
                for prec in a.precisions:
                    count = a.windows or (8 if a.quick else 200)
                    X = windows_of(n, count)
                    rc.batch_rqa(X[:2], rng=0, engine="torch", device=dev,
                                 precision=prec)                      # warm up
                    t0 = time.perf_counter()
                    table = rc.batch_rqa(X, rng=0, engine="torch", device=dev,
                                         precision=prec)
                    s = time.perf_counter() - t0
                    prov = table.attrs["provenance"]
                    peak = prov.get("peak_device_memory_gb")
                    batch = prov.get("batch_size", {}).get(n)
                    st = prov.get("stage_seconds")
                    record(engine="torch", device=dev, precision=prec,
                           measurement="batch_rqa", n_points=n, windows=count, n_jobs=1,
                           seconds=s, windows_per_s=count / s,
                           peak_device_memory_gb=peak, batch_size=batch,
                           stages=";".join(f"{k}={v}" for k, v in st.items()) if st else "")
                    say(f"N={n:6d}  batch_rqa torch on {dev}, {prec}: {count} windows in "
                        f"{s:8.2f} s, {count / s:9.2f} windows/s "
                        f"(batches of {batch}, peak {peak} GB)")
                    if st:
                        total = sum(st.values()) or 1.0
                        say("          stages: " + ", ".join(
                            f"{k} {v:.2f} s ({100 * v / total:.0f}%)" for k, v in st.items()))
        if "numpy" in a.engines and not a.no_windowed:
            for k in jobs:
                records = max(k, 2)
                per = 4 if a.quick else 10
                sources = {f"r{r:02d}": trajectory(n * per, seed=200 + r)
                           for r in range(records)}
                spec = rc.WindowSpec(size=n, step=n, unit="samples")
                t0 = time.perf_counter()
                batch = rc.batch_windowed_recurrence(sources, spec, target_rr=0.05,
                                                     rng=0, n_jobs=k)
                batch.metrics(rqa=True, n_jobs=k, engine="numpy")
                s = time.perf_counter() - t0
                total = records * per
                record(engine="numpy", device="cpu", precision="double",
                       measurement="windowed", n_points=n, windows=total, n_jobs=k,
                       seconds=s, windows_per_s=total / s)
                say(f"N={n:6d}  windowed numpy, n_jobs={k:3d}: {total} windows in "
                    f"{s:8.2f} s, {total / s:9.2f} windows/s")
    say(f"written: {out} and {txt}")
    fh.close()
    tx.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
