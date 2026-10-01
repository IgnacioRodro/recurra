#!/usr/bin/env python3
"""Run every example in order and report what each one produced.

    python examples/run_all.py                 # into ./outputs
    python examples/run_all.py --out results   # somewhere else
    python examples/run_all.py --skip 6        # leave out the cookbook

Each example writes tables, figures and, where applicable, a run report. This
script runs them in sequence, times them, and prints a summary of what landed
in the output directory.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SCRIPTS = sorted(HERE.glob("example_*.py"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--skip", type=int, nargs="*", default=[],
                    help="example numbers to leave out, e.g. --skip 6")
    ap.add_argument("--only", type=int, nargs="*", default=[],
                    help="only these example numbers")
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    chosen = []
    for path in SCRIPTS:
        number = int(path.stem.split("_")[1])
        if number in args.skip:
            continue
        if args.only and number not in args.only:
            continue
        chosen.append((number, path))

    print(f"output directory: {out}")
    print(f"running {len(chosen)} example(s)\n")

    results = []
    for number, path in chosen:
        print(f"[{number}] {path.name} ...", end="", flush=True)
        before = {f.name for f in out.glob("*")}
        started = time.time()
        # Every example honours RECURRA_EXAMPLES_OUT, so --out really decides
        # where the files land and the summary below counts the right ones.
        env = {**os.environ, "RECURRA_EXAMPLES_OUT": str(out)}
        proc = subprocess.run([sys.executable, str(path)], cwd=out.parent,
                              capture_output=True, text=True, env=env)
        elapsed = time.time() - started
        after = {f.name for f in out.glob("*")}
        new = after - before
        ok = proc.returncode == 0
        print(f" {'ok' if ok else 'FAILED'}  {elapsed:6.1f} s  "
              f"{len(new)} new file(s)")
        if not ok:
            print("  --- stderr ---")
            print("  " + "\n  ".join(proc.stderr.strip().splitlines()[-15:]))
        results.append((number, path.name, ok, elapsed, len(new)))

    files = sorted(out.glob("*"))
    figures = [f for f in files if f.suffix == ".png"]
    tables = [f for f in files if f.suffix == ".csv"
              and not f.stem.endswith(("_environment", "_provenance"))]
    sidecars = [f for f in files if f.stem.endswith(("_environment", "_provenance"))]
    reports = [f for f in files if f.suffix == ".txt"]

    print("\n" + "=" * 70)
    print(f"{'example':32s} {'result':>8s} {'seconds':>9s} {'new files':>10s}")
    print("-" * 70)
    for _number, name, ok, elapsed, n_new in results:
        print(f"{name:32s} {'ok' if ok else 'FAILED':>8s} {elapsed:9.1f} {n_new:10d}")
    print("-" * 70)
    print(f"figures      : {len(figures):4d}")
    print(f"tables       : {len(tables):4d}  (+{len(sidecars)} provenance sidecars)")
    print(f"run reports  : {len(reports):4d}   " + ", ".join(f.name for f in reports))
    print(f"total files  : {len(files):4d}   in {out}")
    print("=" * 70)

    return 0 if all(r[2] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
