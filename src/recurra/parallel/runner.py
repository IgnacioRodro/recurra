"""Running independent work in parallel, without changing the answer.

Design note (DD-81) -- parallelism is a scheduling decision, never a numerical
one.

[DD-05] promised that results do not depend on ``n_jobs``. That promise is easy
to make and easy to break: the usual way to break it is to draw random numbers
from a shared generator, so the values a worker receives depend on the order
the scheduler happened to run things in.

Here every unit of work is given its own random stream, derived from the
caller's seed by ``SeedSequence.spawn`` before any work starts. The streams
depend on the *position* of the item in the input, never on when it runs.
Results are reassembled in input order. A test asserts equality between serial
and parallel runs rather than trusting the argument.

Design note (DD-82) -- parallelise across units, not inside one.

A single recurrence plot cannot usefully be split: the line-length accumulator
carries open runs from tile to tile, and that carry is sequential by
construction [DD-56]. Producing tiles in parallel and consuming them serially
is possible but the consumer becomes the bottleneck.

Across *units* -- subjects, windows, parameter cells -- the work is completely
independent, and that is where a corpus spends its time. Forty subjects on
eight cores is close to eight times faster in wall clock; one subject on eight
cores is not.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from ..config import SeedLike, get_config, spawn_rngs
from ..exceptions import BackendUnavailable, ParameterError

BACKENDS = ("serial", "thread", "process", "auto")


def resolve_jobs(n_jobs: int | None = None) -> int:
    """How many workers to use. ``-1`` means every core, ``None`` the config."""
    n = get_config().n_jobs if n_jobs is None else n_jobs
    if n is None or n == 0:
        return 1
    if n < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n)
    return int(n)


def _require_joblib():
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise BackendUnavailable("joblib", "parallel") from exc
    return joblib


def parallel_map(func: Callable[..., Any], items: Sequence[Any], *,
                 n_jobs: int | None = None, backend: str = "auto",
                 rng: SeedLike = None, pass_rng: bool = True,
                 progress: bool = False, **kwargs) -> list:
    """Apply ``func`` to every item, possibly in parallel, in input order.

    Each item receives its own random stream when ``pass_rng`` is true, derived
    from ``rng`` by position before any work starts, so the result cannot
    depend on ``n_jobs`` [DD-81].

    ``backend="process"`` gives true parallelism for numpy-bound work at the
    cost of pickling the arguments; ``"thread"`` avoids that but only helps
    where the work releases the interpreter lock, which the BLAS-backed
    distance kernels do. ``"auto"`` picks threads, which is the safer default
    here because a state space can be large to pickle.
    """
    if backend not in BACKENDS:
        raise ParameterError(f"backend must be one of {BACKENDS}, got {backend!r}")
    items = list(items)
    streams = spawn_rngs(rng, max(1, len(items))) if pass_rng else [None] * len(items)

    n = resolve_jobs(n_jobs)
    if n <= 1 or len(items) <= 1 or backend == "serial":
        out = []
        for i, (item, stream) in enumerate(zip(items, streams, strict=False)):
            if progress:
                print(f"  [{i + 1}/{len(items)}]", flush=True)
            out.append(func(item, **({"rng": stream} if pass_rng else {}), **kwargs))
        return out

    joblib = _require_joblib()
    prefer = "threads" if backend in ("auto", "thread") else "processes"
    with joblib.parallel_backend("loky" if prefer == "processes" else "threading",
                                 n_jobs=n):
        return list(joblib.Parallel(n_jobs=n, verbose=10 if progress else 0)(
            joblib.delayed(func)(item, **({"rng": stream} if pass_rng else {}),
                                 **kwargs)
            for item, stream in zip(items, streams, strict=False)))
