"""Global configuration and the random-number policy.

Design note (DD-05). The old draft called ``np.random.seed`` inside library
functions, which mutates global interpreter state and silently correlates
draws between callers. Here every stochastic operation takes an explicit
``rng``; ``resolve_rng`` is the single funnel. Parallel workers derive
independent streams from a task index, so parallelism never changes results.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

SeedLike = int | np.random.Generator | np.random.SeedSequence | None


def resolve_rng(seed: SeedLike = None) -> np.random.Generator:
    """Return a Generator from anything a user might reasonably pass."""
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    return np.random.default_rng(seed)


def spawn_rngs(seed: SeedLike, n: int) -> list[np.random.Generator]:
    """Derive ``n`` independent, reproducible streams for parallel workers.

    Using SeedSequence.spawn (rather than seed+i) guarantees statistical
    independence between streams, so results do not depend on ``n_jobs``.
    """
    if isinstance(seed, np.random.Generator):
        # Public as `seed_seq` from numpy 1.25; `_seed_seq` before. The declared
        # floor is 1.24, and the public name alone broke every parallel path
        # there -- found by running the suite at the floors, which had never
        # been done.
        bg = seed.bit_generator
        base = getattr(bg, "seed_seq", None) or bg._seed_seq  # type: ignore[attr-defined]
    elif isinstance(seed, np.random.SeedSequence):
        base = seed
    else:
        base = np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in base.spawn(n)]


@dataclass
class Config:
    """Process-wide defaults. Mutate through :func:`set_config`."""

    #: Memory the library may use for a single computation, in GiB.
    memory_budget_gb: float = 4.0
    #: Default dtype for state-space coordinates and distances.
    float_dtype: str = "float64"
    #: Emit warnings when a stage makes a consequential silent choice.
    warn_on_caveat: bool = True
    #: Where CSV exports go when no path is given.
    output_dir: str = "outputs"
    #: Default parallel backend.
    parallel_backend: str = "serial"
    n_jobs: int = 1
    #: Fail instead of warning when data quality checks trip.
    strict_quality: bool = False
    _extra: dict = field(default_factory=dict)


_CONFIG = Config()


def get_config() -> Config:
    return _CONFIG


def set_config(**kwargs) -> Config:
    """Update global defaults; returns the config for chaining."""
    for k, v in kwargs.items():
        if not hasattr(_CONFIG, k):
            raise AttributeError(f"unknown config key {k!r}")
        setattr(_CONFIG, k, v)
    return _CONFIG


def output_path(filename: str, subdir: str | None = None) -> str:
    base = _CONFIG.output_dir
    if subdir:
        base = os.path.join(base, subdir)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, filename)


def available_backends() -> dict[str, bool]:
    """What optional machinery is installed. Backs ``recurra doctor``."""
    def _has(mod: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(mod) is not None

    out = {
        "numpy": True,
        "scipy": True,
        "pandas": True,
        "scikit-learn": _has("sklearn"),
        "h5py": _has("h5py"),
        "pyedflib": _has("pyedflib"),
        "matplotlib": _has("matplotlib"),
        "pillow": _has("PIL"),
        "joblib": _has("joblib"),
        # Detected for the planned GPU backend [DD-16]; nothing uses it yet.
        "torch": _has("torch"),
    }
    if out["torch"]:
        try:
            import torch

            out["torch_cuda"] = bool(torch.cuda.is_available())
        except Exception:
            out["torch_cuda"] = False
    else:
        out["torch_cuda"] = False
    return out
