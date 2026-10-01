"""Running independent work in parallel without changing the answer."""
from .runner import BACKENDS, parallel_map, resolve_jobs

__all__ = ["parallel_map", "resolve_jobs", "BACKENDS"]
