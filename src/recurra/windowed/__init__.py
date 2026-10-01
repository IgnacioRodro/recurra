"""Windowed analysis: many recurrence structures from one record."""
from .meta import meta_recurrence, meta_space, meta_theiler
from .recurrence import (
    BatchWindowedRecurrence,
    WindowedRecurrence,
    batch_windowed_recurrence,
    windowed_cross_recurrence,
    windowed_joint_recurrence,
    windowed_recurrence,
)
from .spec import Window, WindowSpec, resolve_windows

__all__ = [
    "WindowSpec", "Window", "resolve_windows",
    "WindowedRecurrence", "windowed_recurrence",
    "windowed_cross_recurrence", "windowed_joint_recurrence",
    "BatchWindowedRecurrence", "batch_windowed_recurrence",
    "meta_space", "meta_recurrence", "meta_theiler",
]
