"""One rule for reading a raw array as a point cloud.

Design note (DD-123) -- a 2-D array is always (n_points, n_dimensions).

Before 0.26 each function guessed. ``recurrence_plot`` refused an array with
more columns than rows; ``estimate_tau``, ``estimate_m``, ``custom`` and the
dynamical measures transposed it without a word; ``threshold`` and the scaling
helpers read a 1-D vector as one point in N dimensions. Transposing a wide
array is right when the user passed channels as rows and silently wrong when
the array really is wide: a meta-series with 13 measurements over 12 windows
became 13 points in 12 dimensions.

Now every function that accepts a raw array reads it the same way:

* 1-D -> one column, ``(n, 1)``: a scalar series;
* 2-D -> as given, rows are points;
* anything else is an error.

A 2-D array with more columns than rows is still read as given, because that
can be right, and the call warns once, with a fixed text, so that a loop over
windows does not repeat it. A ``StateSpace`` never goes through here: it knows
its own orientation.
"""
from __future__ import annotations

import contextlib
import contextvars
import warnings

import numpy as np

from ..exceptions import GeometryWarning, ParameterError

# Set while a function passes on an array it has already read, so that the
# functions it calls do not warn about the same array a second time.
_ALREADY_READ = contextvars.ContextVar("recurra_points_already_read", default=False)


@contextlib.contextmanager
def already_read():
    """Within this block, ``as_points`` does not warn: the caller already did."""
    token = _ALREADY_READ.set(True)
    try:
        yield
    finally:
        _ALREADY_READ.reset(token)


def as_points(x, *, name: str = "input", warn: bool = True) -> np.ndarray:
    """Read a raw array as ``(n_points, n_dimensions)`` in float [DD-123].

    Parameters
    ----------
    x
        Array-like. A 1-D vector becomes one column; a 2-D array is taken as
        it is, rows as points.
    name
        The calling function, for the messages.
    warn
        Warn when a 2-D array has more columns than rows (it is still read as
        given). Off for internal calls that only need the number of points.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ParameterError(
            f"{name}: expected a 1-D series or a 2-D array (n_points, n_dimensions); "
            f"got {arr.ndim} dimensions with shape {arr.shape}")
    if warn and arr.shape[0] < arr.shape[1] and not _ALREADY_READ.get():
        warnings.warn(
            f"{name}: a 2-D array is read as (n_points, n_dimensions), and this one "
            "has more dimensions than points. It is used as given; if its rows are "
            "channels or measurements, pass its transpose [DD-123].",
            GeometryWarning, stacklevel=3)
    return arr


def n_points(x) -> int:
    """Number of points of a state space or of a raw array, by the same rule."""
    if hasattr(x, "n_points"):
        return int(x.n_points)
    arr = np.asarray(x)
    return int(arr.shape[0]) if arr.ndim >= 1 else 1
