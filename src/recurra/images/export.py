"""Recurrence plots as images, for classifiers that take pictures.

Design note (DD-90) -- resizing a recurrence plot is not resampling a photo.

A recurrence plot of 34 000 points holds 1.16 billion cells. Reducing it to
256 squared by nearest-neighbour resampling -- what an image library does by
default, and what the earlier draft did -- keeps **one cell in seventeen
thousand** and throws the rest away. Two recordings with quite different
structure can then produce near-identical pictures, decided by which cells
happened to land on the grid.

The reduction here **pools**: every output pixel is the mean of the cells it
covers, so every cell contributes and the value is the local recurrence
density. The picture is a density map in [0, 1], not a binary plot, and that
is the more informative input for a network anyway: a convolution over
densities sees gradients that a thresholded image has already destroyed.

Maximum pooling remains available and preserves lines rather than density, but
it saturates fast -- at a reduction factor of 20 and a 5% rate almost every
pixel lights up [DD-38]. ``pooling="auto"`` picks between them and says which
it chose.

Design note (DD-91) -- the brightness scale is shared, or the difference is
gone.

At a 5% recurrence rate the pooled values sit around 0.05, so mapping them
straight onto 0-255 gives a nearly black image. The obvious fix is to divide
each image by its own maximum -- and that is the one thing that must not be
done. Every recording is pinned to the same target rate by construction, so
what distinguishes two subjects is *how the density is distributed*, and
per-image normalisation rescales exactly that away.

``vmax`` is therefore a property of the batch, not of the image. The default
computes one value across every plot exported together, records it in the
metadata, and applies it to all of them. A fixed number can be passed instead
when several batches must be comparable.

**The floor matters as much as the ceiling.** At a reduction factor of 133
each pixel averages seventeen thousand cells, so by the law of large numbers
every density lands near the global rate: measured, a batch spanned 0.050 to
0.105 around a mean of 0.0496. Mapping that onto 0-255 from zero puts the whole
image between grey 121 and 255 and the structure is invisible. ``vmin`` is
taken from a low batch quantile for the same reason ``vmax`` is taken from a
high one, and both are recorded.

Design note (DD-92) -- past some reduction, pool less and decimate instead.

Pooling a 34 000-point plot to 256 pixels costs a full pass over 1.16 billion
cells -- 98 seconds measured -- and the result is nearly uniform because each
pixel averages so many cells. Building the plot from a decimated trajectory of
about a thousand points costs a second and every pixel is a real recurrence
decision.

Neither is wrong and they answer different questions. ``decimate`` exposes the
choice, the metadata records which was used, and images made the two ways are
not comparable with each other.
"""
from __future__ import annotations

import pathlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..exceptions import ParameterError
from ..viz.recurrence import render_recurrence

FORMATS = ("png", "npy")
POOLINGS = ("mean", "max", "auto")


def recurrence_image(source, *, size: int = 256, pooling: str = "mean",
                     **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
    """A square density map of a recurrence structure, plus how it was made.

    Returns ``(image, info)``. The image is float in [0, 1]: each pixel is the
    fraction of recurrent cells in the block it covers [DD-90]. It is **not**
    scaled for display; :func:`export_recurrence_images` does that once for a
    whole batch [DD-91].
    """
    if pooling not in POOLINGS:
        raise ParameterError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
    from ..recurrence.core import RecurrenceMatrix

    if not isinstance(source, RecurrenceMatrix):
        from ..recurrence.builders import recurrence_plot

        source = recurrence_plot(source, **kwargs)

    reduced, factor, method = render_recurrence(source, max_size=size,
                                                pooling=pooling)
    image = np.asarray(reduced, dtype=float)
    peak = float(image.max())
    if peak > 1.0:                              # never expected; guard anyway
        image = image / peak

    # A square is what a network expects; a cross plot need not be square.
    if image.shape[0] != image.shape[1]:
        n = min(image.shape)
        image = image[:n, :n]

    # Pooling lands near `size` but not exactly on it -- 20 000 points reduce
    # to 254, 34 000 to 256 -- and a network needs every image the same shape.
    # The last step to `size` is an interpolation over a factor below 1.05,
    # which is nothing like the seventeen-thousandfold discard this module
    # exists to avoid, and it is applied to the already-pooled array [DD-90].
    resized = False
    if image.shape[0] != size:
        from scipy.ndimage import zoom

        image = zoom(image, size / image.shape[0], order=1)
        image = image[:size, :size]
        if image.shape[0] < size:               # never expected; be certain
            image = np.pad(image, ((0, size - image.shape[0]),
                                   (0, size - image.shape[1])), mode="edge")
        resized = True

    info = {"n_points": source.shape[0], "image_size": int(image.shape[0]),
            "resized_to_fit": resized,
            "pooling": method, "reduction_factor": int(factor),
            "cells_per_pixel": int(factor) ** 2,
            "recurrence_rate": float(source.recurrence_rate()),
            "epsilon": float(source.threshold.scalar),
            "density_mean": float(image.mean()),
            "density_max": float(image.max())}
    return image, info


def _write(image: np.ndarray, path: pathlib.Path, fmt: str, vmax: float,
           vmin: float = 0.0) -> None:
    if fmt == "npy":
        # The raw density, unscaled: a network can normalise it as it likes,
        # and the scaling used for the picture is in the metadata anyway.
        np.save(path, image.astype(np.float32))
        return
    span = (vmax - vmin) or 1.0
    scaled = np.clip((image - vmin) / span, 0.0, 1.0)
    try:
        import matplotlib.image as mpimg

        mpimg.imsave(str(path), scaled, cmap="gray", vmin=0.0, vmax=1.0,
                     format="png")
    except Exception as exc:  # pragma: no cover - depends on Pillow
        raise ParameterError(
            f"could not write {path}: {exc}. PNG writing needs a working "
            "Pillow; use format='npy' to avoid it entirely."
        ) from exc


def export_recurrence_images(sources: Mapping[str, Any], output_dir, *,
                             size: int = 256, pooling: str = "mean",
                             decimate: int | str = 1,
                             labels: Mapping[str, Any] | None = None,
                             fmt: str = "png", vmax: float | str = "auto",
                             vmin: float | str = "auto",
                             quantiles: tuple[float, float] = (0.01, 0.999),
                             metadata_name: str = "images.csv",
                             extra: Mapping[str, Mapping[str, Any]] | None = None,
                             **kwargs) -> pd.DataFrame:
    """Write one image per recurrence structure, and a table describing them.

    Parameters
    ----------
    sources
        Mapping from a name to a :class:`RecurrenceMatrix`, a state space, or
        anything a recurrence plot can be built from.
    labels
        Optional mapping from the same names to a class, written into the
        metadata so a loader can find it without a second file.
    decimate
        Reduce the trajectory before building the plot. ``"match"`` decimates
        so the plot is about ``size`` points and every pixel is one recurrence
        decision -- a second per image instead of a minute, and far more
        contrast [DD-92]. ``1`` pools the full-rate plot instead.
    vmax, vmin
        Brightness ceiling and floor. ``"auto"`` takes quantiles across the
        whole batch and applies them to every image, which is what keeps them
        comparable [DD-91]. Numbers fix them across batches. ``vmax="per_image"``
        normalises each on its own and is available only because someone will
        ask; it rescales away the differences the images exist to show.

    Returns
    -------
    A metadata frame with one row per image: the file, the class, the pooling
    that was used, how many cells each pixel covers, and the brightness ceiling
    that was applied.
    """
    if fmt not in FORMATS:
        raise ParameterError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images, infos = {}, {}
    for name, source in sources.items():
        item = source
        factor = 1
        if decimate != 1 and hasattr(source, "subsample"):
            # Ceiling, not floor: 34000 // 256 leaves 258 points, which the
            # renderer then pools by two into a 129-pixel image instead of the
            # 256 that was asked for.
            # The stride has to leave at least `size` points, so the factor is
            # floor(n/size): a ceiling overshoots and the image comes out one
            # or two pixels short of what was asked for.
            factor = (max(1, source.n_points // size) if decimate == "match"
                      else int(decimate))
            item = source.subsample(factor) if factor > 1 else source
        images[name], infos[name] = recurrence_image(item, size=size,
                                                     pooling=pooling, **kwargs)
        infos[name]["trajectory_decimation"] = factor

    pool = np.concatenate([im.ravel() for im in images.values()])
    if vmax == "auto":
        ceiling = float(np.quantile(pool, quantiles[1])) or 1.0
        vmax_kind = f"batch quantiles {quantiles}"
    elif vmax == "per_image":
        ceiling, vmax_kind = None, "per image (not comparable)"
    else:
        ceiling, vmax_kind = float(vmax), "fixed"
    floor = (float(np.quantile(pool, quantiles[0])) if vmin == "auto"
             else (0.0 if vmin == "per_image" else float(vmin)))

    rows = []
    for name, image in images.items():
        applied = ceiling if ceiling is not None else float(image.max()) or 1.0
        low = floor if ceiling is not None else float(image.min())
        path = out / f"{name}.{fmt}"
        _write(image, path, fmt, applied, low)
        row = {"name": name, "file": path.name, "path": str(path),
               "label": (labels or {}).get(name), "vmin": low, "vmax": applied,
               "vmax_kind": vmax_kind, **infos[name]}
        row.update((extra or {}).get(name, {}))
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / metadata_name, index=False)
    return frame


def attractor_image(space, *, size: int = 256, coords: tuple[int, int] = (0, 1),
                    density: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    """A square occupancy map of the trajectory, projected onto two coordinates.

    Design note (DD-103) -- a scatter plot is not an image a network can read.

    The obvious way to picture an attractor is to draw the points. At 34 000
    points on a 256-pixel grid most pixels are hit many times and the picture
    saturates into a silhouette: the shape survives and how often the
    trajectory visits each part does not. That is the same failure as
    thresholding a recurrence plot [DD-90], and the same answer works --
    **count**, do not mark.

    Each pixel holds the number of trajectory points falling in it, normalised
    by the largest count, so the value is local occupancy. ``density=False``
    gives the binary silhouette instead, which is what a scatter plot would
    show.

    The projection is stated: a PAC space is three-dimensional and any picture
    of it is two of those coordinates. The default takes the first two, which
    for a phase circle is the circle itself; ``coords=(0, 2)`` crosses one
    phase coordinate with the amplitude.
    """
    X = np.asarray(space.weighted_coords if hasattr(space, "weighted_coords")
                   else space, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ParameterError("an attractor image needs at least two coordinates")
    i, j = coords
    if max(i, j) >= X.shape[1]:
        raise ParameterError(
            f"coords {coords} out of range for a {X.shape[1]}-dimensional space")

    x, y = X[:, i], X[:, j]
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        raise ParameterError("no finite points to plot")

    # Robust limits: one outlying burst would otherwise squeeze the trajectory
    # into a handful of pixels [DD-50].
    lo_x, hi_x = np.quantile(x, [0.001, 0.999])
    lo_y, hi_y = np.quantile(y, [0.001, 0.999])
    if hi_x <= lo_x:
        hi_x = lo_x + 1e-12
    if hi_y <= lo_y:
        hi_y = lo_y + 1e-12

    counts, _, _ = np.histogram2d(x, y, bins=size,
                                  range=[[lo_x, hi_x], [lo_y, hi_y]])
    image = counts.T[::-1]                      # rows top-down, as a picture
    peak = float(image.max()) or 1.0
    occupied = float((image > 0).mean())
    if density:
        image = image / peak
    else:
        image = (image > 0).astype(float)

    info = {"n_points": int(x.size), "image_size": int(size),
            "coords": f"{i},{j}", "kind": "density" if density else "silhouette",
            "peak_count": peak, "occupied_fraction": occupied,
            "clipped_fraction": float(1.0 - finite.mean())}
    if hasattr(space, "groups"):
        labels = [g.label for g in space.groups]
        info["blocks"] = "|".join(labels)
    return image, info


def export_attractor_images(spaces: Mapping[str, Any], output_dir, *,
                            size: int = 256, coords: tuple[int, int] = (0, 1),
                            density: bool = True,
                            labels: Mapping[str, Any] | None = None,
                            fmt: str = "png", vmax: float | str = "auto",
                            vmin: float | str = "auto",
                            quantiles: tuple[float, float] = (0.0, 0.999),
                            metadata_name: str = "attractors.csv",
                            extra: Mapping[str, Mapping[str, Any]] | None = None
                            ) -> pd.DataFrame:
    """Write one attractor image per state space, and a table describing them.

    The brightness scale is shared across the batch for the same reason as for
    recurrence plots [DD-91]: per-image normalisation rescales away the
    differences between subjects, which is what the images exist to show.
    """
    if fmt not in FORMATS:
        raise ParameterError(f"fmt must be one of {FORMATS}, got {fmt!r}")
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images, infos = {}, {}
    for name, space in spaces.items():
        images[name], infos[name] = attractor_image(space, size=size,
                                                    coords=coords,
                                                    density=density)
    pool = np.concatenate([im.ravel() for im in images.values()])
    ceiling = (float(np.quantile(pool, quantiles[1])) or 1.0 if vmax == "auto"
               else (None if vmax == "per_image" else float(vmax)))
    floor = (float(np.quantile(pool, quantiles[0])) if vmin == "auto"
             else (0.0 if vmin == "per_image" else float(vmin)))
    kind = ("batch quantiles" if vmax == "auto"
            else "per image (not comparable)" if vmax == "per_image" else "fixed")

    rows = []
    for name, image in images.items():
        applied = ceiling if ceiling is not None else float(image.max()) or 1.0
        low = floor if ceiling is not None else float(image.min())
        path = out / f"{name}.{fmt}"
        _write(image, path, fmt, applied, low)
        row = {"name": name, "file": path.name, "path": str(path),
               "label": (labels or {}).get(name), "vmin": low, "vmax": applied,
               "vmax_kind": kind, **infos[name]}
        row.update((extra or {}).get(name, {}))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / metadata_name, index=False)
    return frame
