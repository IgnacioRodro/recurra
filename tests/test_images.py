"""Tests for recurrence plots exported as images."""
import numpy as np
import pandas as pd
import pytest

import recurra as rc
from recurra import statespace as sm


def _space(n=6000, alpha=0.7, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 250.0
    phase = 2 * np.pi * 6 * t + 0.3 * np.cumsum(rng.standard_normal(n)) / 50
    envelope = 1 + alpha * np.cos(phase - np.pi / 2) + 0.3 * rng.standard_normal(n)
    rec = rc.ingest({"phase_theta": np.angle(np.exp(1j * phase)),
                     "amp_gamma": np.abs(envelope)}, fs=250.0,
                    roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
    return sm.pac_space(rec, smooth=8.0)


# --------------------------------------------------------------- the image
def test_image_is_a_density_map_in_the_unit_interval():
    """DD-90: every output pixel is the mean of the cells it covers, so the
    value is a local recurrence density, not a thresholded pixel."""
    image, info = rc.recurrence_image(_space(), size=128, pooling="mean",
                                      target_rr=0.05, theiler=1, rng=0)
    assert image.shape == (128, 128)
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert info["pooling"] == "mean"
    # The mean of the map is the recurrence rate, because pooling preserves it.
    assert image.mean() == pytest.approx(info["recurrence_rate"], abs=0.01)


def test_pooling_preserves_density_where_nearest_neighbour_would_not():
    """The argument for pooling, made as a measurement: a plot reduced by a
    large factor keeps its rate under mean pooling, and would keep only one
    cell in factor-squared under resampling."""
    space = _space(n=8000)
    image, info = rc.recurrence_image(space, size=100, pooling="mean",
                                      target_rr=0.05, theiler=1, rng=0)
    assert info["cells_per_pixel"] > 1000
    assert image.mean() == pytest.approx(0.05, abs=0.01)


@pytest.mark.requires("matplotlib")
def test_every_image_has_the_size_that_was_asked_for():
    """Recordings of different length must still give one shape, or a network
    cannot take them [DD-90]."""
    sources = {f"s{i}": _space(n=n, seed=i)
               for i, n in enumerate((5000, 6500, 8000))}
    frame = rc.export_recurrence_images(sources, "/tmp/rc_test_size", size=64,
                                        decimate="match", target_rr=0.05,
                                        theiler=1, rng=0)
    assert frame["image_size"].nunique() == 1
    assert frame["image_size"].iloc[0] == 64


def test_unknown_pooling_is_rejected():
    with pytest.raises(rc.ParameterError, match="pooling must be"):
        rc.recurrence_image(_space(), pooling="bicubic")


# ------------------------------------------------------- the shared scale
@pytest.mark.requires("matplotlib")
def test_the_brightness_scale_is_shared_across_the_batch(tmp_path):
    """DD-91: per-image normalisation rescales away exactly the differences the
    images exist to show, because every recording is pinned to the same rate."""
    sources = {"weak": _space(alpha=0.1, seed=1), "strong": _space(alpha=0.9, seed=2)}
    frame = rc.export_recurrence_images(sources, tmp_path, size=64,
                                        target_rr=0.05, theiler=1, rng=0)
    assert frame["vmax"].nunique() == 1
    assert frame["vmin"].nunique() == 1
    assert "batch quantile" in frame["vmax_kind"].iloc[0]


@pytest.mark.requires("matplotlib")
def test_per_image_normalisation_is_available_and_marked(tmp_path):
    sources = {"a": _space(seed=3), "b": _space(alpha=0.2, seed=4)}
    frame = rc.export_recurrence_images(sources, tmp_path, size=64,
                                        vmax="per_image", target_rr=0.05,
                                        theiler=1, rng=0)
    assert frame["vmax"].nunique() == 2
    assert "not comparable" in frame["vmax_kind"].iloc[0]


@pytest.mark.requires("matplotlib")
def test_a_fixed_scale_can_be_shared_between_batches(tmp_path):
    sources = {"a": _space(seed=5)}
    frame = rc.export_recurrence_images(sources, tmp_path, size=64, vmax=0.2,
                                        vmin=0.0, target_rr=0.05, theiler=1, rng=0)
    assert frame["vmax"].iloc[0] == 0.2
    assert frame["vmax_kind"].iloc[0] == "fixed"


# --------------------------------------------------------------- metadata
@pytest.mark.requires("matplotlib")
def test_metadata_describes_how_each_image_was_made(tmp_path):
    sources = {"a": _space(seed=6), "b": _space(seed=7)}
    frame = rc.export_recurrence_images(sources, tmp_path, size=64,
                                        labels={"a": "control", "b": "study"},
                                        target_rr=0.05, theiler=1, rng=0)
    for column in ("name", "file", "label", "pooling", "cells_per_pixel",
                   "recurrence_rate", "epsilon", "vmin", "vmax", "image_size"):
        assert column in frame.columns
    assert list(frame["label"]) == ["control", "study"]
    assert (tmp_path / "images.csv").is_file()
    assert len(pd.read_csv(tmp_path / "images.csv")) == 2


def test_npy_output_needs_no_image_library(tmp_path):
    """PNG writing depends on Pillow, which is not always healthy. The raw
    density can always be written."""
    frame = rc.export_recurrence_images({"a": _space(seed=8)}, tmp_path, size=32,
                                        fmt="npy", target_rr=0.05, theiler=1, rng=0)
    saved = np.load(tmp_path / "a.npy")
    assert saved.shape == (32, 32)
    assert saved.dtype == np.float32
    assert frame["file"].iloc[0].endswith(".npy")


def test_unknown_format_is_rejected(tmp_path):
    with pytest.raises(rc.ParameterError, match="fmt must be"):
        rc.export_recurrence_images({"a": _space()}, tmp_path, fmt="jpeg2000")


# ---------------------------------------------- decimate rather than pool
@pytest.mark.requires("matplotlib")
def test_decimating_records_what_it_did(tmp_path):
    """DD-92: the two routes answer different questions and images made one way
    are not comparable with the other, so which was used is recorded."""
    source = _space(n=8000, seed=9)
    pooled = rc.export_recurrence_images({"a": source}, tmp_path / "p", size=64,
                                         decimate=1, target_rr=0.05, theiler=1,
                                         rng=0)
    decimated = rc.export_recurrence_images({"a": source}, tmp_path / "d", size=64,
                                            decimate="match", target_rr=0.05,
                                            theiler=1, rng=0)
    assert pooled["trajectory_decimation"].iloc[0] == 1
    assert decimated["trajectory_decimation"].iloc[0] > 1
    assert pooled["cells_per_pixel"].iloc[0] > decimated["cells_per_pixel"].iloc[0]


# ------------------------------------------- attractor images [DD-103]
def test_an_attractor_image_counts_rather_than_marks():
    """DD-103. Drawing 34 000 points on a 256-pixel grid saturates into a
    silhouette: the shape survives and how often the trajectory visits each
    part does not. The same failure as thresholding a recurrence plot, and the
    same answer -- count, do not mark."""
    space = _space(n=20000)
    density, info = rc.attractor_image(space, size=64, density=True)
    silhouette, _ = rc.attractor_image(space, size=64, density=False)
    assert density.shape == (64, 64)
    assert info["kind"] == "density"
    assert set(np.unique(silhouette)) <= {0.0, 1.0}
    # the density carries information the silhouette has thrown away
    assert len(np.unique(density)) > 10
    assert info["peak_count"] > 1


def test_the_projection_is_recorded():
    """A picture of a three-dimensional space is two of its coordinates, and
    which two is not a detail [DD-103]."""
    space = _space(n=8000)
    _, a = rc.attractor_image(space, size=32, coords=(0, 1))
    _, b = rc.attractor_image(space, size=32, coords=(0, 2))
    assert a["coords"] == "0,1"
    assert b["coords"] == "0,2"
    assert a["blocks"] == "phase_theta|amp_gamma"


def test_an_impossible_projection_is_refused():
    space = _space(n=4000)
    with pytest.raises(rc.ParameterError, match="out of range"):
        rc.attractor_image(space, coords=(0, 9))


@pytest.mark.requires("matplotlib")
def test_attractor_export_shares_its_scale(tmp_path):
    """DD-91 again: per-image normalisation would rescale away the differences
    between subjects."""
    spaces = {"weak": _space(alpha=0.1, seed=1), "strong": _space(alpha=0.9, seed=2)}
    frame = rc.export_attractor_images(spaces, tmp_path, size=64,
                                       labels={"weak": "a", "strong": "b"})
    assert frame["vmax"].nunique() == 1
    assert list(frame["label"]) == ["a", "b"]
    assert (tmp_path / "attractors.csv").is_file()
    assert frame["image_size"].nunique() == 1


def test_an_outlying_burst_does_not_squeeze_the_trajectory(tmp_path):
    """DD-50: robust limits, or one burst puts everything in a few pixels."""
    space = _space(n=8000)
    coords = np.asarray(space.coords).copy()
    coords[100, 2] += 500.0                      # one enormous excursion
    from recurra.statespace.core import StateSpace

    spiked = StateSpace(coords=coords, groups=space.groups, fs=space.fs)
    image, info = rc.attractor_image(spiked, size=64, coords=(0, 2))
    assert info["occupied_fraction"] > 0.02, (
        "the trajectory collapsed into a corner; the limits are not robust")
