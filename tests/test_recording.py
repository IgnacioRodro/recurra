import numpy as np
import pytest

import recurra as rc
from recurra.provenance import Step


def test_copy_on_write_shares_buffers(rec):
    out = rec.with_channels({"new": np.zeros(rec.n_samples)}, step=Step("t"))
    name = rec.names[0]
    assert out.get(name) is rec.get(name)          # shared, not copied
    assert rec.n_channels == 1 and out.n_channels == 2   # original untouched


def test_arrays_are_read_only(rec):
    with pytest.raises(ValueError):
        rec.get(rec.names[0])[0] = 123.0


def test_length_mismatch_rejected():
    with pytest.raises(rc.ParameterError):
        rc.Recording({"a": np.zeros(10), "b": np.zeros(11)}, fs=1.0)


def test_provenance_accumulates(rec):
    n0 = len(rec.provenance)
    out = rec.with_channels({"z": np.zeros(rec.n_samples)}, step=Step("s1"))
    out = out.with_channels({"w": np.zeros(rec.n_samples)}, step=Step("s2"))
    assert len(out.provenance) == n0 + 2
    assert list(out.provenance_frame()["step"])[-2:] == ["s1", "s2"]


def test_channel_frame_is_exportable(rec):
    df = rec.channel_frame()
    assert {"channel", "role", "digest", "std"} <= set(df.columns)
    assert len(df) == rec.n_channels
