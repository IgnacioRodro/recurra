import numpy as np
import pytest

import recurra as rc

T, FS = 800, 200.0


def test_1d_array():
    r = rc.ingest(np.random.default_rng(1000).standard_normal(T), fs=FS)
    assert r.n_channels == 1 and r.n_samples == T
    assert rc.Capability.RAW in r.caps


def test_2d_array_with_labels():
    r = rc.ingest(np.random.default_rng(1600).standard_normal((T, 3)), fs=FS, labels=["a", "b", "c"])
    assert r.names == ["a", "b", "c"]


def test_transposed_input_is_rejected_with_advice():
    with pytest.raises(rc.IngestError, match="transpose"):
        rc.ingest(np.random.default_rng(2200).standard_normal((3, T)), fs=FS)


def test_dict_of_hilbert_components_sets_roles():
    r = rc.ingest({"phase_theta": np.random.default_rng(2600).random(T), "amp_gamma": np.random.default_rng(2601).random(T)},
                  fs=FS)
    assert rc.Capability.PHASE in r.caps and rc.Capability.AMPLITUDE in r.caps
    assert rc.Capability.RAW not in r.caps


def test_explicit_roles_beat_name_guessing():
    r = rc.ingest({"phase_theta": np.random.default_rng(3300).random(T)}, fs=FS,
                  roles={"phase_theta": "raw"})
    assert r.roles["phase_theta"] == "raw"


def test_complex_channel_is_split():
    z = np.exp(1j * np.linspace(0, 20, T))
    r = rc.ingest({"sig": z}, fs=FS)
    assert "sig_phase" in r.names and "sig_amplitude" in r.names


def test_tuple_phase_amplitude():
    r = rc.ingest((np.random.default_rng(4500).random((T)), np.random.default_rng(4501).random(T)), fs=FS)
    assert r.roles["phase"] == "phase" and r.roles["amplitude"] == "amplitude"


def test_statespace_kind():
    r = rc.ingest(np.random.default_rng(5000).standard_normal((T, 3)), fs=FS, kind="statespace")
    assert all(v == "state" for v in r.roles.values())
    assert rc.Capability.STATESPACE in r.caps


def test_missing_fs_is_an_error():
    with pytest.raises(rc.IngestError, match="fs is required"):
        rc.ingest(np.random.default_rng(5700).standard_normal(T))


def test_recording_passthrough_is_idempotent():
    r = rc.ingest(np.random.default_rng(6100).standard_normal(T), fs=FS)
    assert rc.ingest(r).n_samples == r.n_samples


def test_csv_roundtrip(tmp_path):
    import pandas as pd

    p = tmp_path / "d.csv"
    pd.DataFrame({"time": np.arange(T) / FS, "ch0": np.random.default_rng(6900).standard_normal(T)}).to_csv(p, index=False)
    r = rc.ingest(str(p))
    assert r.n_channels == 1 and abs(r.fs - FS) < 1e-6


def test_quality_flags_constant_channel():
    r = rc.ingest({"a": np.random.default_rng(7500).standard_normal(T), "flat": np.ones(T)}, fs=FS)
    assert not r.quality.ok
    assert "CONSTANT" in set(r.quality.to_frame()["code"])
