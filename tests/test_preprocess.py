import numpy as np
import pytest

import recurra as rc
from recurra.preprocess.filters import design_bandpass


def test_auto_order_guarantees_minimum_cycles():
    d = design_bandpass((4, 8), fs=500.0, method="fir", order="auto", min_cycles=3.0)
    assert d.cycles >= 3.0
    assert d.order % 2 == 1          # odd -> exact linear phase


def test_band_outside_nyquist_rejected():
    with pytest.raises(rc.ParameterError):
        design_bandpass((10, 300), fs=500.0)


def test_bandpass_isolates_the_band():
    fs, n = 500.0, 10000
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 6 * t) + np.sin(2 * np.pi * 60 * t)
    rec = rc.ingest(x, fs=fs)
    out = rc.bandpass(rec, (4, 8), name="theta")
    y = np.asarray(out.get("theta"))[500:-500]
    f = np.fft.rfftfreq(y.size, 1 / fs)
    peak = f[np.argmax(np.abs(np.fft.rfft(y)))]
    assert 4 < peak < 8


def test_bandpass_records_edge_cost():
    rec = rc.ingest(np.random.default_rng(3200).standard_normal(6000), fs=500.0)
    out = rc.bandpass(rec, (4, 8))
    step = [s for s in out.provenance if s.name == "bandpass"][0]
    assert step.params["edge_samples"] > 0
    assert any("transient" in c for c in step.caveats)


def test_analytic_produces_all_three_components():
    rec = rc.ingest(np.random.default_rng(4000).standard_normal(8000), fs=500.0)
    rec = rc.bandpass(rec, (8, 12), name="alpha")
    out = rc.analytic(rec, sources=["alpha"])
    assert {"phase_alpha", "amp_alpha", "ifreq_alpha"} <= set(out.names)
    assert rc.Capability.PHASE in out.caps and rc.Capability.AMPLITUDE in out.caps


def test_analytic_phase_is_wrapped_and_amplitude_positive():
    fs, n = 400.0, 6000
    t = np.arange(n) / fs
    rec = rc.ingest(np.sin(2 * np.pi * 10 * t), fs=fs)
    rec = rc.bandpass(rec, (8, 12), name="a")
    out = rc.analytic(rec, sources=["a"])
    ph = np.asarray(out.get("phase_a"))
    am = np.asarray(out.get("amp_a"))
    assert ph.min() >= -np.pi - 1e-9 and ph.max() <= np.pi + 1e-9
    assert (am >= 0).all()


def test_edge_trim_shortens_and_is_recorded():
    rec = rc.ingest(np.random.default_rng(6000).standard_normal(10000), fs=500.0)
    rec = rc.bandpass(rec, (4, 8), name="theta")
    out = rc.analytic(rec, sources=["theta"], edge_policy="trim")
    assert out.n_samples < rec.n_samples
    trim = [s for s in out.provenance if s.name == "edge_trim"][0]
    assert trim.params["n_after"] == out.n_samples
    assert out.t0 > rec.t0


def test_mirror_policy_preserves_length():
    rec = rc.ingest(np.random.default_rng(7000).standard_normal(8000), fs=500.0)
    rec = rc.bandpass(rec, (4, 8), name="theta")
    out = rc.analytic(rec, sources=["theta"], edge_policy="mirror")
    assert out.n_samples == rec.n_samples


def test_resample_changes_fs_and_length():
    rec = rc.ingest(np.random.default_rng(7700).standard_normal(4000), fs=500.0)
    out = rc.resample(rec, 250.0)
    assert out.fs == 250.0
    assert abs(out.n_samples - 2000) <= 2


def test_full_chain_recovers_known_pac():
    """End to end: generate -> ingest -> filter -> Hilbert -> canonical index."""
    strong = rc.generate_cfc(modality="pac", duration=30.0, fs=500.0, alpha=0.9,
                             preferred_phase=np.pi / 2, snr_db=25.0, seed=17)
    weak = rc.generate_cfc(modality="pac", duration=30.0, fs=500.0, alpha=0.0,
                           snr_db=25.0, seed=17)
    vals = []
    for sig in (weak, strong):
        rec = sig.to_recording()
        rec = rc.filterbank(rec, {"lo": (4, 8), "hi": (50, 70)})
        rec = rc.analytic(rec, sources=["lo", "hi"])
        vals.append(rc.mvl(rec.get("phase_lo"), rec.get("amp_hi")))
    assert vals[1] > 3 * vals[0]
