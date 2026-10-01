import numpy as np
import pytest
from scipy.stats import spearmanr

import recurra as rc


def test_colored_noise_slopes():
    for beta in (0.0, 1.0, 2.0):
        x = rc.colored_noise(8192, beta=beta, fs=250.0, rng=1)
        f = np.fft.rfftfreq(8192, 1 / 250.0)
        P = np.abs(np.fft.rfft(x)) ** 2
        m = (f > 2) & (f < 60)
        slope = np.polyfit(np.log(f[m]), np.log(P[m]), 1)[0]
        assert slope == pytest.approx(-beta, abs=0.25)


@pytest.mark.parametrize("modality", ["pac", "ppc", "aac", "ppa", "none"])
def test_all_modalities_generate(modality):
    s = rc.generate_cfc(modality=modality, duration=6.0, fs=400.0, seed=0)
    assert s.data.shape == (2400, 1)
    assert np.all(np.isfinite(s.data))


def test_pac_strength_tracks_alpha():
    """H1.1 equivalent: the generator must be monotone in alpha."""
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mv, mi = [], []
    for a in alphas:
        s = rc.generate_cfc(modality="pac", duration=40.0, fs=500.0, alpha=a,
                            preferred_phase=np.pi / 2, snr_db=20.0, seed=11)
        ph, am = s.components["phase_low"], s.components["amp_high"]
        mv.append(rc.mvl(ph, am))
        mi.append(rc.modulation_index(ph, am))
    assert spearmanr(alphas, mv).statistic > 0.95
    assert spearmanr(alphas, mi).statistic > 0.95
    assert mv[0] < 0.05                      # alpha=0 -> essentially no coupling
    assert mv[-1] > 5 * mv[0]


def test_ppc_locks_phases():
    lo = rc.generate_cfc(modality="ppc", duration=30.0, fs=500.0, alpha=0.0, seed=5)
    hi = rc.generate_cfc(modality="ppc", duration=30.0, fs=500.0, alpha=1.0, seed=5)
    plv_lo = rc.plv(lo.components["phase_low"], lo.components["phase_high"])
    plv_hi = rc.plv(hi.components["phase_low"], hi.components["phase_high"])
    assert plv_hi > plv_lo


def test_aac_correlates_envelopes():
    lo = rc.generate_cfc(modality="aac", duration=30.0, fs=500.0, alpha=0.0, seed=6)
    hi = rc.generate_cfc(modality="aac", duration=30.0, fs=500.0, alpha=0.9, seed=6)
    r_lo = abs(rc.aac_index(lo.components["amp_low"], lo.components["amp_high"]))
    r_hi = abs(rc.aac_index(hi.components["amp_low"], hi.components["amp_high"]))
    assert r_hi > r_lo


def test_nonstationary_profiles():
    for kind in ("transient", "ramp", "intermittent", "switching"):
        s = rc.generate_cfc(modality="pac", duration=10.0, fs=400.0, alpha=0.8,
                            nonstationarity=kind, seed=2)
        assert s.truth["nonstationarity"] == kind
        assert not s.truth["alpha_profile_constant"]


def test_reproducible_from_seed():
    a = rc.generate_cfc(modality="pac", duration=5.0, fs=400.0, seed=99).data
    b = rc.generate_cfc(modality="pac", duration=5.0, fs=400.0, seed=99).data
    np.testing.assert_allclose(a, b)


def test_nyquist_violation_is_caught():
    with pytest.raises(rc.ParameterError, match="Nyquist"):
        rc.generate_cfc(f_high=200.0, bandwidth_high=40.0, fs=400.0, duration=4.0)


def test_parallel_rngs_are_independent():
    """Results must not depend on n_jobs."""
    a = rc.spawn_rngs(7, 4)
    b = rc.spawn_rngs(7, 4)
    va = [r.random() for r in a]
    vb = [r.random() for r in b]
    assert va == vb                       # reproducible
    assert len(set(va)) == 4              # and mutually distinct


# ---------------------------------------------------- PPC locking [DD-48]
def test_ppc_plv_rises_monotonically_with_alpha():
    """The bug this covers: an earlier version mixed unwrapped phases convexly,
    which produced a weighted-average frequency rather than locking. PLV was
    ~0 at every alpha except exactly 1."""
    alphas = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    values = []
    for a in alphas:
        s = rc.generate_cfc(modality="ppc", duration=60.0, fs=500.0, alpha=a,
                            nm_ratio=(10, 1), f_low=6.0, f_high=60.0,
                            snr_db=20.0, seed=18)
        values.append(rc.plv(s.components["phase_low"],
                             s.components["phase_high"], n=10, m=1))
    assert spearmanr(alphas, values).statistic > 0.95
    assert values[0] < 0.2            # no locking at alpha = 0
    assert values[-1] > 0.95          # full locking at alpha = 1
    assert values[3] > 3 * values[0]  # and something in between


def test_ppc_locked_component_stays_in_its_band():
    """The locked oscillation must survive a band-pass at the high band."""
    s = rc.generate_cfc(modality="ppc", duration=40.0, fs=500.0, alpha=0.9,
                        nm_ratio=(10, 1), f_low=6.0, f_high=60.0, seed=18)
    phi = np.unwrap(s.components["phase_high"])
    mean_freq = float(np.mean(np.diff(phi)) * 500.0 / (2 * np.pi))
    assert 50.0 < mean_freq < 70.0


def test_incoherent_nm_ratio_warns():
    with pytest.warns(rc.ParameterWarning, match="will not survive"):
        rc.generate_cfc(modality="ppc", duration=10.0, fs=500.0, alpha=0.9,
                        nm_ratio=(1, 1), f_low=6.0, f_high=60.0, seed=1)


# ------------------------------------------- the harmonic artefact [DD-64]
def _pac_indices(sig, seed=0):
    rec = rc.analytic(rc.filterbank(sig.to_recording(f"h{seed}"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    ph, am = rec.get("phase_theta"), rec.get("amp_gamma")
    return rc.mvl(ph, am), rc.modulation_index(ph, am)


def test_harmonic_contamination_actually_fakes_coupling():
    """The bug this covers: the artefact was advertised for several phases and
    produced nothing, because the distortion was applied to the waveform of a
    band-limited noise rather than to its phase [DD-64]. Every adverse-scenario
    test built on it was testing nothing."""
    clean = np.mean([_pac_indices(rc.generate_cfc(
        modality="none", duration=30.0, fs=500.0, snr_db=20.0,
        harmonic_contamination=0.0, seed=300 + k))[1] for k in range(3)])
    dirty = np.mean([_pac_indices(rc.generate_cfc(
        modality="none", duration=30.0, fs=500.0, snr_db=20.0,
        harmonic_contamination=2.0, seed=300 + k))[1] for k in range(3)])
    assert dirty > 10 * clean, (
        f"contamination raised MI only from {clean:.5f} to {dirty:.5f}; the "
        "artefact is not being produced")


def test_harmonic_artefact_is_monotone_in_contamination():
    values = [np.mean([_pac_indices(rc.generate_cfc(
        modality="none", duration=30.0, fs=500.0, snr_db=20.0,
        harmonic_contamination=c, seed=400 + k))[1] for k in range(3)])
        for c in (0.0, 0.5, 1.0, 2.0)]
    assert values == sorted(values)


def _harmonic_ratio(sig, f_low=6.0, harmonics=(2, 3, 4)):
    """Power at multiples of the slow frequency, relative to the fundamental.

    This is what "non-sinusoidal slow rhythm" means, measured directly. The
    total power in the high band is useless for the purpose, because the
    uncoupled control already puts half its power there through its own fast
    component.
    """
    from scipy import signal as sps

    f, P = sps.welch(sig.data[:, 0], fs=sig.fs, nperseg=4096)
    def near(target):
        return P[np.abs(f - target) < 1.5].sum()
    return sum(near(h * f_low) for h in harmonics) / (near(f_low) or 1e-30)


def test_harmonic_artefact_makes_the_slow_rhythm_non_sinusoidal():
    """The mechanism, measured directly: harmonics of the slow frequency."""
    clean = rc.generate_cfc(modality="none", duration=30.0, fs=500.0, snr_db=40.0,
                            harmonic_contamination=0.0, seed=1)
    dirty = rc.generate_cfc(modality="none", duration=30.0, fs=500.0, snr_db=40.0,
                            harmonic_contamination=2.0, seed=1)
    assert _harmonic_ratio(dirty) > 3 * _harmonic_ratio(clean)


def test_sharpness_controls_how_far_the_harmonics_reach():
    ratios = [_harmonic_ratio(
        rc.generate_cfc(modality="none", duration=20.0, fs=500.0, snr_db=40.0,
                        harmonic_contamination=2.0, harmonic_sharpness=k, seed=2),
        harmonics=(6, 8, 10))
        for k in (5.0, 60.0)]
    assert ratios[1] > ratios[0]


def test_truth_records_the_sharpness():
    sig = rc.generate_cfc(modality="none", duration=10.0, fs=500.0,
                          harmonic_contamination=1.0, harmonic_sharpness=25.0,
                          seed=3)
    assert sig.truth["harmonic_sharpness"] == 25.0
    assert sig.truth["harmonic_contamination"] == 1.0


# ---------------------------- the picture that shows coupling [DD-106]
def _pac_pair(alpha, n=20000, fs=250.0, seed=0, tail=0.8):
    """A phase and a heavy-tailed envelope, like a real gamma envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    phase = 2 * np.pi * 6 * t + 0.3 * np.cumsum(rng.standard_normal(n)) / 50
    env = (np.abs(1 + alpha * np.cos(phase - np.pi / 2))
           * np.exp(tail * rng.standard_normal(n)))
    return np.angle(np.exp(1j * phase)), env


def test_a_scatter_cannot_show_coupling_but_the_mean_can():
    """DD-106. The spread of amplitude within one phase bin is several times
    the movement of the mean across bins, so individual points drown the
    effect even when the coupling is strong. Averaging recovers it."""
    phase, env = _pac_pair(0.3)
    prof = rc.modulogram(phase, env)
    idx = np.clip(np.digitize(phase, np.linspace(-np.pi, np.pi, 19)) - 1, 0, 17)
    within = np.mean([env[idx == k].std() for k in range(18)])
    across = np.array([env[idx == k].mean() for k in range(18)]).std()
    assert within > 3 * across, (
        "the scatter no longer drowns the effect; DD-106 rests on it doing so")
    # and the mean profile still finds the coupling
    assert prof["amplitude"].max() - prof["amplitude"].min() > 0.02


def test_the_modulogram_recovers_the_preferred_phase():
    for alpha in (0.3, 0.9):
        phase, env = _pac_pair(alpha)
        prof = rc.modulogram(phase, env)
        peak = float(prof.loc[prof["amplitude"].idxmax(), "phase"])
        assert peak == pytest.approx(np.pi / 2, abs=0.4), (
            f"peak at {peak:.2f} for alpha={alpha}, true phase is pi/2")


def test_the_modulogram_is_flat_without_coupling():
    phase, env = _pac_pair(0.0)
    prof = rc.modulogram(phase, env)
    contrast = prof["amplitude"].max() - prof["amplitude"].min()
    coupled = rc.modulogram(*_pac_pair(0.9))
    assert contrast < 0.2 * (coupled["amplitude"].max()
                             - coupled["amplitude"].min())


def test_the_modulogram_reports_its_own_uncertainty():
    """A bin with few samples wanders on its own; a peak inside the error bars
    is not a peak."""
    phase, env = _pac_pair(0.5)
    prof = rc.modulogram(phase, env)
    assert {"phase", "amplitude", "sem", "count"} <= set(prof.columns)
    assert (prof["count"] > 0).all()
    assert np.isfinite(prof["sem"]).all()


def test_the_comodulogram_finds_which_pair_couples():
    """The map that says which band pair couples rather than assuming one."""
    n, fs = 15000, 250.0
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    phase, env = _pac_pair(0.6, n=n, fs=fs, seed=1)
    rec = rc.ingest({"phase_theta": phase,
                     "phase_alpha": np.angle(np.exp(1j * 2 * np.pi * 10 * t)),
                     "amp_gamma": env,
                     "amp_beta": np.abs(rng.standard_normal(n))}, fs=fs,
                    roles={"phase_theta": "phase", "phase_alpha": "phase",
                           "amp_gamma": "amplitude", "amp_beta": "amplitude"})
    frame = rc.comodulogram(rec)
    assert len(frame) == 4
    best = frame.loc[frame["value"].idxmax()]
    assert best["phase_band"] == "phase_theta"
    assert best["amplitude_band"] == "amp_gamma"
    others = frame[frame["value"] < best["value"]]["value"].max()
    assert best["value"] > 10 * others


def test_coupling_indices_carry_the_modulogram_shape():
    phase, env = _pac_pair(0.6)
    out = rc.coupling_indices(phase, env)
    assert {"mvl", "mi_tort", "mod_contrast", "preferred_phase"} <= set(out)
    assert out["preferred_phase"] == pytest.approx(np.pi / 2, abs=0.4)


def test_a_comodulogram_needs_both_roles():
    rec = rc.ingest({"amp_gamma": np.abs(np.random.default_rng(0).standard_normal(2000))},
                    fs=250.0, roles={"amp_gamma": "amplitude"})
    with pytest.raises(rc.ParameterError, match="at least one phase channel"):
        rc.comodulogram(rec)


def test_surrogates_are_reachable_from_the_top_level():
    """It existed from the first phase and was only ever reachable as
    recurra.metrics.classic.surrogate_significance, which is where a user
    running a study will not look. The omission cost a full 31-channel run."""
    assert hasattr(rc, "surrogate_significance")


def test_surrogates_separate_real_coupling_from_its_own_null():
    """The baseline for 'no coupling' cannot be a synthetic number: real
    signals are 1/f with non-sinusoidal waveforms and both raise the
    modulation index without any coupling. Measured, the surrogate null of a
    realistic signal came out 0.0075 against 0.00002 for a synthetic one --
    375 times higher, and the difference between a finding and a scare."""
    coupled = _pac_pair(0.6, n=12000)
    flat = _pac_pair(0.0, n=12000, seed=3)
    hit = rc.surrogate_significance(rc.modulation_index, *coupled,
                                    n_surrogates=200, rng=0)
    miss = rc.surrogate_significance(rc.modulation_index, *flat,
                                     n_surrogates=200, rng=0)
    assert hit["p_value"] < 0.05
    assert miss["p_value"] > 0.05
    # the null is a property of the signal, not a universal constant
    assert hit["null_mean"] > 10 * miss["null_mean"]


def test_the_surrogate_z_score_saturates_and_the_p_value_does_not():
    """DD-107. A circular shift preserves the envelope entirely, so an envelope
    genuinely modulated at the phase frequency stays modulated after shifting
    -- only its preferred phase moves. The null therefore rises with the
    signal and the z-score saturates: measured, 2.04 at alpha 0.2 and 2.61 at
    0.9, where the modulation index itself rose twentyfold.

    Report the p-value. A z of 2.6 is not weak evidence here; it is the
    ceiling of this surrogate.
    """
    zs, ps = [], []
    for alpha in (0.2, 0.9):
        out = rc.surrogate_significance(rc.modulation_index,
                                        *_pac_pair(alpha, n=12000),
                                        n_surrogates=200, rng=0)
        zs.append(out["z_score"]); ps.append(out["p_value"])
    assert zs[1] < 2 * zs[0], (
        f"the z-score no longer saturates ({zs}); DD-107 rests on it doing so")
    assert all(p < 0.05 for p in ps)


def test_modulogram_wraps_the_phase_like_the_modulation_index():
    """A phase given in [0, 2*pi) must bin as the same phase in (-pi, pi]."""
    rng = np.random.default_rng(3)
    ph = rng.uniform(0, 2 * np.pi, 20000)
    am = 1 + 0.5 * np.cos(ph)
    a = rc.modulogram(ph, am)["amplitude"].to_numpy()
    b = rc.modulogram(np.angle(np.exp(1j * ph)), am)["amplitude"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_canonical_indices_flag_an_amplitude_that_is_not_an_envelope():
    """MVL and MI are defined on an envelope. A negative 'amplitude' warns, and
    MI refuses once a phase bin's mean amplitude is negative, because the
    divergence is then taken from something that is not a distribution."""
    rng = np.random.default_rng(4)
    ph = rng.uniform(-np.pi, np.pi, 20000)
    env = 1 + 0.5 * np.cos(ph)
    with pytest.warns(rc.ParameterWarning, match="not an envelope"):
        rc.mvl(ph, env - 1.2)
    with pytest.warns(rc.ParameterWarning, match="not an envelope"):
        assert np.isnan(rc.modulation_index(ph, env - 1.2))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", rc.ParameterWarning)
        assert rc.modulation_index(ph, env) > 0      # a real envelope is silent
