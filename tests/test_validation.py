"""Synthetic validation: what the framework can and cannot do.

**These tests are slow by nature.** Each regenerates a corpus and re-derives a
measured finding, which is the only way to keep the findings falsifiable, and
it costs minutes rather than milliseconds. They carry the ``slow`` marker so
the everyday run stays quick:

    pytest -m "not slow"      # the library, in about four minutes
    pytest                    # everything, including the validation findings


These tests fix *measured findings*, including negative ones. A test that
asserts an advantage the library does not have is worse than no test; a test
that fixes the absence of an advantage stops it being quietly re-claimed.
"""
import numpy as np
import pytest
from scipy import stats

import recurra as rc
from recurra import statespace as sm

pytestmark = pytest.mark.slow

#: The conditions these findings were measured under, pinned here so that a
#: change of default cannot silently invalidate them. Line floors of 2: the
#: defaults until DD-98 moved them to 8, after which seven of these tests
#: failed for a year without anyone re-running them. A Theiler window of 2:
#: what ``theiler=1`` meant before DD-115 adopted the standard convention.
FLOORS = {"l_min": 2, "v_min": 2}
THEILER = 2
#: And the envelope smoother: the Butterworth that was the only one until
#: DD-117 made a sign-preserving Gaussian the default.
SMOOTH_METHOD = "butterworth"


def _rqa(source, **kw):
    """``rc.rqa`` at the measured floors, unless a test asks for others."""
    for key, value in FLOORS.items():
        kw.setdefault(key, value)
    return rc.rqa(source, **kw)


def _signal(modality, alpha, contamination, seed, duration=25.0):
    sig = rc.generate_cfc(modality=modality, duration=duration, fs=500.0,
                          alpha=alpha, preferred_phase=np.pi / 2, snr_db=20.0,
                          seed=seed, harmonic_contamination=contamination)
    return rc.analytic(rc.filterbank(sig.to_recording(f"s{seed}"),
                                     {"theta": (4, 8), "gamma": (50, 70)}),
                       sources=["theta", "gamma"])


def _measure(rec, metrics=("DET", "LAM", "TT", "L")):
    ph, am = rec.get("phase_theta"), rec.get("amp_gamma")
    ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD).subsample(max_points=2000)
    out = {"MI": rc.modulation_index(ph, am)}
    out.update(_rqa(ss, target_rr=0.05, theiler=THEILER, rng=0, metrics=list(metrics)))
    return out


def _group_effect(rows, metric):
    """Effect of being genuine, with log(MI) partialled out. Returns (t, p)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    g, h = df[df.kind == "genuine"], df[df.kind == "artefact"]
    lo, hi = max(g.MI.min(), h.MI.min()), min(g.MI.max(), h.MI.max())
    sub = df[(df.MI >= lo) & (df.MI <= hi)]
    if len(sub) < 12:
        pytest.skip("not enough overlap in the classical index")
    x = np.log(sub.MI.to_numpy(float))
    X = np.column_stack([np.ones(x.size), x,
                         (sub.kind == "genuine").to_numpy(float)])
    y = sub[metric].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = x.size - 3
    cov = (resid @ resid / dof) * np.linalg.inv(X.T @ X)
    t = beta[2] / np.sqrt(cov[2, 2])
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), dof)))


def _corpus(seed_base, n_seeds=8):
    """Genuine coupling and a harmonic artefact spanning the same MI range.

    **Every signal gets its own seed.** Reusing seeds across coupling levels
    gives many rows but few independent draws, and a regression that treats
    them as independent can miss a real effect entirely: with seven seeds
    reused across seven levels the group effect on DET came out at t = 0.12,
    and with twelve distinct seeds and the same protocol at t = 4.42 [DD-65].
    """
    rows = []
    seed = seed_base
    for alpha in (0.2, 0.35, 0.5, 0.65):
        for _ in range(n_seeds):
            seed += 1
            rows.append({"kind": "genuine", "seed": seed,
                         **_measure(_signal("pac", alpha, 0.0, seed))})
    for contam in (0.5, 0.9, 1.3, 1.7):
        for _ in range(n_seeds):
            seed += 1
            rows.append({"kind": "artefact", "seed": seed,
                         **_measure(_signal("none", 0.0, contam, seed))})
    return rows


@pytest.fixture(scope="module")
def matched_corpus():
    return _corpus(2_000_000)


def test_every_signal_has_its_own_seed(matched_corpus):
    """Guarding the design flaw that produced a false null [DD-65]."""
    seeds = [r["seed"] for r in matched_corpus]
    assert len(set(seeds)) == len(seeds)


def test_the_artefact_covers_the_same_classical_range(matched_corpus):
    """Without overlap the comparison is about coupling strength, not mechanism."""
    import pandas as pd

    df = pd.DataFrame(matched_corpus)
    g, h = df[df.kind == "genuine"].MI, df[df.kind == "artefact"].MI
    assert min(g.max(), h.max()) > max(g.min(), h.min()), "no overlap in MI"


def test_determinism_distinguishes_genuine_coupling_from_the_artefact(matched_corpus):
    """DD-65. At equal modulation index, DET is higher for genuine coupling.

    The effect is modest -- about 0.006 on a scale where DET is 0.93 -- so this
    test uses an uncorrected threshold and a directional expectation. It exists
    to keep the finding falsifiable, not to certify it: the definitive run had
    120 signals and this fixture has 64.
    """
    t, p = _group_effect(matched_corpus, "DET")
    assert t > 0, f"DET is higher in the artefact (t={t:.2f}), contradicting DD-65"
    assert p < 0.05, (
        f"DET no longer separates the two mechanisms (t={t:.2f}, p={p:.4f}). "
        "DD-65 measured t = 3.31 on 101 signals with unique seeds. Check for "
        "pseudo-replication before concluding the effect is gone.")


def test_the_vertical_family_points_the_other_way(matched_corpus):
    """Trapping time is *higher* for the artefact, opposite to the diagonal
    family. A mechanism that flipped the sign of one without the other would
    mean something changed [DD-65]."""
    t_det, _ = _group_effect(matched_corpus, "DET")
    t_tt, _ = _group_effect(matched_corpus, "TT")
    assert t_det > 0 > t_tt, (
        f"the two families no longer disagree in sign: DET t={t_det:.2f}, "
        f"TT t={t_tt:.2f}")


def test_classical_index_responds_across_the_sweep(matched_corpus):
    """The premise: MI varies within each mechanism, which is why matching on it
    is necessary and why the artefact is a problem worth testing."""
    import pandas as pd

    df = pd.DataFrame(matched_corpus)
    for kind in ("genuine", "artefact"):
        sub = df[df.kind == kind]
        assert sub.MI.max() > 4 * sub.MI.min(), (
            f"MI barely varies within {kind}; the sweep is not spanning a range")


# ------------------------------------------------- detection floor [DD-66]
CLASSICAL = ("MVL", "MI")


def _one_metric(rec, metric):
    """Either a classical index or a single RQA metric, whichever it is."""
    if metric in CLASSICAL:
        ph, am = rec.get("phase_theta"), rec.get("amp_gamma")
        return (rc.mvl(ph, am) if metric == "MVL"
                else rc.modulation_index(ph, am))
    ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD).subsample(max_points=2000)
    return _rqa(ss, target_rr=0.05, theiler=THEILER, rng=0, metrics=[metric])[metric]


def _detection_auc(metric, alpha, snr, n_seeds=7, seed_base=4_000_000):
    """Folded AUC of a metric separating coupled signals from matched controls."""
    from scipy.stats import mannwhitneyu

    seed = seed_base + int((snr + 20) * 1000) + int(alpha * 100) * 7
    coupled, control = [], []
    for _ in range(n_seeds):
        seed += 1
        coupled.append(_one_metric(_signal_snr("pac", alpha, snr, seed), metric))
        seed += 1
        control.append(_one_metric(_signal_snr("none", 0.0, snr, seed), metric))
    u, _ = mannwhitneyu(coupled, control, alternative="two-sided")
    n = n_seeds * n_seeds
    return max(u, n - u) / n


def _signal_snr(modality, alpha, snr, seed, duration=25.0):
    sig = rc.generate_cfc(modality=modality, duration=duration, fs=500.0,
                          alpha=alpha, preferred_phase=np.pi / 2, snr_db=snr,
                          seed=seed)
    return rc.analytic(rc.filterbank(sig.to_recording(f"s{seed}"),
                                     {"theta": (4, 8), "gamma": (50, 70)}),
                       sources=["theta", "gamma"])


def test_classical_indices_detect_weaker_coupling_than_det():
    """DD-66. The framework must not be sold on sensitivity: a modulation index
    detects coupling that DET misses, at a hundredth of the computation.

    Measured on the full surface: MI reaches AUC 0.90 at alpha 0.30 at every
    noise level; DET needs 0.90 at 20 dB and never reaches it below 10 dB.
    """
    mi = _detection_auc("MI", alpha=0.3, snr=10.0)
    det = _detection_auc("DET", alpha=0.3, snr=10.0)
    assert mi > 0.9, f"MI should separate at alpha=0.3 (got AUC {mi:.3f})"
    assert mi > det, (
        f"DET now detects alpha=0.3 as well as MI (MI {mi:.3f}, DET {det:.3f}). "
        "DD-66 measured MI far ahead; re-run the detection surface before "
        "changing what the papers claim.")


def test_strong_coupling_is_detected_by_everything():
    """The other end of the surface: at alpha 0.9 there is nothing subtle left."""
    for metric in ("MI", "DET", "LAM"):
        assert _detection_auc(metric, alpha=0.9, snr=20.0) > 0.85, metric


def test_the_vertical_family_detects_better_than_det():
    """DD-66: LAM reaches a floor of 0.50 where DET needs 0.90, inverting the
    usual emphasis on determinism."""
    lam = _detection_auc("LAM", alpha=0.5, snr=10.0)
    det = _detection_auc("DET", alpha=0.5, snr=10.0)
    assert lam > det, f"LAM {lam:.3f} no longer beats DET {det:.3f} at alpha=0.5"


# ------------------------------------------- window length [DD-67, DD-59]
def _windowed_metrics(modality, alpha, seconds, seed, metrics, n_windows=3):
    """Metrics on non-overlapping windows of one 40 s record, no decimation."""
    sig = rc.generate_cfc(modality=modality, duration=40.0, fs=500.0, alpha=alpha,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=seed)
    rec = rc.analytic(rc.filterbank(sig.to_recording(f"s{seed}"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD)
    wr = rc.windowed_recurrence(ss, rc.WindowSpec(size=seconds, step=seconds,
                                                  unit="seconds"),
                                target_rr=0.05, theiler=THEILER, rng=0)
    return [_rqa(wr[i], rng=0, metrics=list(metrics))
            for i in range(min(n_windows, len(wr)))]


@pytest.mark.parametrize("metric", ["RR", "DET", "LAM"])
def test_these_metrics_do_not_drift_with_window_length(metric):
    """DD-67: RR, DET and LAM are the only ones stable enough to compare
    between records of different length."""
    means = []
    for seconds in (1.0, 4.0):
        vals = [v[metric] for seed in (7_100_001, 7_100_002)
                for v in _windowed_metrics("pac", 0.7, seconds, seed, [metric])]
        means.append(float(np.mean(vals)))
    assert abs(means[1] - means[0]) / abs(means[0]) < 0.02, (
        f"{metric} drifted {100*(means[1]-means[0])/means[0]:+.1f}% between 1 s "
        f"and 4 s windows; DD-67 recorded it as stable")


def test_lmax_scales_with_the_window():
    """DD-67: Lmax is bounded by N, so it grows with the window and cannot be
    compared between records of different length at all."""
    means = []
    for seconds in (1.0, 4.0):
        vals = [v["Lmax"] for v in _windowed_metrics("pac", 0.7, seconds,
                                                     7_200_001, ["Lmax"])]
        means.append(float(np.mean(vals)))
    assert means[1] > 2.5 * means[0], (
        f"Lmax went from {means[0]:.0f} to {means[1]:.0f} for a fourfold window; "
        "DD-67 measured it scaling roughly with N")


def test_normalised_entropy_does_not_fix_the_length_dependence():
    """DD-59, corrected: dividing by log(distinct lengths) overcorrects, so
    ENTR_norm drifts *more* than raw ENTR and in the opposite direction."""
    raw, norm = [], []
    for seconds in (1.0, 4.0):
        vals = [v for seed in (7_300_001, 7_300_002)
                for v in _windowed_metrics("pac", 0.7, seconds, seed,
                                           ["ENTR", "ENTR_norm"])]
        raw.append(float(np.mean([v["ENTR"] for v in vals])))
        norm.append(float(np.mean([v["ENTR_norm"] for v in vals])))
    assert raw[1] > raw[0], "raw entropy should rise with N"
    assert norm[1] < norm[0], "normalised entropy should fall with N"


def test_one_second_windows_detect_coupling():
    """DD-67: trapping time separates coupled from control at 500 points, which
    sets the temporal resolution of the method at about one cycle of the
    modulating rhythm rather than tens."""
    from scipy.stats import mannwhitneyu

    coupled, control = [], []
    for k in range(5):
        coupled.append(np.mean([v["TT"] for v in _windowed_metrics(
            "pac", 0.7, 1.0, 7_400_000 + 2 * k, ["TT"], n_windows=4)]))
        control.append(np.mean([v["TT"] for v in _windowed_metrics(
            "none", 0.0, 1.0, 7_400_001 + 2 * k, ["TT"], n_windows=4)]))
    u, _ = mannwhitneyu(coupled, control, alternative="two-sided")
    auc = max(u, 25 - u) / 25
    assert auc >= 0.8, f"TT no longer detects at 1 s windows (AUC {auc:.2f})"


# ----------------------------------------------- window length [DD-67]
def _length_series(modality, alpha, seed, durations, metrics):
    """One record, windows of increasing length taken from it."""
    sig = rc.generate_cfc(modality=modality, duration=40.0, fs=500.0, alpha=alpha,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=seed)
    rec = rc.analytic(rc.filterbank(sig.to_recording(f"s{seed}"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD).subsample(5)          # 500 -> 100 Hz
    out = {}
    for dur in durations:
        win = ss.window(0, int(dur * ss.fs))
        out[dur] = _rqa(win, target_rr=0.05, theiler=THEILER, rng=0,
                          metrics=list(metrics))
    return out


def test_lam_works_in_a_shorter_window_than_det():
    """DD-67. LAM separates coupled from control at two seconds; DET needs
    sixteen. A four-second window -- a common default -- is below the length at
    which DET functions.
    """
    from scipy.stats import mannwhitneyu

    durations = (2.0,)
    metrics = ("DET", "LAM")
    seed = 8_000_000
    vals = {k: {m: [] for m in metrics} for k in ("coupled", "control")}
    for _ in range(7):
        for kind, modality, alpha in (("coupled", "pac", 0.7),
                                      ("control", "none", 0.0)):
            seed += 1
            got = _length_series(modality, alpha, seed, durations, metrics)
            for m in metrics:
                vals[kind][m].append(got[2.0][m])

    def auc(m):
        a, b = vals["coupled"][m], vals["control"][m]
        u, _ = mannwhitneyu(a, b, alternative="two-sided")
        n = len(a) * len(b)
        return max(u, n - u) / n

    assert auc("LAM") > auc("DET"), (
        f"at a two-second window DET now matches LAM (LAM {auc('LAM'):.3f}, "
        f"DET {auc('DET'):.3f}); DD-67 measured LAM far ahead")


def test_det_lam_and_l_are_length_invariant():
    """DD-67: the ratio family barely moves with window length, which is what
    lets it be compared across records once N is matched [DD-43]."""
    got = _length_series("pac", 0.7, 8_100_001, (2.0, 16.0), ("DET", "LAM", "L"))
    for m in ("DET", "LAM", "L"):
        ratio = got[16.0][m] / got[2.0][m]
        assert 0.85 < ratio < 1.15, f"{m} drifted by {ratio:.2f} with length"


def test_lmax_is_not_length_invariant():
    """The counter-case: Lmax is the longest line found, so it can only grow
    with the record and must never be compared across lengths [DD-67]."""
    got = _length_series("pac", 0.7, 8_200_001, (2.0, 16.0), ("Lmax",))
    assert got[16.0]["Lmax"] > 1.4 * got[2.0]["Lmax"]


# ------------------------------------------- window length [DD-67]
def _window_auc(metric, size_s, n_seeds=8, seed_base=9_000_000, alpha=0.7,
                snr=20.0, averaged=False, duration=30.0):
    """AUC separating coupled from control, from one window or from all of them."""
    from scipy.stats import mannwhitneyu

    seed = seed_base + int(size_s * 1000)
    groups = {"pac": [], "none": []}
    for modality in ("pac", "none"):
        for _ in range(n_seeds):
            seed += 1
            sig = rc.generate_cfc(modality=modality, duration=duration, fs=500.0,
                                  alpha=alpha, preferred_phase=np.pi / 2,
                                  snr_db=snr, seed=seed)
            rec = rc.analytic(rc.filterbank(sig.to_recording("s"),
                                            {"theta": (4, 8), "gamma": (50, 70)}),
                              sources=["theta", "gamma"])
            ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD)
            if averaged:
                wr = rc.windowed_recurrence(
                    ss, rc.WindowSpec(size=size_s, step=size_s, unit="seconds"),
                    target_rr=0.05, theiler=THEILER, rng=0)
                value = float(wr.metrics(rqa=[metric], **FLOORS)[metric].mean())
            else:
                win = ss.window(0, int(size_s * ss.fs))
                value = _rqa(win, target_rr=0.05, theiler=THEILER, rng=0,
                               metrics=[metric])[metric]
            groups[modality].append(value)
    a, b = groups["pac"], groups["none"]
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return max(u, len(a) * len(b) - u) / (len(a) * len(b))


def test_a_single_short_window_is_not_enough():
    """DD-69, and DD-68 by a different route: one window needs about 4 s -- 24 cycles of the slow rhythm --
    before any measure classifies a subject. Half a second does not."""
    short = _window_auc("LAM", 0.5)
    long_ = _window_auc("LAM", 4.0)
    assert long_ > short, (
        f"a 0.5 s window now matches a 4 s one (AUC {short:.3f} vs {long_:.3f}); "
        "DD-67 measured 0.76 against 0.92")
    assert long_ >= 0.85, f"LAM should separate at 4 s (got {long_:.3f})"


def test_averaging_many_short_windows_recovers_what_one_cannot():
    """DD-69: with the whole record available, cutting finely and averaging
    beats using few long windows. The two analyses answer different questions
    and must not be conflated."""
    # DD-68 measures the same floor on a decimated space and agrees.
    single = _window_auc("LAM", 1.0, averaged=False, duration=30.0)
    many = _window_auc("LAM", 1.0, averaged=True, duration=30.0)
    assert many >= single, (
        f"averaging {30/1.0:.0f} windows no longer helps (single {single:.3f}, "
        f"averaged {many:.3f})")
    assert many >= 0.9, f"averaged 1 s windows should separate (got {many:.3f})"


# --------------------------------------- the other three modalities [DD-70]
def _modality_auc(build, metric, alpha, n_seeds=7, seed_base=30_000_000):
    """AUC of one measure separating a coupled modality from a matched control."""
    from scipy.stats import mannwhitneyu

    seed = seed_base + int(alpha * 1000)
    coupled, control = [], []
    for _ in range(n_seeds):
        seed += 1
        coupled.append(build(alpha, seed)[metric])
        seed += 1
        control.append(build(0.0, seed)[metric])
    u, _ = mannwhitneyu(coupled, control, alternative="two-sided")
    n = n_seeds * n_seeds
    return max(u, n - u) / n


def _ppc(alpha, seed):
    sig = rc.generate_cfc(modality="ppc", duration=25.0, fs=500.0, alpha=alpha,
                          nm_ratio=(10, 1), f_low=6.0, f_high=60.0,
                          snr_db=20.0, seed=seed)
    rec = rc.analytic(rc.filterbank(sig.to_recording("p"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    out = {"PLV": rc.plv(rec.get("phase_theta"), rec.get("phase_gamma"), n=10, m=1)}
    joint = sm.ppc_space(rec, "phase_theta", "phase_gamma").subsample(max_points=1500)
    out.update(_rqa(joint, target_rr=0.05, theiler=THEILER, rng=0, metrics=["L", "DET"]))
    return out


def _ppa(alpha, seed):
    sig = rc.generate_cfc(modality="ppa", duration=25.0, fs=500.0, alpha=alpha,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=seed)
    rec = rc.analytic(rc.filterbank(sig.to_recording("q"),
                                    {"delta": (2, 4), "theta": (8, 12),
                                     "gamma": (50, 70)}),
                      sources=["delta", "theta", "gamma"])
    out = {"MI": rc.modulation_index(rec.get("phase_delta"), rec.get("amp_gamma"))}
    space = sm.ppa_space(rec, "phase_delta", "phase_theta",
                         "amp_gamma").subsample(max_points=1500)
    out.update(_rqa(space, target_rr=0.05, theiler=THEILER, rng=0, metrics=["LAM", "L"]))
    return out


def test_recurrence_beats_plv_on_phase_phase_coupling():
    """DD-70. The first case where a recurrence measure clearly wins.

    Band-pass filtering at the high band mixes the locked component with the
    free-running one, so the measured PLV collapses -- from about 0.80 to 0.12
    at alpha 0.9. PLV asks whether one phase difference is constant; mean
    diagonal line length asks whether two trajectories stay close for a while,
    and that survives the filter.
    """
    line = _modality_auc(_ppc, "L", alpha=0.9)
    plv = _modality_auc(_ppc, "PLV", alpha=0.9)
    assert line >= 0.9, f"L should separate PPC at alpha=0.9 (got {line:.3f})"
    assert line >= plv, (
        f"PLV now matches L on PPC (PLV {plv:.3f}, L {line:.3f}); DD-70 measured "
        "L at 1.00 against PLV 0.98, and 1.00 against 0.77 at alpha 0.3")


def test_only_recurrence_sees_trivariate_coupling():
    """DD-70. In PPA the fast envelope follows the *sum* of two slow phases, so
    a bivariate index between one of them and the envelope sees nothing. This
    is the clearest argument for the framework: a coupling with no adequate
    classical index."""
    lam = _modality_auc(_ppa, "LAM", alpha=0.9)
    mi = _modality_auc(_ppa, "MI", alpha=0.9)
    assert lam >= 0.9, f"LAM should separate PPA at alpha=0.9 (got {lam:.3f})"
    assert mi < 0.8, (
        f"the modulation index now detects trivariate coupling (AUC {mi:.3f}); "
        "DD-70 measured it at 0.61, never above 0.63 at any strength")


# ------------------------- dynamical measures on the artefact [DD-74]
def test_hjorth_activity_is_a_scale_confound_not_a_finding():
    """DD-74. Hjorth activity is the variance of the series, and the harmonic
    artefact adds energy in the high band by construction. It separates the two
    conditions strongly and means nothing dynamical.

    This test exists so the confound cannot quietly become a claim.
    """
    rows = []
    seed = 41_000_000
    for kind, alpha, contam in (("genuine", 0.4, 0.0), ("artefact", 0.0, 1.0)):
        for _ in range(6):
            seed += 1
            rec = _signal(("pac" if kind == "genuine" else "none"), alpha, contam, seed)
            ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD).subsample(max_points=1500)
            rows.append({"kind": kind,
                         **rc.dynamics_measures(ss, measures=["hjorth_activity",
                                                              "higuchi_fd"])})
    import pandas as pd

    df = pd.DataFrame(rows)
    g = df[df.kind == "genuine"]["hjorth_activity_amp_gamma"].mean()
    a = df[df.kind == "artefact"]["hjorth_activity_amp_gamma"].mean()
    assert a > g, (
        "the artefact no longer carries more envelope power than genuine "
        "coupling; DD-74 measured +8.4%, and the whole caveat rests on it")


def test_higuchi_of_the_envelope_is_scale_invariant():
    """DD-74: the reason Higuchi is the defensible discriminator and Hjorth
    activity is not. Scaling the series must not move the dimension."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3000)
    base = rc.higuchi_dimension(x).value
    scaled = rc.higuchi_dimension(100.0 * x).value
    assert scaled == pytest.approx(base, rel=1e-6)
    assert rc.hjorth_parameters(100.0 * x)["hjorth_activity"].value > \
           100 * rc.hjorth_parameters(x)["hjorth_activity"].value


# ----------------------------- both families on the surface [DD-75]
def test_phase_block_measures_are_blind_to_pac():
    """DD-75. PAC modulates the amplitude of the fast rhythm, not the dynamics
    of the slow phase, so no measure of the phase block should detect it. A
    measure that did would be a warning, not a result."""
    def phase_measure(alpha, seed):
        rec = _signal_snr("pac" if alpha > 0 else "none", alpha, 20.0, seed)
        ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD).subsample(max_points=1500)
        return rc.dynamics_measures(
            ss, measures=["higuchi_fd", "permutation_entropy"])

    from scipy.stats import mannwhitneyu

    coupled, control = [], []
    seed = 45_000_000
    for _ in range(8):
        seed += 1
        coupled.append(phase_measure(0.9, seed)["higuchi_fd_phase_theta"])
        seed += 1
        control.append(phase_measure(0.0, seed)["higuchi_fd_phase_theta"])
    u, _ = mannwhitneyu(coupled, control, alternative="two-sided")
    auc = max(u, 64 - u) / 64
    assert auc < 0.85, (
        f"a phase-block measure now detects PAC (AUC {auc:.3f}). Either the "
        "generator changed or the block series does; DD-75 measured no "
        "detection at any coupling strength or noise level")


def test_envelope_measures_do_detect_pac():
    """The other half of the same check: the amplitude block does see it."""
    from scipy.stats import mannwhitneyu

    coupled, control = [], []
    seed = 46_000_000
    for _ in range(8):
        seed += 1
        rec = _signal_snr("pac", 0.9, 20.0, seed)
        ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD).subsample(max_points=1500)
        coupled.append(rc.dynamics_measures(ss, measures=["hjorth_mobility"])
                       ["hjorth_mobility_amp_gamma"])
        seed += 1
        rec = _signal_snr("none", 0.0, 20.0, seed)
        ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD).subsample(max_points=1500)
        control.append(rc.dynamics_measures(ss, measures=["hjorth_mobility"])
                       ["hjorth_mobility_amp_gamma"])
    u, _ = mannwhitneyu(coupled, control, alternative="two-sided")
    assert max(u, 64 - u) / 64 >= 0.85


# ------------------------------------- short windows [DD-76]
def _short_window_auc(which, size_s, n_seeds=10, seed_base=75_000_000):
    """AUC from a single window of the given length."""
    from scipy.stats import mannwhitneyu

    seed = seed_base + int(size_s * 1000)
    a, b = [], []
    for _ in range(n_seeds):
        for target, modality in ((a, "pac"), (b, "none")):
            seed += 1
            rec = _signal_snr(modality, 0.7 if modality == "pac" else 0.0,
                              20.0, seed)
            ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD)
            win = ss.window(0, int(size_s * ss.fs))
            if which == "svd_entropy":
                target.append(rc.dynamics_measures(
                    win, measures=["svd_entropy"])["svd_entropy_amp_gamma"])
            else:
                target.append(_rqa(win, target_rr=0.05, theiler=THEILER, rng=0,
                                     metrics=[which])[which])
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return max(u, n_seeds ** 2 - u) / n_seeds ** 2


def test_svd_entropy_beats_rqa_on_a_one_second_window():
    """DD-76. A single window needs 2-4 s for RQA and 1 s for SVD entropy: a
    factor of four in the temporal resolution a time-resolved analysis can
    claim, from a measure outside the recurrence family."""
    svd = _short_window_auc("svd_entropy", 1.0)
    lam = _short_window_auc("LAM", 1.0)
    assert svd >= 0.85, f"SVD entropy should separate at 1 s (got {svd:.3f})"
    assert svd > lam, (
        f"LAM now matches SVD entropy at 1 s (LAM {lam:.3f}, SVD {svd:.3f}); "
        "DD-76 measured 0.92 against 0.68 on independent seeds")


def test_every_measure_is_computable_on_a_short_window():
    """A measure that quietly fails at 250 points would read as one that
    detects nothing [DD-76]."""
    rec = _signal_snr("pac", 0.7, 20.0, 76_000_001)
    ss = sm.pac_space(rec, smooth=8.0, smooth_method=SMOOTH_METHOD)
    win = ss.window(0, 250)
    values = rc.dynamics_measures(win, measures="all")
    missing = [k for k, v in values.items() if not np.isfinite(v)]
    assert not missing, f"not computable on 250 points: {missing}"


# ------------------- univariate is not a coupling measure [DD-77]
def _ppc_recording(alpha, seed):
    sig = rc.generate_cfc(modality="ppc", duration=25.0, fs=500.0, alpha=alpha,
                          nm_ratio=(10, 1), f_low=6.0, f_high=60.0,
                          snr_db=20.0, seed=seed)
    return rc.analytic(rc.filterbank(sig.to_recording("p"),
                                     {"theta": (4, 8), "gamma": (50, 70)}),
                       sources=["theta", "gamma"])


def _paired_measures(theta_rec, gamma_rec):
    n = min(theta_rec.n_samples, gamma_rec.n_samples)
    merged = rc.ingest({"phase_theta": np.asarray(theta_rec.get("phase_theta"))[:n],
                        "phase_gamma": np.asarray(gamma_rec.get("phase_gamma"))[:n]},
                       fs=500.0,
                       roles={"phase_theta": "phase", "phase_gamma": "phase"})
    ss = sm.ppc_space(merged, "phase_theta", "phase_gamma").subsample(max_points=1500)
    out = rc.dynamics_measures(ss, measures=["svd_entropy"])
    out["L"] = _rqa(ss, target_rr=0.05, theiler=THEILER, rng=0, metrics=["L"])["L"]
    return out


def test_a_univariate_measure_cannot_tell_coupling_from_regularity():
    """DD-77. Nine block-wise measures separated PPC at AUC 1.000, beating PLV.
    They are computed on the gamma block alone and never see theta.

    Pairing a regular gamma phase with an *independent* theta must break a
    coupling measure and must not break a regularity measure. That is the
    difference, and it is why no block-wise measure may be reported as evidence
    of coupling.
    """
    from scipy.stats import mannwhitneyu

    n = 7
    matched, mismatched, control = [], [], []
    seed = 95_000_000
    for _ in range(n):
        seed += 1
        coupled = _ppc_recording(0.9, seed)
        other = _ppc_recording(0.9, seed + 500)
        ctrl = _ppc_recording(0.0, seed + 1000)
        matched.append(_paired_measures(coupled, coupled))
        mismatched.append(_paired_measures(other, coupled))
        control.append(_paired_measures(ctrl, ctrl))

    def auc(a, b):
        u, _ = mannwhitneyu(a, b, alternative="two-sided")
        return max(u, len(a) * len(b) - u) / (len(a) * len(b))

    key = "svd_entropy_phase_gamma"
    uni_true = auc([m[key] for m in matched], [m[key] for m in control])
    uni_fake = auc([m[key] for m in mismatched], [m[key] for m in control])
    joint_true = auc([m["L"] for m in matched], [m["L"] for m in control])
    joint_fake = auc([m["L"] for m in mismatched], [m["L"] for m in control])

    assert uni_true > 0.85 and uni_fake > 0.85, (
        "the univariate measure no longer separates the mismatched pairing; "
        "DD-77 rests on it separating both equally")
    # The contrast, not an absolute gap: mismatching the pairing must damage
    # the joint measure and must not damage the univariate one. With seven per
    # group the absolute gap is noisy -- 0.30 in the exploratory run, 0.12
    # here -- but the ordering is the claim.
    assert joint_fake < uni_fake - 0.05, (
        f"mismatching the pairing no longer damages the joint measure more "
        f"than the univariate one (joint {joint_fake:.3f}, univariate "
        f"{uni_fake:.3f}). That asymmetry is what makes one a coupling measure "
        "and the other a regularity measure [DD-77].")
    assert joint_fake < joint_true, "the joint measure was not damaged at all"
