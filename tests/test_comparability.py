"""Tests for comparability contracts and group separation [DD-42..44]."""
import numpy as np
import pandas as pd
import pytest

import recurra as rc
from recurra import statespace as sm
from recurra.comparability import (
    METRIC_REQUIREMENTS,
    ComparabilityContract,
    check_comparability,
    contract_from,
    describe_units,
    feature_matrix,
    group_comparison,
    rank_metrics,
)


def _space(alpha, seed, duration):
    s = rc.generate_cfc(modality="pac", duration=duration, fs=500.0, alpha=alpha,
                        preferred_phase=np.pi / 2, snr_db=18.0, seed=seed)
    r = rc.analytic(rc.filterbank(s.to_recording(f"s{seed}"),
                                  {"theta": (4, 8), "gamma": (50, 70)}),
                    sources=["theta", "gamma"])
    return sm.pac_space(r, smooth=8.0)


@pytest.fixture(scope="module")
def unequal_spaces():
    """Realistic corpus: same protocol, different record lengths."""
    return {"sub00": _space(0.2, 10, 30.0), "sub01": _space(0.5, 11, 24.0),
            "sub02": _space(0.8, 12, 36.0)}


FIXED = rc.WindowSpec(size=6.0, overlap=0.5, unit="seconds")


# ------------------------------------------------- detecting the failures
def test_n_windows_on_unequal_records_is_flagged(unequal_spaces):
    """Failure mode 2: 'window 0' stops meaning the same thing."""
    batch = rc.batch_windowed_recurrence(unequal_spaces,
                                         rc.WindowSpec(n_windows=4, overlap=0.5),
                                         target_rr=0.05, theiler=1, rng=0)
    rep = batch.report
    assert not rep.ok
    violated = {a.axis for a in rep.violations}
    assert {"window_samples", "n_points"} <= violated


def test_n_windows_warns_in_batch(unequal_spaces):
    with pytest.warns(UserWarning, match="stops meaning the same thing"):
        rc.batch_windowed_recurrence(unequal_spaces,
                                     rc.WindowSpec(n_windows=4, overlap=0.5),
                                     target_rr=0.05, theiler=1, rng=0)


def test_fixed_window_size_is_comparable(unequal_spaces):
    batch = rc.batch_windowed_recurrence(unequal_spaces, FIXED, target_rr=0.05,
                                         theiler=1, rng=0)
    assert batch.report.ok
    m = batch.metrics()
    assert m["n_rows"].nunique() == 1               # same matrix size everywhere
    w0 = m[m["window"] == 0]
    assert w0["t_start_s"].nunique() == 1           # window 0 is the same stretch
    assert w0["t_stop_s"].nunique() == 1
    assert m.groupby("subject").size().nunique() > 1   # but counts may differ


def test_different_dimensions_are_caught():
    """Failure mode 1: auto embedding puts subjects in different spaces."""
    a = _space(0.5, 20, 20.0)
    units = {"a": a, "b": a.split(["phase_theta"])}
    rep = check_comparability(units)
    assert not rep.ok
    assert "dim" in {v.axis for v in rep.violations}


def test_different_effective_fs_is_caught():
    """Failure mode 3: asking for the same point count from records of
    different length decimates them by different factors, so their effective
    sampling rates diverge. Since DD-47 the rate is carried on the object, so
    the comparability check sees it."""
    a, b = _space(0.5, 21, 30.0), _space(0.5, 22, 20.0)
    sa, sb = a.subsample(max_points=3000), b.subsample(max_points=3000)
    assert sa.fs != sb.fs                       # the decimation was different
    rep = check_comparability({"a": sa, "b": sb})
    assert not rep.ok
    assert "fs" in {v.axis for v in rep.violations}
    # and the consequence: nothing in time units may be compared
    valid = rep.metric_validity().set_index("metric_family")
    assert not valid.loc["timescale", "comparable"]
    assert not valid.loc["invariant", "comparable"]


# ------------------------------------------------------------- contracts
def test_contract_declares_only_what_is_set():
    c = ComparabilityContract(dim=3, target_rr=0.05, metric=None, theiler=None)
    assert set(c.declared()) == {"dim", "target_rr"}


def test_contract_violation_is_named(unequal_spaces):
    batch = rc.batch_windowed_recurrence(unequal_spaces, FIXED, target_rr=0.05,
                                         theiler=1, rng=0)
    rep = batch.check_comparability(ComparabilityContract(dim=99))
    assert not rep.ok
    v = [a for a in rep.violations if a.axis == "dim"][0]
    assert v.required == 99 and len(v.offenders) == 3


def test_strict_contract_raises(unequal_spaces):
    batch = rc.batch_windowed_recurrence(unequal_spaces, FIXED, target_rr=0.05,
                                         theiler=1, rng=0)
    with pytest.raises(rc.ParameterError, match="contract violated"):
        batch.check_comparability(ComparabilityContract(dim=99, strict=True))


def test_contract_from_a_reference(unequal_spaces):
    batch = rc.batch_windowed_recurrence(unequal_spaces, FIXED, target_rr=0.05,
                                         theiler=1, rng=0)
    c = contract_from(batch["sub00"])
    assert c.dim == 3 and c.window_samples == 3000
    assert batch.check_comparability(c).ok


def test_tolerance_allows_small_numeric_drift():
    a, b = _space(0.5, 30, 20.0), _space(0.5, 31, 20.0)
    strict = check_comparability({"a": a, "b": b},
                                 ComparabilityContract(fs=500.0, tolerance=0.0))
    assert strict.ok


def test_describe_units_tabulates(unequal_spaces):
    df = describe_units(unequal_spaces)
    assert len(df) == 3
    assert {"label", "dim", "fs", "n_points", "route", "scaling"} <= set(df.columns)


# ----------------------------------------------- what may be compared [DD-43]
def test_matched_rate_licenses_ratios_not_the_rate(unequal_spaces):
    batch = rc.batch_windowed_recurrence(unequal_spaces, FIXED, target_rr=0.05,
                                         theiler=1, rng=0)
    valid = batch.report.metric_validity().set_index("metric_family")
    assert valid.loc["line_ratio", "comparable"]        # DET, LAM
    assert valid.loc["line_length", "comparable"]       # N is matched too
    assert not valid.loc["recurrence_rate", "comparable"]   # epsilon differs


def test_unmatched_length_forbids_length_metrics(unequal_spaces):
    batch = rc.batch_windowed_recurrence(unequal_spaces,
                                         rc.WindowSpec(n_windows=4, overlap=0.5),
                                         target_rr=0.05, theiler=1, rng=0)
    valid = batch.report.metric_validity().set_index("metric_family")
    assert not valid.loc["line_length", "comparable"]
    assert "n_points" in valid.loc["line_length", "missing"]
    assert valid.loc["line_ratio", "comparable"]        # ratios survive


def test_metric_requirements_cover_every_family():
    from recurra.comparability import METRIC_FAMILIES

    assert set(METRIC_REQUIREMENTS) == set(METRIC_FAMILIES)


# ------------------------------------------------------ group separation
@pytest.fixture(scope="module")
def two_groups():
    rng = np.random.default_rng(0)
    spaces, groups = {}, {}
    for i in range(12):
        grp = "control" if i < 6 else "study"
        alpha = rng.uniform(0.0, 0.2) if grp == "control" else rng.uniform(0.6, 0.9)
        dur = float(rng.choice([24.0, 30.0]))
        ss = _space(alpha, 500 + i, dur)
        spaces[f"sub{i:02d}"] = [ss.split(["phase_theta"]), ss.split(["amp_gamma"])]
        groups[f"sub{i:02d}"] = grp
    batch = rc.batch_windowed_recurrence(spaces, FIXED, kind="jrp", target_rr=0.10,
                                         theiler=1, rng=0)
    return batch, groups


def test_group_comparison_is_window_by_window(two_groups):
    """DD-44: window k of one subject against window k of another."""
    batch, groups = two_groups
    gc = batch.group_comparison(groups, values=["independence_ratio"])
    assert "window" in gc.columns
    assert gc["window"].nunique() > 1
    assert (gc["n_a"] >= 2).all() and (gc["n_b"] >= 2).all()


def test_group_comparison_separates_the_groups(two_groups):
    batch, groups = two_groups
    gc = batch.group_comparison(groups, values=["independence_ratio"])
    assert gc["mean_b"].mean() > gc["mean_a"].mean()   # study couples more
    assert gc["auc"].mean() > 0.85
    assert gc["cohens_d"].mean() > 1.0
    assert (gc["p_bonferroni"] < 0.05).mean() > 0.5


def test_group_comparison_reports_multiplicity(two_groups):
    batch, groups = two_groups
    gc = batch.group_comparison(groups, values=["independence_ratio"])
    assert gc["n_tests"].iloc[0] == len(gc)
    assert (gc["p_bonferroni"] >= gc["p_value"]).all()


def test_group_comparison_needs_exactly_two_groups(two_groups):
    batch, groups = two_groups
    three = {k: (v if i % 3 else "third") for i, (k, v) in enumerate(groups.items())}
    with pytest.raises(rc.ParameterError, match="exactly 2 groups"):
        batch.group_comparison(three)


def test_pooled_comparison_with_by_none(two_groups):
    batch, groups = two_groups
    m = batch.metrics().assign(group=lambda d: d["subject"].map(groups))
    gc = group_comparison(m, "group", values=["independence_ratio"], by=None)
    assert len(gc) == 1


def test_feature_matrix_shape(two_groups):
    batch, groups = two_groups
    X = batch.metrics_by_window("independence_ratio")
    assert X.shape[0] == 12
    assert set(X.index) == set(groups)


def test_feature_matrix_rejects_unknown_column(two_groups):
    batch, _ = two_groups
    with pytest.raises(rc.ParameterError, match="no column"):
        feature_matrix(batch.metrics(), "nonexistent")


def test_rank_metrics_orders_by_separation(two_groups):
    batch, groups = two_groups
    gc = batch.group_comparison(groups)
    ranked = rank_metrics(gc)
    assert "mean_effect" in ranked.columns
    assert ranked["mean_effect"].is_monotonic_decreasing


# ------------------------------------------------------------------ viz
@pytest.mark.requires("matplotlib")
def test_comparability_figures_render(two_groups):
    import matplotlib.pyplot as plt

    batch, groups = two_groups
    gc = batch.group_comparison(groups, values=["independence_ratio"])
    for fig in (rc.plot_group_comparison(gc),
                rc.plot_feature_matrix(batch.metrics_by_window("independence_ratio"),
                                       groups=groups),
                rc.plot_comparability(batch.report)):
        assert fig is not None
        plt.close(fig)


@pytest.mark.requires("matplotlib")
def test_group_figure_needs_one_metric(two_groups):
    batch, groups = two_groups
    gc = batch.group_comparison(groups)
    if gc["metric"].nunique() > 1:
        with pytest.raises(rc.ParameterError, match="name one"):
            rc.plot_group_comparison(gc)


@pytest.mark.requires("matplotlib")
def test_comparability_figure_carries_a_legend(two_groups):
    """Four statuses in colour alone is not readable [DD-55]."""
    import matplotlib.pyplot as plt

    batch, _ = two_groups
    fig = rc.plot_comparability(batch.report)
    axes_panel, families_panel = fig.get_axes()[:2]
    legend = axes_panel.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert any(lbl.startswith("ok:") for lbl in labels)
    assert families_panel.get_legend() is not None
    plt.close(fig)


@pytest.mark.requires("matplotlib")
def test_absent_axes_are_annotated(two_groups):
    """With per-window thresholds epsilon is absent, not violated, and the
    figure must say which [DD-55]."""
    import matplotlib.pyplot as plt

    batch, _ = two_groups
    statuses = {a.axis: a.status for a in batch.report.axes}
    assert statuses.get("epsilon") == "absent"
    fig = rc.plot_comparability(batch.report)
    notes = [t.get_text() for t in fig.get_axes()[0].texts]
    assert any("not reported" in n for n in notes)
    withheld = [t.get_text() for t in fig.get_axes()[1].texts]
    assert any(n.startswith("needs ") for n in withheld)
    plt.close(fig)


# ------------------------------------------------------------- DD-114
def test_group_timecourse_finds_a_late_effect_and_holds_its_level():
    """p_peak catches an effect confined to late windows; a null metric stays null."""
    import recurra as rc

    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        s, g = f"s{i:02d}", ("b" if i >= 15 else "a")
        for w in range(12):
            eff = 1.2 if (g == "b" and w >= 6) else 0.0
            rows.append({"subject": s, "group": g, "window": w,
                         "kept": rng.random() > 0.15,
                         "late": rng.normal() + eff,
                         "null": rng.normal()})
    table = pd.DataFrame(rows)
    course, summary = rc.group_timecourse(table, "group", values=["late", "null"],
                                          keep="kept", n_permutations=300, rng=0)
    late = summary[summary.metric == "late"].iloc[0]
    null = summary[summary.metric == "null"].iloc[0]
    assert late.p_peak < 0.05 and late.p_trend < 0.05
    assert null.p_peak > 0.05
    # the curve itself separates the halves
    c = course[course.metric == "late"]
    assert c[c.window >= 6].auc.mean() > c[c.window < 6].auc.mean() + 0.1
    # excluded rows stay excluded: counts reflect the mask, not the corpus
    assert (course.n_a + course.n_b).max() < 30 * 1.001


def test_group_timecourse_needs_two_groups_and_known_columns():
    import recurra as rc
    from recurra.exceptions import ParameterError

    table = pd.DataFrame({"subject": ["a", "b"], "group": ["x", "x"],
                          "window": [0, 0], "m": [1.0, 2.0]})
    with pytest.raises(ParameterError):
        rc.group_timecourse(table, "group", values=["m"], n_permutations=0)
    with pytest.raises(ParameterError):
        rc.group_timecourse(table, "group", values=["absent"], n_permutations=0)


@pytest.mark.requires("joblib")
def test_group_timecourse_p_is_independent_of_n_jobs():
    """Permutations are drawn before any split, so p_peak/p_trend must match."""
    rng = np.random.default_rng(2)
    rows = []
    for w in range(6):
        for i in range(20):
            rows.append({"subject": f"s{i}", "group": "a" if i < 12 else "b",
                         "window": w, "DET": rng.normal(), "kept": True})
    tab = pd.DataFrame(rows)
    out = [rc.group_timecourse(tab, "group", values=["DET"], keep="kept",
                               n_permutations=60, rng=7, n_jobs=j)
           for j in (1, 2)]
    pd.testing.assert_frame_equal(out[0][0], out[1][0])
    pd.testing.assert_frame_equal(out[0][1], out[1][1])

