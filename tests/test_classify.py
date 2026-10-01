"""Tests for F13: classification evaluated so the number can be believed."""
import numpy as np
import pandas as pd
import pytest

import recurra as rc
from recurra import statespace as sm
from recurra.classify.features import build_features, select_features


@pytest.fixture(scope="module")
def corpus():
    """Sixteen subjects, deliberately imbalanced, with windowed metrics."""
    rng = np.random.default_rng(0)
    spaces, labels = {}, {}
    for i in range(16):
        group = "control" if i < 10 else "study"
        alpha = rng.uniform(0.0, 0.25) if group == "control" else rng.uniform(0.55, 0.85)
        sig = rc.generate_cfc(modality="pac", duration=24.0, fs=500.0, alpha=alpha,
                              preferred_phase=np.pi / 2, snr_db=20.0, seed=600 + i)
        rec = rc.analytic(rc.filterbank(sig.to_recording(f"sub{i:02d}"),
                                        {"theta": (4, 8), "gamma": (50, 70)}),
                          sources=["theta", "gamma"])
        spaces[f"sub{i:02d}"] = sm.pac_space(rec, smooth=8.0).subsample(max_points=3000)
        labels[f"sub{i:02d}"] = group
    batch = rc.batch_windowed_recurrence(spaces, rc.WindowSpec(size=6.0, overlap=0.5),
                                         target_rr=0.05, theiler=1, rng=0)
    return batch.metrics(rqa=True, dynamics=True), labels


# ---------------------------------------------------------- feature sets
def test_families_select_the_right_columns(corpus):
    table, _ = corpus
    rqa = select_features(table, "rqa")
    dyn = select_features(table, "dynamics")
    assert "DET" in rqa and "LAM" in rqa
    assert any(c.startswith("higuchi_fd") for c in dyn)
    assert not set(rqa) & set(dyn)
    assert set(select_features(table, "both")) == set(rqa) | set(dyn)


def test_bookkeeping_columns_are_never_features(corpus):
    table, _ = corpus
    chosen = set(select_features(table, "all"))
    for leak in ("window", "t_center_s", "n_rows", "theiler", "target_rr"):
        assert leak not in chosen, f"{leak} would leak the design into the model"


def test_unknown_family_is_rejected(corpus):
    table, _ = corpus
    with pytest.raises(rc.ParameterError, match="unknown family"):
        select_features(table, "astrology")


def test_subject_level_reduces_to_one_row_each(corpus):
    table, _ = corpus
    X, groups = build_features(table, families="rqa", level="subject")
    assert len(X) == table["subject"].nunique() == len(np.unique(groups))
    X2, _ = build_features(table, families="rqa", level="subject",
                           aggregate="mean_std_trend")
    assert X2.shape[1] == 3 * X.shape[1]


def test_window_level_keeps_the_rows_and_names_their_subject(corpus):
    table, _ = corpus
    X, groups = build_features(table, families="rqa", level="window")
    assert len(X) == len(table)
    assert len(np.unique(groups)) == table["subject"].nunique()


# ------------------------------------------------- the leakage guard [DD-83]
@pytest.mark.requires("sklearn")
def test_windows_of_a_subject_never_straddle_a_fold(corpus):
    """The result this whole module depends on. If a subject's windows appear
    in training and testing, the model recognises the subject, not the group.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    table, labels = corpus
    X, groups = build_features(table, families="rqa", level="window")
    y = np.asarray([labels[g] == "study" for g in groups]).astype(int)
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0)
    for train, test in splitter.split(X, y, groups):
        assert not set(groups[train]) & set(groups[test])


@pytest.mark.requires("sklearn")
def test_shuffled_labels_score_at_chance(corpus):
    """A pipeline that leaks would score above chance on random labels. This is
    the check that a passing AUC is worth anything at all [DD-83, DD-84]."""
    table, labels = corpus
    subjects = sorted(set(labels))
    rng = np.random.default_rng(7)
    fake = dict(zip(subjects, rng.permutation(list(labels.values())), strict=False))
    report = rc.classify_groups(table, fake, families="rqa", level="window",
                                n_repeats=4, seed=1)
    assert report.auc < 0.85, (
        f"random labels reach AUC {report.auc:.3f}. Either the folds leak or "
        "the shuffle was not independent of the features.")


# --------------------------------------------------------- what it reports
@pytest.mark.requires("sklearn")
def test_every_requested_score_is_present(corpus):
    table, labels = corpus
    report = rc.classify_groups(table, labels, families="rqa", n_repeats=3)
    for score in ("auc", "accuracy", "balanced_accuracy", "sensitivity",
                  "specificity", "precision", "f1", "mcc"):
        assert score in report.scores.columns
        assert np.isfinite(report.scores[score]).any()


@pytest.mark.requires("sklearn")
def test_imbalance_is_reported_with_a_baseline(corpus):
    """DD-85: accuracy alone on 10 against 6 is not interpretable."""
    table, labels = corpus
    report = rc.classify_groups(table, labels, families="rqa", n_repeats=3)
    assert report.class_counts == {"control": 10, "study": 6}
    assert report.baseline["accuracy"] == pytest.approx(10 / 16)
    assert report.baseline["balanced_accuracy"] == 0.5


@pytest.mark.requires("sklearn")
def test_repeats_give_a_spread(corpus):
    """DD-86: one split is an anecdote."""
    table, labels = corpus
    report = rc.classify_groups(table, labels, families="rqa", n_repeats=6)
    assert len(report.scores) == 6
    assert report.scores["auc"].std(ddof=1) >= 0


@pytest.mark.requires("sklearn")
def test_small_sample_is_flagged(corpus):
    table, labels = corpus
    report = rc.classify_groups(table, labels, families="rqa", n_repeats=2)
    assert any("small sample" in w for w in report.settings["warnings"])


@pytest.mark.requires("sklearn")
def test_more_features_than_subjects_is_flagged(corpus):
    table, labels = corpus
    report = rc.classify_groups(table, labels, families="all", n_repeats=2,
                                aggregate="mean_std_trend")
    assert any("degrees of freedom" in w for w in report.settings["warnings"])


@pytest.mark.requires("sklearn")
def test_report_exports_and_summarises(corpus):
    table, labels = corpus
    report = rc.classify_groups(table, labels, families="rqa", n_repeats=3)
    frame = report.to_frame()
    assert {"auc_mean", "auc_sd", "baseline_accuracy", "n_control"} <= set(frame.columns)
    assert "ClassificationReport" in report.summary()
    assert report.importances is not None and len(report.importances) > 0


# ------------------------------------------------------- permutation [DD-87]
@pytest.mark.requires("sklearn")
def test_permutation_null_is_centred_at_chance(corpus):
    table, labels = corpus
    report = rc.classify_groups(table, labels, families="rqa", n_repeats=2,
                                permutations=25, seed=3)
    perm = report.permutation
    assert perm["n_permutations"] == 25
    assert perm["null_mean"] == pytest.approx(0.5, abs=0.2), (
        "the permutation null is not at chance, which means the shuffle is not "
        "breaking the label-feature relation")
    assert perm["p_value"] < 0.2
    assert perm["p_value"] >= 1 / 26      # can never be reported as zero


@pytest.mark.requires("sklearn")
def test_permutation_shuffles_whole_subjects(corpus):
    """DD-87: shuffling rows rather than subjects would leave a subject with
    two labels and make the null far too easy to beat."""
    table, labels = corpus
    report = rc.classify_groups(table, labels, families="rqa", level="window",
                                n_repeats=1, permutations=15, seed=4)
    assert report.permutation["null_mean"] < 0.8


# ------------------------------------------------------------- interface
@pytest.mark.requires("sklearn")
def test_needs_exactly_two_groups(corpus):
    table, labels = corpus
    three = {k: (v if i % 3 else "third") for i, (k, v) in enumerate(labels.items())}
    with pytest.raises(rc.ParameterError, match="exactly two groups"):
        rc.classify_groups(table, three, families="rqa")


@pytest.mark.requires("sklearn")
def test_missing_label_is_reported(corpus):
    table, labels = corpus
    partial = {k: v for k, v in list(labels.items())[:5]}
    with pytest.raises(rc.ParameterError, match="no label for subject"):
        rc.classify_groups(table, partial, families="rqa")


@pytest.mark.parametrize("model", ["logistic", "forest"])
@pytest.mark.requires("sklearn")
def test_models_run(corpus, model):
    table, labels = corpus
    report = rc.classify_groups(table, labels, families="rqa", model=model,
                                n_repeats=2)
    assert 0.0 <= report.auc <= 1.0


@pytest.mark.requires("sklearn")
def test_unknown_model_is_rejected(corpus):
    table, labels = corpus
    with pytest.raises(rc.ParameterError, match="model must be"):
        rc.classify_groups(table, labels, families="rqa", model="crystal ball")


@pytest.mark.requires("sklearn")
def test_whole_recording_is_a_single_window(corpus):
    """The whole-signal case is the same code with one window per subject."""
    table, labels = corpus
    single = table[table["window"] == 0].copy()
    report = rc.classify_groups(single, labels, families="rqa", n_repeats=3)
    assert report.settings["n_rows"] == 16
    assert 0.0 <= report.auc <= 1.0


# ------------------------------------------------------------- DD-114
def test_classify_timecourse_returns_one_honest_row_per_window():
    pytest.importorskip("sklearn")
    import recurra as rc

    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        s, g = f"s{i:02d}", ("b" if i >= 12 else "a")
        for w in range(4):
            eff = 1.5 if (g == "b" and w >= 2) else 0.0
            rows.append({"subject": s, "group": g, "window": w,
                         "DET": rng.normal() + eff, "LAM": rng.normal(),
                         "L": rng.normal() - eff})
    table = pd.DataFrame(rows)
    labels = table.groupby("subject").group.first().to_dict()
    out = rc.classify_timecourse(table, labels, time="window", families="rqa",
                                 n_repeats=2, seed=0)
    assert list(out.window) == [0, 1, 2, 3]
    assert (out[["n_a", "n_b"]].to_numpy() == 12).all()
    assert out[out.window >= 2].auc.mean() > out[out.window < 2].auc.mean()


def test_classical_family_finds_coupling_indices_output():
    """select_features('classical') must see what coupling_indices writes."""
    from recurra.classify.features import select_features

    table = pd.DataFrame({"subject": ["a", "b"], "mvl": [0.1, 0.2],
                          "mi_tort": [0.01, 0.02], "mod_contrast": [0.3, 0.4],
                          "preferred_phase": [0.5, -0.5], "DET": [0.8, 0.9]})
    cols = select_features(table, "classical")
    assert set(cols) == {"mvl", "mi_tort", "mod_contrast"}


@pytest.mark.requires("sklearn")
def test_degenerate_test_folds_are_flagged_not_scored(monkeypatch):
    """A single-class test fold gets NaN scores and a flag, and no sklearn
    warning; its predictions still count in the repeat-level scores."""
    import warnings

    import sklearn.model_selection as sms

    class TwoFolds:
        def __init__(self, n_splits=2, shuffle=True, random_state=0):
            pass

        def split(self, X, y, groups):
            # fold 1 tests one patient with three controls (both classes,
            # scored); fold 2 tests controls only (degenerate, flagged).
            # Both trains hold both classes, so neither fold is skipped.
            yield np.array([3, 4, 5, 6, 7, 8]), np.array([0, 1, 2, 9])
            yield np.array([0, 1, 2, 6, 7, 8, 9]), np.array([3, 4, 5])

    monkeypatch.setattr(sms, "StratifiedGroupKFold", TwoFolds)
    tab = pd.DataFrame({"subject": [f"s{i}" for i in range(10)],
                        "DET": np.linspace(0, 1, 10), "LAM": np.linspace(1, 0, 10)})
    labels = {f"s{i}": ("a" if i < 8 else "b") for i in range(10)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.filterwarnings("error",
                                message="y_pred contains classes not in y_true")
        rep = rc.classify_groups(tab, labels, families=["DET", "LAM"],
                                 n_repeats=2, seed=0)
    pf = rep.per_fold
    assert pf.degenerate_test.sum() == 2                 # fold B of each repeat
    assert pf.loc[pf.degenerate_test, "auc"].isna().all()
    assert np.isfinite(pf.loc[~pf.degenerate_test, "balanced_accuracy"]).all()
    assert len(rep.scores) == 2 and np.isfinite(rep.scores["auc"]).all()


@pytest.mark.requires("sklearn")
def test_permutation_p_is_independent_of_n_jobs():
    """The shuffles are seeded by position, so p must not depend on workers."""
    rng = np.random.default_rng(0)
    tab = pd.DataFrame({"subject": [f"s{i}" for i in range(24)],
                        "DET": rng.normal(size=24), "LAM": rng.normal(size=24)})
    labels = {f"s{i}": ("a" if i < 14 else "b") for i in range(24)}
    reps = [rc.classify_groups(tab, labels, families=["DET", "LAM"],
                               n_repeats=2, permutations=12, seed=3, n_jobs=j)
            for j in (1, 2)]
    assert reps[0].permutation["p_value"] == reps[1].permutation["p_value"]
    assert np.allclose(np.sort(reps[0].permutation["null"]),
                       np.sort(reps[1].permutation["null"]), equal_nan=True)


@pytest.mark.requires("sklearn")
def test_classify_timecourse_is_independent_of_n_jobs():
    rng = np.random.default_rng(1)
    rows = []
    for w in range(4):
        for i in range(24):
            rows.append({"subject": f"s{i}", "window": w,
                         "DET": rng.normal(), "LAM": rng.normal()})
    tab = pd.DataFrame(rows)
    labels = {f"s{i}": ("a" if i < 14 else "b") for i in range(24)}
    a = rc.classify_timecourse(tab, labels, families=["DET", "LAM"],
                               n_repeats=2, seed=5, n_jobs=1)
    b = rc.classify_timecourse(tab, labels, families=["DET", "LAM"],
                               n_repeats=2, seed=5, n_jobs=2)
    pd.testing.assert_frame_equal(a, b)

