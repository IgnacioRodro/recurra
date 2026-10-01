"""Classifying two groups, evaluated so the number can be believed.

Design note (DD-84) -- everything that learns goes inside the fold.

Scaling, imputation and any feature selection are fitted on the training part
of each fold and applied to the held-out part. Fitting them on all the data
first and then cross-validating is the commonest way to produce an AUC that
does not survive new subjects, and it is invisible in the output.

Design note (DD-85) -- report imbalanced-aware metrics, and a baseline.

Plain accuracy on 30 controls against 10 patients is 0.75 for a model that
always says control. Balanced accuracy, the AUC and the F1 of the minority
class are reported alongside it, together with the **majority-class baseline**,
so a number can be compared against doing nothing. The classifier is fitted
with balanced class weights by default.

Design note (DD-86) -- one split is an anecdote.

With forty subjects a single five-fold split has eight per test fold, and the
AUC swings by more than any effect this project is looking for. The whole
cross-validation is therefore repeated with different partitions and the mean
and spread across repeats are reported. A result whose spread crosses the
baseline is not a result.

Design note (DD-87) -- significance comes from permutation, not from a table.

The features are many, correlated, and derived from overlapping windows; no
closed-form test applies. The labels are shuffled **at the subject level**, the
entire repeated cross-validation is rebuilt, and the p-value is the fraction of
shuffles reaching the observed score, with the conventional +1 correction so it
can never be reported as zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import SeedLike, resolve_rng
from ..exceptions import BackendUnavailable, ParameterError
from ..parallel.runner import parallel_map
from .features import align_labels, build_features

MODELS = ("logistic", "forest", "svm")
SCORES = ("auc", "balanced_accuracy", "accuracy", "sensitivity", "specificity",
          "precision", "f1", "mcc")


def _sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise BackendUnavailable("scikit-learn", "ml") from exc
    return True


def _make_model(name: str, seed: int):
    _sklearn()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if name == "logistic":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                 random_state=seed)
    elif name == "forest":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=seed, n_jobs=1)
    elif name == "svm":
        clf = SVC(kernel="rbf", probability=True, class_weight="balanced",
                  random_state=seed)
    else:
        raise ParameterError(f"model must be one of {MODELS}, got {name!r}")
    # Imputation and scaling are fitted per fold, never on all the data [DD-84].
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("clf", clf)])


def _scores(y_true: np.ndarray, y_pred: np.ndarray,
            y_prob: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    out: dict[str, float] = {}
    try:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out["auc"] = np.nan
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else np.nan
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
    out["precision"] = float(tp / (tp + fp)) if (tp + fp) else np.nan
    return out


@dataclass
class ClassificationReport:
    """Cross-validated performance, its spread, and what it means."""

    scores: pd.DataFrame                       # one row per repeat
    per_fold: pd.DataFrame                     # one row per fold
    importances: pd.DataFrame | None = None
    baseline: dict[str, float] = field(default_factory=dict)
    permutation: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def auc(self) -> float:
        return float(self.scores["auc"].mean())

    def to_frame(self) -> pd.DataFrame:
        row = {f"{k}_mean": float(self.scores[k].mean()) for k in SCORES
               if k in self.scores}
        row.update({f"{k}_sd": float(self.scores[k].std(ddof=1)) for k in SCORES
                    if k in self.scores})
        row.update({f"baseline_{k}": v for k, v in self.baseline.items()})
        row.update({f"n_{k}": v for k, v in self.class_counts.items()})
        if self.permutation:
            row["permutation_p"] = self.permutation.get("p_value")
            row["permutation_n"] = self.permutation.get("n_permutations")
            row["permutation_null_mean"] = self.permutation.get("null_mean")
        row.update({k: v for k, v in self.settings.items()
                    if isinstance(v, (int, float, str, bool))})
        return pd.DataFrame([row])

    def summary(self) -> str:
        s = self.scores
        lines = [
            f"ClassificationReport  {self.settings.get('model', '?')} on "
            f"{self.settings.get('n_features', '?')} features, "
            f"{self.settings.get('level', '?')} level",
            "  groups      : " + ", ".join(f"{k}={v}"
                                            for k, v in self.class_counts.items()),
            f"  scheme      : {self.settings.get('n_splits')}-fold x "
            f"{self.settings.get('n_repeats')} repeats, grouped by subject",
        ]
        for k in SCORES:
            if k in s:
                lines.append(f"  {k:<18s}: {s[k].mean():.3f} "
                             f"+- {s[k].std(ddof=1):.3f}")
        if self.baseline:
            lines.append("  baseline (majority class): "
                         + ", ".join(f"{k} {v:.3f}"
                                     for k, v in self.baseline.items()))
        if self.permutation:
            p = self.permutation
            lines.append(f"  permutation : p = {p['p_value']:.4f} "
                         f"({p['n_permutations']} shuffles, null AUC "
                         f"{p['null_mean']:.3f} +- {p['null_sd']:.3f})")
        for w in self.settings.get("warnings", []):
            lines.append(f"  WARNING     : {w}")
        return "\n".join(lines)


def _run_cv(X, y, groups, *, model, n_splits, n_repeats, seed,
            collect_importances=False):
    from sklearn.model_selection import StratifiedGroupKFold

    fold_rows, repeat_rows, importances = [], [], []
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=seed + repeat)
        y_true_all, y_pred_all, y_prob_all = [], [], []
        for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
            if len(np.unique(y[tr])) < 2:
                continue
            pipe = _make_model(model, seed + repeat)
            pipe.fit(X[tr], y[tr])
            prob = pipe.predict_proba(X[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            if len(np.unique(y[te])) < 2:
                # A test fold with a single class: its per-fold scores are
                # undefined -- sklearn's balanced accuracy would warn
                # "y_pred contains classes not in y_true" and return a
                # degenerate number, and the AUC does not exist. The
                # predictions themselves are valid and still enter the
                # repeat-level concatenation below, which always holds both
                # classes; only the per-fold row is flagged instead of scored.
                fold_rows.append({"repeat": repeat, "fold": fold,
                                  "n_train": len(tr), "n_test": len(te),
                                  "degenerate_test": True,
                                  **{k: np.nan for k in SCORES}})
            else:
                fold_rows.append({"repeat": repeat, "fold": fold,
                                  "n_train": len(tr), "n_test": len(te),
                                  "degenerate_test": False,
                                  **_scores(y[te], pred, prob)})
            y_true_all.append(y[te]); y_pred_all.append(pred); y_prob_all.append(prob)
            if collect_importances:
                clf = pipe.named_steps["clf"]
                if hasattr(clf, "coef_"):
                    importances.append(np.abs(clf.coef_.ravel()))
                elif hasattr(clf, "feature_importances_"):
                    importances.append(clf.feature_importances_)
        if y_true_all and np.unique(np.concatenate(y_true_all)).size == 2:
            # A repeat whose usable folds only ever tested one class -- possible
            # when the other class is so small that every fold holding it was
            # skipped by the train-side guard -- has no score either.
            repeat_rows.append({"repeat": repeat,
                                **_scores(np.concatenate(y_true_all),
                                          np.concatenate(y_pred_all),
                                          np.concatenate(y_prob_all))})
    return (pd.DataFrame(repeat_rows), pd.DataFrame(fold_rows),
            np.mean(importances, axis=0) if importances else None)


def _one_label_permutation(k, *, X, y, groups, subj, subj_label, model,
                           n_splits, base_seed):
    """One shuffle of the subject labels and one rebuilt cross-validation.

    Module level and picklable; seeded by its position ``k``, so the null is
    identical whatever ``n_jobs`` runs it [DD-81, DD-87].
    """
    perm_rng = np.random.default_rng(base_seed + 10_000 + k)
    shuffled = perm_rng.permutation([subj_label[g] for g in subj])
    mapping = dict(zip(subj, shuffled, strict=False))
    y_perm = np.asarray([mapping[g] for g in groups])
    s, _, _ = _run_cv(X, y_perm, groups, model=model, n_splits=n_splits,
                      n_repeats=1, seed=base_seed + k)
    return float(s["auc"].mean()) if not s.empty else np.nan


def classify_groups(table, labels, *, families="both", level: str = "subject",
                    aggregate: str = "mean", model: str = "logistic",
                    n_splits: int = 5, n_repeats: int = 10,
                    permutations: int = 0, subject: str = "subject",
                    seed: SeedLike = 0, positive=None,
                    n_jobs: int | None = 1) -> ClassificationReport:
    """Separate two groups from a metrics table, evaluated honestly.

    Parameters
    ----------
    table
        A metrics table, from ``metrics(rqa=True, dynamics=True)`` or a batch.
        One row per subject and window, or one row per subject for a
        whole-recording analysis.
    labels
        Mapping from subject to group, or one label per row.
    families
        ``"rqa"``, ``"dynamics"``, ``"classical"``, ``"both"``, ``"all"``, or an
        explicit list of columns.
    level
        ``"subject"`` aggregates the windows of a subject into one row;
        ``"window"`` keeps them, and folds still split by subject [DD-83].
    permutations
        Shuffles for the significance test. Zero skips it [DD-87].
    n_jobs
        Workers for the permutation loop [DD-109]. The default 1 keeps it
        serial, which is right when the caller already parallelises across
        channels; the shuffles are seeded by position, so the p-value is
        identical at any worker count.

    Every transformation that learns is fitted inside the fold [DD-84], the
    scheme is repeated [DD-86], and imbalanced-aware scores are reported beside
    a majority-class baseline [DD-85].
    """
    _sklearn()
    X_df, groups = build_features(table, families=families, level=level,
                                  aggregate=aggregate, subject=subject)
    y_raw = align_labels(groups, labels)
    classes = sorted(set(map(str, y_raw)))
    if len(classes) != 2:
        raise ParameterError(
            f"classification needs exactly two groups, found {classes}")
    pos = str(positive) if positive is not None else classes[-1]
    if pos not in classes:
        raise ParameterError(f"positive={pos!r} is not one of {classes}")
    y = (np.asarray(list(map(str, y_raw))) == pos).astype(int)

    X = X_df.to_numpy(dtype=float)
    n_subjects = len(np.unique(groups))
    per_class = {c: int(np.sum(np.asarray(list(map(str, y_raw))) == c))
                 for c in classes}

    warnings_out = []
    smallest = min(len(np.unique(groups[y == 0])), len(np.unique(groups[y == 1])))
    if smallest < n_splits:
        n_splits = max(2, smallest)
        warnings_out.append(
            f"the smaller group has {smallest} subjects, so the scheme was "
            f"reduced to {n_splits} folds")
    if n_subjects < 20:
        warnings_out.append(
            f"{n_subjects} subjects is a small sample; read the spread across "
            "repeats, not the mean alone")
    if X.shape[1] > n_subjects:
        warnings_out.append(
            f"{X.shape[1]} features for {n_subjects} subjects. A fitted "
            "combination spends its degrees of freedom on noise in this regime "
            "[DD-75]; consider a single family or a summary aggregate")

    rng = resolve_rng(seed)
    base_seed = int(rng.integers(0, 2**31 - 1))
    scores, per_fold, imp = _run_cv(X, y, groups, model=model, n_splits=n_splits,
                                    n_repeats=n_repeats, seed=base_seed,
                                    collect_importances=True)
    if scores.empty:
        raise ParameterError("no fold produced both classes; the sample is too small")

    majority = max(per_class.values()) / sum(per_class.values())
    baseline = {"accuracy": majority, "balanced_accuracy": 0.5, "auc": 0.5}

    permutation: dict[str, Any] = {}
    if permutations:
        observed = float(scores["auc"].mean())
        # Shuffle at the subject level: a subject's label, not a row's [DD-87],
        # in parallel when asked [DD-109].
        subj = np.unique(groups)
        subj_label = {g: y[groups == g][0] for g in subj}
        null = parallel_map(_one_label_permutation, list(range(permutations)),
                            n_jobs=n_jobs, backend="process", pass_rng=False,
                            X=X, y=y, groups=groups, subj=subj,
                            subj_label=subj_label, model=model,
                            n_splits=n_splits, base_seed=base_seed)
        null = np.asarray([v for v in null if np.isfinite(v)], dtype=float)
        permutation = {
            "observed": observed, "n_permutations": int(null.size),
            "null_mean": float(np.nanmean(null)) if null.size else np.nan,
            "null_sd": float(np.nanstd(null, ddof=1)) if null.size > 1 else np.nan,
            "p_value": float((1 + np.sum(null >= observed)) / (null.size + 1)),
            "null": null,
        }

    importances = None
    if imp is not None:
        importances = (pd.DataFrame({"feature": X_df.columns, "importance": imp})
                       .sort_values("importance", ascending=False)
                       .reset_index(drop=True))

    return ClassificationReport(
        scores=scores, per_fold=per_fold, importances=importances,
        baseline=baseline, permutation=permutation, class_counts=per_class,
        settings={"model": model, "level": level, "aggregate": aggregate,
                  "families": str(families), "n_features": X.shape[1],
                  "n_subjects": n_subjects, "n_rows": X.shape[0],
                  "n_splits": n_splits, "n_repeats": n_repeats,
                  "positive_class": pos, "warnings": warnings_out},
    )


def _one_timecourse_point(task, *, labels, families, model, n_splits, reps,
                          sd, min_per_group, subject, positive, time):
    """One aligned window's classification; module level and picklable."""
    t, part = task
    try:
        rep = classify_groups(part, labels, families=families, level="subject",
                              model=model, n_splits=n_splits, n_repeats=reps,
                              subject=subject, seed=sd, positive=positive)
    except ParameterError:
        return None
    counts = sorted(rep.class_counts.items())
    n_a, n_b = counts[0][1], counts[1][1]
    if min(n_a, n_b) < min_per_group:
        return None
    return {time: t, "auc": float(rep.scores["auc"].mean()),
            "auc_sd": float(rep.scores["auc"].std(ddof=1)),
            "n_a": n_a, "n_b": n_b,
            "n_features": rep.settings.get("n_features")}


def _one_curve_permutation(k, *, frame, subj, subj_label, groups_all, time,
                           families, model, n_splits, min_per_group, subject,
                           positive, base):
    """One label shuffle and one whole rebuilt curve, for the peak null."""
    perm_rng = np.random.default_rng(base + 20_000 + k)
    shuffled = perm_rng.permutation([subj_label[g] for g in subj])
    mapping = dict(zip(subj, shuffled, strict=False))
    tasks = [(t, frame[frame[time] == t]) for t in sorted(frame[time].unique())]
    rows = [r for r in (
        _one_timecourse_point(task, labels=mapping, families=families,
                              model=model, n_splits=n_splits, reps=1,
                              sd=base + k, min_per_group=min_per_group,
                              subject=subject, positive=positive, time=time)
        for task in tasks) if r is not None]
    if not rows:
        return np.nan
    return float(np.nanmax(np.abs(np.asarray([r["auc"] for r in rows]) - 0.5)))


def classify_timecourse(table, labels, *, time: str = "window",
                        keep: str | None = None, families="both",
                        model: str = "logistic", n_splits: int = 5,
                        n_repeats: int = 3, min_per_group: int = 5,
                        subject: str = "subject", seed: SeedLike = 0,
                        positive=None, permutations: int = 0,
                        n_jobs: int | None = 1) -> pd.DataFrame:
    """One honest classification per aligned window: does separation evolve?

    Design note (DD-114) -- the aggregated comparison collapses each subject
    to one number and cannot say *when* the groups differ. Here every window
    index gets its own :func:`classify_groups` over the subjects present in
    that window (one row per subject per window; rows excluded by ``keep``
    stay out), with the same in-fold discipline [DD-83, DD-84]. The result is
    a curve of cross-validated AUC over the recording.

    The curve is descriptive: its windows are as correlated as the data
    behind them, and the peak of T looks must not be read against an
    uncorrected threshold. With ``permutations`` > 0, subject labels are
    shuffled and the **whole curve** recomputed each time; ``p_peak`` in the
    frame's ``attrs`` is the probability of a peak this large anywhere on the
    curve under the null -- corrected for taking the best window. Expensive:
    permutations x windows x folds fits.

    Returns a frame with one row per window: ``auc`` (mean over repeats),
    ``auc_sd``, ``n_a``, ``n_b``, ``n_features``. Windows with fewer than
    ``min_per_group`` subjects in either group are skipped.
    """
    _sklearn()
    frame = table[table[keep].astype(bool)] if keep else table
    rng = resolve_rng(seed)
    base = int(rng.integers(0, 2**31 - 1))

    # The observed curve: every aligned window is an independent task [DD-109].
    tasks = [(t, frame[frame[time] == t]) for t in sorted(frame[time].unique())]
    rows = parallel_map(_one_timecourse_point, tasks, n_jobs=n_jobs,
                        backend="process", pass_rng=False, labels=labels,
                        families=families, model=model, n_splits=n_splits,
                        reps=n_repeats, sd=base, min_per_group=min_per_group,
                        subject=subject, positive=positive, time=time)
    out = pd.DataFrame([r for r in rows if r is not None])
    if permutations and not out.empty:
        groups_all = frame[subject].to_numpy()
        y_raw = align_labels(groups_all, labels)
        subj = np.unique(groups_all)
        subj_label = {g: str(y_raw[groups_all == g][0]) for g in subj}
        observed = float(np.nanmax(np.abs(out["auc"].to_numpy() - 0.5)))
        # Whole curves under shuffled labels, seeded by position [DD-81]: the
        # peak's null is a max-statistic over the curve [DD-114].
        peaks = parallel_map(_one_curve_permutation, list(range(permutations)),
                             n_jobs=n_jobs, backend="process", pass_rng=False,
                             frame=frame, subj=subj, subj_label=subj_label,
                             groups_all=groups_all, time=time,
                             families=families, model=model, n_splits=n_splits,
                             min_per_group=min_per_group, subject=subject,
                             positive=positive, base=base)
        hits = 1 + int(np.nansum(np.asarray(peaks, dtype=float)
                                 >= observed - 1e-12))
        out.attrs["p_peak"] = hits / (permutations + 1)
        out.attrs["auc_peak"] = 0.5 + observed if \
            out.auc.max() - 0.5 >= 0.5 - out.auc.min() else 0.5 - observed
    return out
