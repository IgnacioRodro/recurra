"""Turning a metrics table into a feature matrix, without leaking.

Design note (DD-83) -- windows of one subject are not independent samples.

A windowed metrics table has one row per subject and window. Treating those
rows as independent observations inflates every estimate: fifteen windows of
one subject share that subject's anatomy, electrode placement, alertness and
noise floor, and a fold that puts some of them in training and the rest in
testing is asking the model to recognise a subject, not a group.

Two honest levels are offered.

``level="subject"`` reduces the windows of a subject to one row -- their mean,
and optionally their spread and temporal trend -- and folds are drawn over
subjects. This is the default because it is the one that cannot leak.

``level="window"`` keeps the rows, and every fold splits **by subject**, so all
windows of a subject fall on the same side. That is more data for the model and
still no leakage, but the effective sample size remains the number of subjects
and the confidence intervals must be read accordingly.

The whole-recording case is the same code with a single window.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..exceptions import ParameterError

#: Column families, matched against the metrics table by name.
FAMILIES: dict[str, tuple[str, ...]] = {
    # Both spellings: coupling_indices writes lowercase (mvl, mi_tort,
    # mod_contrast); older tables and external ones may carry uppercase.
    # preferred_phase is circular and is not a linear feature.
    "classical": ("MVL", "MI", "PLV", "AAC", "aac_index", "modulation_index",
                  "mvl", "mi_tort", "mod_contrast", "plv", "aac"),
    "rqa": ("RR", "DET", "L", "Lmax", "DIV", "ENTR", "ENTR_norm", "LAM", "TT",
            "Vmax", "W", "Wmax", "ENTW", "RTE", "independence_ratio",
            "recurrence_rate"),
    "dynamics": ("higuchi_fd", "katz_fd", "petrosian_fd", "svd_entropy",
                 "permutation_entropy", "sample_entropy", "spectral_entropy",
                 "DFA_alpha", "hurst", "lempel_ziv", "hjorth_activity",
                 "hjorth_mobility", "hjorth_complexity", "D2", "lambda_1", "K2"),
}

#: Columns that describe the analysis rather than the signal, never features.
BOOKKEEPING = {
    "subject", "group", "window", "window_label", "kind", "scope", "seed",
    "kept", "reasons", "n_masks", "channel", "preferred_phase",
    "start_sample", "stop_sample", "n_samples", "t_start_s", "t_stop_s",
    "t_center_s", "n_rows", "n_cols", "dim", "theiler", "metric",
    "threshold_mode", "target_rr", "l_min", "v_min", "w_min", "denominator",
    "fs", "epsilon",
}

AGGREGATES = ("mean", "mean_std", "mean_std_trend")


def select_features(table: pd.DataFrame,
                    families: str | Sequence[str] = "both") -> list[str]:
    """Column names belonging to the requested families.

    ``"rqa"``, ``"dynamics"``, ``"classical"``, ``"both"`` (RQA and dynamics),
    ``"all"``, or an explicit list of column names. A dynamical column carries
    its block as a suffix, so matching is on the prefix.
    """
    if isinstance(families, str):
        wanted = {"both": ["rqa", "dynamics"], "all": list(FAMILIES)}.get(
            families, [families])
    else:
        explicit = [c for c in families if c in table.columns]
        missing = set(families) - set(explicit)
        if missing:
            raise ParameterError(f"columns not in the table: {sorted(missing)}")
        return explicit
    unknown = set(wanted) - set(FAMILIES)
    if unknown:
        raise ParameterError(
            f"unknown family {sorted(unknown)}; use one of "
            f"{list(FAMILIES)} or 'both'/'all'")

    prefixes = tuple(p for fam in wanted for p in FAMILIES[fam])
    out = []
    for col in table.columns:
        if col in BOOKKEEPING or not pd.api.types.is_numeric_dtype(table[col]):
            continue
        if col in prefixes or any(col.startswith(f"{p}_") for p in prefixes):
            out.append(col)
    if not out:
        raise ParameterError(
            f"no column of family {wanted} found. Did the metrics table include "
            "rqa=True or dynamics=True?")
    return out


def _trend(values: np.ndarray) -> float:
    """Slope of a metric across the windows of one subject, per window."""
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(y.size, dtype=float)[ok]
    return float(np.polyfit(x, y[ok], 1)[0])


def build_features(table: pd.DataFrame, *, families="both",
                   level: str = "subject", aggregate: str = "mean",
                   subject: str = "subject") -> tuple[pd.DataFrame, np.ndarray]:
    """Feature matrix and the subject each row belongs to [DD-83].

    Returns ``(features, groups)`` where ``groups`` names the subject of every
    row, which is what the fold splitter must respect.
    """
    if level not in ("subject", "window"):
        raise ParameterError("level must be 'subject' or 'window'")
    if aggregate not in AGGREGATES:
        raise ParameterError(f"aggregate must be one of {AGGREGATES}")
    if subject not in table.columns:
        raise ParameterError(f"no subject column {subject!r}")

    cols = select_features(table, families)
    if level == "window":
        return table[cols].copy(), table[subject].to_numpy()

    parts = [table.groupby(subject)[cols].mean().add_suffix("_mean")]
    if aggregate in ("mean_std", "mean_std_trend"):
        parts.append(table.groupby(subject)[cols].std().add_suffix("_std"))
    if aggregate == "mean_std_trend":
        parts.append(table.groupby(subject)[cols].agg(_trend).add_suffix("_trend"))
    feats = pd.concat(parts, axis=1)
    feats = feats.dropna(axis=1, how="all")
    return feats, feats.index.to_numpy()


def align_labels(groups: np.ndarray, labels) -> np.ndarray:
    """Turn a mapping or a sequence of labels into one label per row."""
    if isinstance(labels, Mapping):
        missing = sorted(set(groups) - set(labels))
        if missing:
            raise ParameterError(f"no label for subject(s) {missing[:5]}")
        return np.asarray([labels[g] for g in groups])
    labels = np.asarray(labels)
    if labels.size != groups.size:
        raise ParameterError(
            f"{labels.size} labels for {groups.size} rows; pass a mapping from "
            "subject to group if the table has several rows per subject")
    return labels
