#!/usr/bin/env python3
"""recurra -- comparability across subjects, and separating two groups.

Run:      python examples/example_05_comparability.py
Writes:   outputs/   CSV tables, figures, and a plain-text run report.

The question this answers: can recurrence structures built from different
subjects legitimately be compared, and if so, which metrics may be compared?
Sections 1 and 2 show the three ways comparability breaks silently; section 3
fixes them with a contract; sections 4 to 6 separate two groups window by
window and build the feature matrix a classifier would consume.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

import recurra as rc
from recurra import statespace as sm
from recurra.viz import compare

warnings.simplefilter("ignore", rc.RecurraWarning)
# run_all.py --out sets RECURRA_EXAMPLES_OUT; run alone, the default applies
rc.set_config(output_dir=os.environ.get("RECURRA_EXAMPLES_OUT", "outputs"))
SEP = "=" * 80

report = rc.RunReport("comparability_demo",
                      title="recurra: cross-subject comparability and group separation")
report.parameter(fs=500.0, bands="theta 4-8 Hz, gamma 50-70 Hz",
                 window="6 s, 50% overlap", target_rr=0.10, theiler=1,
                 n_subjects=20, groups="control (low alpha) vs study (high alpha)")


def banner(text: str) -> None:
    print(f"\n{SEP}\n{text}\n{SEP}")


def show(df, cols=None) -> str:
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    return d.to_string(index=False)


def pac_space(alpha, seed, duration):
    sig = rc.generate_cfc(modality="pac", duration=duration, fs=500.0, alpha=alpha,
                          preferred_phase=np.pi / 2, snr_db=15.0, seed=seed)
    rec = rc.analytic(rc.filterbank(sig.to_recording(f"s{seed}"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    return sm.pac_space(rec, smooth=8.0)


# ================================================================== 1
banner("1. THREE WAYS COMPARABILITY BREAKS SILENTLY  [DD-42]")
report.section("Failure modes")

unequal = {"sub00": pac_space(0.2, 10, 60.0),
           "sub01": pac_space(0.5, 11, 45.0),
           "sub02": pac_space(0.8, 12, 72.0)}

rows = []
print("  (a) auto embedding puts subjects in different phase spaces")
for name, space in unequal.items():
    rec_like = space
    auto = sm.takens(rc.ingest(np.asarray(space.coords)[:, 2:3], fs=500.0,
                               subject=name), None, m="auto", tau="auto", rng=0)
    print(f"      {name}: dim = {auto.dim}")
    rows.append({"failure": "auto embedding", "subject": name, "quantity": "dim",
                 "value": auto.dim})

print("\n  (b) n_windows resolves against each record's own length")
for name, space in unequal.items():
    w, _ = rc.WindowSpec(n_windows=6, overlap=0.5).resolve(space.n_points, space.fs)
    print(f"      {name}: {space.n_points:6d} samples -> window 0 covers "
          f"{w[0].t_start:5.2f}-{w[0].t_stop:6.2f} s "
          f"({w[0].n_samples / space.fs:5.2f} s)")
    rows.append({"failure": "n_windows", "subject": name,
                 "quantity": "window_seconds",
                 "value": round(w[0].n_samples / space.fs, 3)})

print("\n  (c) subsample(max_points) decimates each subject by a different factor")
for name, space in unequal.items():
    sub = space.subsample(max_points=5000)
    factor = space.n_points / sub.n_points
    print(f"      {name}: factor {factor:5.2f} -> effective fs "
          f"{space.fs / factor:6.1f} Hz")
    rows.append({"failure": "subsample", "subject": name,
                 "quantity": "effective_fs_hz", "value": round(space.fs / factor, 2)})

report.export(pd.DataFrame(rows), "comparability_failure_modes",
              "the same request producing different geometry per subject")
report.note("all three are reachable from the API and none announces itself")


# ================================================================== 2
banner("2. THE LIBRARY DETECTS IT")
report.section("Detection")

bad = rc.batch_windowed_recurrence(unequal, rc.WindowSpec(n_windows=6, overlap=0.5),
                                   target_rr=0.05, theiler=1, rng=0)
print(bad.report.summary())
report.export(bad.report.to_frame(), "comparability_report_bad",
              "axis-by-axis verdict for a non-comparable batch")
report.export(bad.report.metric_validity(), "metric_validity_bad",
              "which metric families remain valid despite the violations")
report.warn("n_windows on records of unequal length: window k is not the same "
            "stretch across subjects")


# ================================================================== 3
banner("3. THE FIX: A FIXED WINDOW GRID AND AN EXPLICIT CONTRACT  [DD-43]")
report.section("Contract")

SPEC = rc.WindowSpec(size=6.0, overlap=0.5, unit="seconds")
good = rc.batch_windowed_recurrence(unequal, SPEC, target_rr=0.05, theiler=1, rng=0)
print(good.report.summary())

m = good.metrics()
print("\n  Window 0 across subjects, now the same stretch of every record:")
print(show(m[m.window == 0].round(4),
           ["subject", "window", "t_start_s", "t_stop_s", "n_rows", "epsilon"]))
print(f"\n  Windows per subject (records differ in length, so counts differ): "
      f"{m.groupby('subject').size().to_dict()}")

contract = rc.contract_from(good["sub00"],
                            axes=["dim", "fs", "n_points", "metric", "theiler",
                                  "threshold_mode", "target_rr", "scaling",
                                  "window_samples", "window_step"])
print()
print(contract.summary())
print()
print(f"  contract satisfied: {good.check_comparability(contract).ok}")

report.export(contract.to_frame(), "comparability_contract",
              "the parameters held fixed across subjects")
report.export(good.report.to_frame(), "comparability_report_good",
              "axis-by-axis verdict after fixing the window grid")
report.export(good.report.metric_validity(), "metric_validity_good",
              "metric families licensed by the observed matching")


# ================================================================== 4
banner("4. TWO GROUPS, TWENTY SUBJECTS, UNEQUAL RECORD LENGTHS")
report.section("Corpus")

rng = np.random.default_rng(0)
spaces, groups, truth = {}, {}, {}
for i in range(20):
    grp = "control" if i < 10 else "study"
    alpha = rng.uniform(0.0, 0.25) if grp == "control" else rng.uniform(0.55, 0.85)
    duration = float(rng.choice([40.0, 45.0, 50.0]))
    name = f"sub{i:02d}"
    ss = pac_space(alpha, 1000 + i, duration)
    spaces[name] = [ss.split(["phase_theta"]), ss.split(["amp_gamma"])]
    groups[name] = grp
    truth[name] = alpha

batch = rc.batch_windowed_recurrence(spaces, SPEC, kind="jrp", target_rr=0.10,
                                     theiler=1, rng=0)
print(batch.summary())
print(f"\n  comparability verified : {batch.report.ok}")
print(f"  structures computed    : {batch.n_matrices}")
print(f"  record durations       : "
      f"{sorted({len(w) for w in batch.results.values()})} windows per subject")

metrics = batch.metrics()
metrics["group"] = metrics["subject"].map(groups)
metrics["alpha_true"] = metrics["subject"].map(truth)
report.export(metrics, "corpus_group_metrics",
              "20 subjects x windows, joint recurrence, with group labels")
report.result("structures computed", batch.n_matrices)
report.result("comparability verified", batch.report.ok)


# ================================================================== 5
banner("5. GROUP SEPARATION, WINDOW BY WINDOW  [DD-44]")
report.section("Group separation")

gc = batch.group_comparison(groups, values=["independence_ratio", "recurrence_rate"])
cols = ["window", "metric", "n_a", "n_b", "mean_a", "mean_b", "difference",
        "cohens_d", "auc", "p_value", "p_bonferroni"]
sub = gc[gc.metric == "independence_ratio"]
print("  Joint independence ratio, control (a) vs study (b):")
print(show(sub.round(4), cols).replace("independence_ratio", "indep_ratio"))

print("\n  Metrics ranked by how well they separate the groups:")
print(show(rc.rank_metrics(gc).round(4)))

n_sig = int((sub["p_bonferroni"] < 0.05).sum())
print(f"\n  {n_sig} of {len(sub)} windows separate the groups after Bonferroni "
      f"correction; mean AUC {sub['auc'].mean():.3f}, "
      f"mean Cohen's d {sub['cohens_d'].mean():.2f}")

report.export(gc, "group_comparison",
              "per-window group statistics, effect sizes and corrected p-values")
report.export(rc.rank_metrics(gc), "metric_ranking",
              "metrics ordered by group separation")
report.result("windows significant after Bonferroni", f"{n_sig}/{len(sub)}")
report.result("mean AUC", f"{sub['auc'].mean():.3f}")


# ================================================================== 6
banner("6. FEATURE MATRIX FOR A CLASSIFIER")
report.section("Feature matrix")

X = batch.metrics_by_window("independence_ratio")
print(f"  subject-by-window matrix: {X.shape[0]} subjects x {X.shape[1]} windows")
print(f"  missing entries (shorter records): {int(X.isna().sum().sum())}")
print()
print(X.iloc[:6, :8].round(3).to_string())

subject_means = metrics.groupby("subject").agg(
    group=("group", "first"), alpha_true=("alpha_true", "first"),
    independence_ratio=("independence_ratio", "mean")).reset_index()
from scipy.stats import spearmanr

rho = spearmanr(subject_means.alpha_true, subject_means.independence_ratio).statistic
print(f"\n  Spearman(true coupling alpha, mean independence ratio) = {rho:+.3f}")
print("  -- a recurrence-based index tracking the classical coupling strength,")
print("     which is the parallel the project is looking for.")

report.export(X.reset_index(), "feature_matrix",
              "subject-by-window matrix, ready for a classifier")
report.export(subject_means, "subject_level_summary",
              "one row per subject: group, true alpha, mean recurrence index")
report.result("Spearman(alpha, recurrence index)", f"{rho:+.3f}")


# ================================================================== 7
banner("7. FIGURES")
report.section("Figures")

report.save(rc.plot_comparability(bad.report), "fig_comparability_bad.png",
            "axes and licensed metric families for a non-comparable batch")
report.save(rc.plot_comparability(batch.report), "fig_comparability_good.png",
            "the same after fixing the window grid")
report.save(rc.plot_group_comparison(sub, "independence_ratio"),
            "fig_group_comparison.png",
            "group means per window with AUC and significance")
report.save(rc.plot_feature_matrix(X, groups=groups,
                                   title="Independence ratio, subjects by windows"),
            "fig_feature_matrix.png", "the classifier's input, ordered by group")
report.save(compare.compare_window_series(
    {k: batch[k] for k in ["sub00", "sub02", "sub14", "sub18"]},
    column="independence_ratio",
    title="Window series: two controls and two study subjects"),
    "fig_compare_group_series.png", "per-subject window series across groups",
    kind="comparison")

for a in [x.path for x in report.artifacts if x.kind.startswith("figure")]:
    print(f"  {a}")

path = report.write()
print(f"\n  Run report: {path}")
print(f"\n{SEP}\nDone. See outputs/\n{SEP}")
