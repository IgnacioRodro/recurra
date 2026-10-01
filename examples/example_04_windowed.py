#!/usr/bin/env python3
"""recurra -- windowed recurrence with overlap, across modes and subjects.

Run:      python examples/example_04_windowed.py
Writes:   outputs/   CSV tables, figures, and a plain-text run report.

One 60-second record becomes ten overlapping recurrence plots, each with its
own threshold, metrics and figure. The same call covers RP, CRP and JRP, every
threshold mode, every state-space route, and a corpus of subjects.
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

report = rc.RunReport("windowed_demo",
                      title="recurra: windowed recurrence with overlap")
report.parameter(fs=500.0, duration_s=60.0, bands="theta 4-8 Hz, gamma 50-70 Hz",
                 windows=10, overlap=0.5, target_rr=0.05, theiler=1, seed=17)


def banner(text: str) -> None:
    print(f"\n{SEP}\n{text}\n{SEP}")


def show(df, cols=None) -> str:
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    return d.to_string(index=False)


def build(modality, *, alpha=0.9, seed=17, duration=60.0, n_points=9000,
          nonstationarity=None, **kw):
    sig = rc.generate_cfc(modality=modality, duration=duration, fs=500.0, alpha=alpha,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=seed,
                          nonstationarity=nonstationarity, **kw)
    rec = rc.analytic(rc.filterbank(sig.to_recording(f"{modality}{seed}"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    return rec, sm.pac_space(rec, smooth=8.0).subsample(max_points=n_points)


# ================================================================== 1
banner("1. HOW TO DECLARE WINDOWS")
report.section("Window specification")

specs = {
    "10 windows, 50% overlap": rc.WindowSpec(n_windows=10, overlap=0.5),
    "10 windows, no overlap": rc.WindowSpec(n_windows=10, overlap=0.0),
    "4 s windows, 50% overlap": rc.WindowSpec(size=4.0, overlap=0.5),
    "4 s windows, 0.5 s step": rc.WindowSpec(size=4.0, step=0.5),
    "2000-sample windows": rc.WindowSpec(size=2000, step=1000, unit="samples"),
}
rows = []
for label, spec in specs.items():
    windows, caveats = spec.resolve(30000, 500.0)
    rows.append({"spec": label, "n_windows": len(windows),
                 "window_samples": windows[0].n_samples,
                 "window_seconds": round(windows[0].n_samples / 500.0, 3),
                 "step_samples": (windows[1].start - windows[0].start
                                  if len(windows) > 1 else 0),
                 "spans_record": windows[-1].stop == 30000,
                 "caveats": " | ".join(caveats)})
df_specs = pd.DataFrame(rows)
print(show(df_specs, ["spec", "n_windows", "window_samples", "window_seconds",
                      "step_samples", "spans_record"]))
report.export(df_specs, "window_specifications",
              "the same 60 s record cut five different ways")
report.note("asking for n windows gives exactly n, spanning the record")


# ================================================================== 2
banner("2. TEN OVERLAPPING RECURRENCE PLOTS FROM ONE RECORD")
report.section("Windowed recurrence plots")

rec_pac, ss_pac = build("pac", alpha=0.9)
spec = rc.WindowSpec(n_windows=10, overlap=0.5)
print(spec.summary(ss_pac.n_points, ss_pac.fs))

wr = rc.windowed_recurrence(ss_pac, spec, target_rr=0.05, theiler=1, rng=0)
print()
print(wr.summary())
print()
print(show(wr.metrics().round(5),
           ["window", "t_start_s", "t_stop_s", "n_rows", "epsilon",
            "target_rr", "recurrence_rate"]))
report.export(wr.metrics(), "windowed_metrics_pac",
              "one row per window: geometry, threshold and recurrence rate")
report.export(wr.window_frame(), "window_grid", "the resolved window grid")
report.result("windows produced", len(wr))


# ================================================================== 3
banner("3. THRESHOLD SCOPE CHANGES WHAT THE WINDOWS CAN SAY  [DD-40]")
report.section("Threshold scope")

scope_results = {}
rows = []
for scope in ("per_window", "global"):
    w = rc.windowed_recurrence(ss_pac, spec, scope=scope, target_rr=0.05,
                               theiler=1, rng=0)
    scope_results[scope] = w
    m = w.metrics()
    rows.append({"scope": scope,
                 "epsilon_mean": m.epsilon.mean(), "epsilon_std": m.epsilon.std(),
                 "rr_mean": m.recurrence_rate.mean(),
                 "rr_std": m.recurrence_rate.std(),
                 "rr_min": m.recurrence_rate.min(), "rr_max": m.recurrence_rate.max()})
df_scope = pd.DataFrame(rows)
print(show(df_scope.round(5)))
print("\n  per_window : epsilon adapts, the rate is pinned. Structure comparable,")
print("               density carries no information by construction.")
print("  global     : one epsilon for all. The rate becomes a time series, and")
print("               a window whose amplitude drifts really does come out sparser.")
report.export(df_scope, "threshold_scope_comparison",
              "per-window vs global threshold: which quantity becomes informative")


# ================================================================== 4
banner("4. NON-STATIONARY COUPLING SEEN THROUGH THE WINDOWS")
report.section("Non-stationary coupling")

_, ss_transient = build("pac", alpha=0.95, seed=31, nonstationarity="transient")
_, ss_ramp = build("pac", alpha=0.95, seed=32, nonstationarity="ramp")
_, ss_flat = build("pac", alpha=0.0, seed=33)

fine = rc.WindowSpec(size=4.0, overlap=0.75)
nonstationary = {}
for name, space in [("transient", ss_transient), ("ramp", ss_ramp),
                    ("no coupling", ss_flat)]:
    nonstationary[name] = rc.windowed_joint_recurrence(
        [space.split(["phase_theta"]), space.split(["amp_gamma"])],
        fine, target_rr=0.10, theiler=1, rng=0)

rows = []
for name, w in nonstationary.items():
    m = w.metrics()
    rows.append({"signal": name, "n_windows": len(w),
                 "independence_ratio_mean": m.independence_ratio.mean(),
                 "independence_ratio_min": m.independence_ratio.min(),
                 "independence_ratio_max": m.independence_ratio.max()})
df_ns = pd.DataFrame(rows)
print(show(df_ns.round(4)))
print("\n  The joint independence ratio is 1.0 when the two subsystems recur")
print("  independently. A transient coupling shows up as a bump in the series.")
report.export(pd.concat([w.metrics().assign(signal=n)
                         for n, w in nonstationary.items()]),
              "nonstationary_windowed_metrics",
              "windowed joint recurrence over non-stationary coupling")
report.export(df_ns, "nonstationary_summary", "independence ratio per signal type")


# ================================================================== 5
banner("5. EVERY MODE WINDOWS")
report.section("Coverage of modes")

rows = []
c_theta = sm.phase_circle(rec_pac, "phase_theta").subsample(max_points=9000)
c_gamma = sm.phase_circle(rec_pac, "phase_gamma").subsample(max_points=9000)
X_lorenz = rc.lorenz(n_points=9000, dt=0.01, transient=20.0)
rec_lorenz = rc.ingest(X_lorenz, fs=100.0, labels=["x", "y", "z"], subject="lorenz")

cases = [
    ("RP, target_rr", lambda: rc.windowed_recurrence(
        ss_pac, 8, target_rr=0.05, theiler=1, rng=0)),
    ("RP, FAN k=30", lambda: rc.windowed_recurrence(
        ss_pac, 8, threshold="fan", n_neighbors=30, theiler=1, rng=0)),
    ("RP, per_block", lambda: rc.windowed_recurrence(
        ss_pac, 8, threshold="per_block", target_rr=0.05, theiler=1, rng=0)),
    ("RP, fixed epsilon", lambda: rc.windowed_recurrence(
        ss_pac, 8, threshold=0.45, theiler=1, rng=0)),
    ("RP, chebyshev metric", lambda: rc.windowed_recurrence(
        ss_pac, 8, metric="chebyshev", target_rr=0.05, theiler=1, rng=0)),
    ("RP, hybrid space", lambda: rc.windowed_recurrence(
        sm.pac_space(rec_pac, embed_amplitude=3, tau=8).subsample(max_points=9000),
        8, target_rr=0.05, theiler=1, rng=0)),
    ("RP, Takens on Lorenz", lambda: rc.windowed_recurrence(
        sm.takens(rec_lorenz, "x", m=3, tau=16), 8, target_rr=0.05,
        theiler=1, rng=0)),
    ("CRP, theta vs gamma", lambda: rc.windowed_cross_recurrence(
        c_theta, c_gamma, 8, target_rr=0.05, rng=0)),
    ("JRP, phase and amplitude", lambda: rc.windowed_joint_recurrence(
        [ss_pac.split(["phase_theta"]), ss_pac.split(["amp_gamma"])],
        8, target_rr=0.10, theiler=1, rng=0)),
]
for label, factory in cases:
    w = factory()
    m = w.metrics()
    rows.append({"case": label, "kind": w.kind, "n_windows": len(w),
                 "window_samples": w.windows[0].n_samples, "dim": int(m.dim.iloc[0]),
                 "threshold_mode": m.threshold_mode.iloc[0],
                 "epsilon_mean": m.epsilon.mean(),
                 "rr_mean": m.recurrence_rate.mean(),
                 "rr_std": m.recurrence_rate.std()})
df_modes = pd.DataFrame(rows)
print(show(df_modes.round(5)))
report.export(df_modes, "windowed_mode_coverage",
              "windowing works across every threshold mode, metric, route and kind")
report.result("mode combinations exercised", len(cases))


# ================================================================== 6
banner("6. A CORPUS: SUBJECTS TIMES WINDOWS")
report.section("Multi-subject batch")

subjects = {}
truth = {}
for i in range(8):
    alpha = i / 7.0
    _, space = build("pac", alpha=alpha, seed=300 + i, duration=40.0, n_points=6000)
    subjects[f"sub{i:02d}"] = space
    truth[f"sub{i:02d}"] = alpha

batch = rc.batch_windowed_recurrence(subjects, rc.WindowSpec(n_windows=6, overlap=0.5),
                                     target_rr=0.05, theiler=1, rng=0)
print(batch.summary())

m = batch.metrics()
m["alpha_true"] = m["subject"].map(truth)
print(f"\n  tidy table: {m.shape[0]} rows x {m.shape[1]} columns "
      f"({len(subjects)} subjects x {len(batch['sub00'])} windows)")
print()
print(show(m.groupby("subject").agg(
    alpha_true=("alpha_true", "first"),
    epsilon_mean=("epsilon", "mean"),
    epsilon_std=("epsilon", "std"),
    rr_mean=("recurrence_rate", "mean")).round(5).reset_index()))

report.export(m, "corpus_windowed_metrics",
              "8 subjects x 6 windows, one tidy row per window")
report.export(batch.describe(), "corpus_windowed_summary", "one row per subject")
report.result("matrices computed", batch.n_matrices)
report.note("matrices are built on demand and not retained: a corpus of hundreds "
            "of subjects costs one window of memory at a time")

# Joint recurrence over the same corpus, to see coupling per subject and window
batch_jrp = rc.batch_windowed_recurrence(
    {k: [v.split(["phase_theta"]), v.split(["amp_gamma"])] for k, v in subjects.items()},
    rc.WindowSpec(n_windows=6, overlap=0.5), kind="jrp",
    target_rr=0.10, theiler=1, rng=0)
mj = batch_jrp.metrics()
mj["alpha_true"] = mj["subject"].map(truth)
agg = mj.groupby("subject").agg(alpha_true=("alpha_true", "first"),
                                independence_ratio=("independence_ratio", "mean"))
print("\n  Joint independence ratio per subject (windowed, averaged):")
print(show(agg.round(4).reset_index()))
from scipy.stats import spearmanr

rho = spearmanr(agg.alpha_true, agg.independence_ratio).statistic
print(f"\n  Spearman(alpha, independence ratio) = {rho:+.3f}")
report.export(mj, "corpus_windowed_joint",
              "windowed joint recurrence across the corpus")
report.result("Spearman(alpha, independence ratio)", f"{rho:+.3f}")


# ================================================================== 7
banner("7. SINGLE FIGURES")
report.section("Figures")

report.save(rc.plot_window_coverage(wr), "fig_window_coverage.png",
            "how the ten windows tile the record")
report.save(rc.plot_window_series(scope_results["global"], "recurrence_rate"),
            "fig_window_rate_global.png",
            "recurrence rate across windows, global threshold")
report.save(rc.plot_window_series(scope_results["per_window"], "epsilon"),
            "fig_window_epsilon_per_window.png",
            "epsilon across windows, per-window threshold")
report.save(rc.plot_window_panel(wr, columns=("epsilon", "recurrence_rate")),
            "fig_window_panel_pac.png", "epsilon and rate stacked")
report.save(rc.plot_window_series(nonstationary["transient"], "independence_ratio"),
            "fig_window_independence_transient.png",
            "joint independence ratio over transient coupling")

wr_keep = rc.windowed_recurrence(ss_pac, spec, target_rr=0.05, theiler=1,
                                 rng=0, keep=True)
for i in (0, 4, 9):
    report.save(rc.plot_recurrence(wr_keep[i]),
                f"fig_recurrence_window_{i:02d}.png",
                f"recurrence plot of window {i} alone")

for a in [x.path for x in report.artifacts if x.kind.startswith("figure")][-8:]:
    print(f"  {a}")


# ================================================================== 8
banner("8. COMPARISON MODE")

report.save(compare.compare_recurrence(
    {w.label: wr_keep[i] for i, w in enumerate(wr_keep.windows)},
    ncols=5, max_size=500, title="Ten overlapping windows of one 60 s record"),
    "fig_compare_windows_pac.png", "all ten windows side by side", kind="comparison")

report.save(compare.compare_window_series(
    nonstationary, column="independence_ratio",
    title="Joint independence ratio: transient vs ramp vs no coupling"),
    "fig_compare_nonstationary.png",
    "windowed coupling series across signal types", kind="comparison")

report.save(compare.compare_window_series(
    {k: batch[k] for k in ["sub00", "sub03", "sub07"]}, column="epsilon",
    title="Epsilon across windows, three subjects"),
    "fig_compare_subjects_epsilon.png", "per-subject window series",
    kind="comparison")

report.save(compare.compare_window_series(
    scope_results, column="recurrence_rate",
    title="Recurrence rate under per-window and global thresholds"),
    "fig_compare_scope.png", "the effect of threshold scope", kind="comparison")

for a in [x.path for x in report.artifacts if x.kind.startswith("figure")][-4:]:
    print(f"  {a}")


# ================================================================== 9
banner("9. PROVENANCE AND RUN REPORT")
report.section("Provenance")

print(wr.provenance_frame()[["step", "params"]].to_string(index=False,
                                                          max_colwidth=62))
report.export(wr.provenance_frame(), "windowed_provenance",
              "full history behind the windowed analysis")
for step in wr.provenance:
    for caveat in step.caveats:
        report.warn(f"{step.name}: {caveat}")

path = report.write()
print(f"\n  Run report: {path}")
print(f"\n{SEP}\nDone. See outputs/\n{SEP}")
