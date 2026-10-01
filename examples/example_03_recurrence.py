#!/usr/bin/env python3
"""recurra -- thresholds and recurrence structures (F4 and F5).

Run:      python examples/example_03_recurrence.py
Writes:   outputs/   CSV tables, figures, and a plain-text run report.

The run report (outputs/recurrence_demo_report.txt) is the summary of what
this execution did and produced.
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

report = rc.RunReport("recurrence_demo",
                      title="recurra F4 + F5: thresholds and recurrence structures")
report.parameter(fs=500.0, duration_s=40.0, bands="theta 4-8 Hz, gamma 50-70 Hz",
                 n_points_per_space=2500, default_target_rr=0.05, theiler=1, seed=17)


def banner(text: str) -> None:
    print(f"\n{SEP}\n{text}\n{SEP}")


def show(df, cols=None) -> str:
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    return d.to_string(index=False)


def build_space(modality, *, alpha=0.9, seed=17, n_points=2500, **kw):
    sig = rc.generate_cfc(modality=modality, duration=40.0, fs=500.0, alpha=alpha,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=seed, **kw)
    rec = rc.analytic(rc.filterbank(sig.to_recording(f"{modality}01"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    space = sm.pac_space(rec, smooth=8.0).subsample(max_points=n_points)
    return rec, space


# ================================================================== 1
banner("1. STATE SPACES UNDER TEST")
report.section("Build state spaces")

rec_pac, ss_pac = build_space("pac", alpha=0.9)
rec_null, ss_null = build_space("none", alpha=0.0, seed=21)
_, ss_weak = build_space("pac", alpha=0.3, seed=17)

X_lorenz = rc.lorenz(n_points=12000, dt=0.01, transient=20.0)
rec_lorenz = rc.ingest(X_lorenz, fs=100.0, labels=["x", "y", "z"], subject="lorenz")
ss_lorenz = sm.channels(rec_lorenz).subsample(max_points=2500)

spaces = {"PAC alpha=0.9": ss_pac, "PAC alpha=0.3": ss_weak,
          "uncoupled": ss_null, "Lorenz": ss_lorenz}
for name, s in spaces.items():
    print(f"  {name:16s} dim {s.dim}, {s.n_points} points, scaling {s.scaling}")
    report.note(f"{name}: dim {s.dim}, {s.n_points} points")


# ================================================================== 2
banner("2. THRESHOLD MODES  (F4)")
report.section("Threshold estimation")

configs = [
    ("target_rr 5% (quantile)", dict(mode="target_rr", target_rr=0.05)),
    ("target_rr 5% (bisect)", dict(mode="target_rr", target_rr=0.05, method="bisect")),
    ("target_rr 1%", dict(mode="target_rr", target_rr=0.01)),
    ("target_rr 15%", dict(mode="target_rr", target_rr=0.15)),
    ("percentile 0.05", dict(mode="percentile", percentile=0.05)),
    ("fan k=25", dict(mode="fan", n_neighbors=25)),
    ("std_fraction 0.1", dict(mode="std_fraction", factor=0.1)),
    ("maxdist_fraction 0.1", dict(mode="maxdist_fraction", factor=0.1)),
    ("per_block 5%", dict(mode="per_block", target_rr=0.05)),
    ("adaptive_dim 5%", dict(mode="adaptive_dim", target_rr=0.05)),
    ("fixed 0.40", dict(mode="fixed", value=0.40)),
]

thresholds, rows = {}, []
for label, kwargs in configs:
    th = rc.threshold(ss_pac, theiler=1, rng=0, **kwargs)
    thresholds[label] = th
    row = th.to_frame().iloc[0].to_dict()
    row["label"] = label
    rows.append(row)

df_thresholds = pd.DataFrame(rows)
print(show(df_thresholds.round(5),
           ["label", "mode", "epsilon", "target_rr", "achieved_rr", "pointwise",
            "per_block", "n_samples"]))
report.export(df_thresholds, "threshold_modes",
              "epsilon and achieved recurrence rate for every threshold mode")
report.result("modes evaluated", len(configs))

print("\n  Detail of the bisection mode:")
print(thresholds["target_rr 5% (bisect)"].summary())

for label in ("fan k=25", "adaptive_dim 5%"):
    warning = thresholds[label].diagnostics.get("warning")
    if warning:
        report.warn(f"{label}: {warning}")


# ================================================================== 3
banner("3. TARGET RATE IS ACTUALLY ACHIEVED")
report.section("Target-rate accuracy")

rows = []
for target in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]:
    for method in ("quantile", "bisect"):
        th = rc.threshold(ss_pac, mode="target_rr", target_rr=target,
                          method=method, theiler=1, rng=0)
        rm = rc.recurrence_plot(ss_pac, th, theiler=1)
        rows.append({"target_rr": target, "method": method,
                     "epsilon": th.scalar, "estimated_rr": th.achieved_rr,
                     "actual_rr": rm.recurrence_rate(),
                     "abs_error": abs(rm.recurrence_rate() - target)})
df_accuracy = pd.DataFrame(rows)
print(show(df_accuracy.round(5)))
worst = df_accuracy["abs_error"].max()
print(f"\n  Worst absolute deviation across all targets and methods: {worst:.5f}")
report.export(df_accuracy, "target_rate_accuracy",
              "requested vs achieved recurrence rate, both threshold methods")
report.result("worst absolute deviation from target", f"{worst:.5f}")


# ================================================================== 4
banner("4. STREAMING EQUALS DENSE, EXACTLY  (F5)")
report.section("Streaming equivalence and memory planning")

th = rc.threshold(ss_pac, target_rr=0.05, theiler=1, rng=0)
dense = rc.recurrence_plot(ss_pac, th, theiler=1, store="memory")

rows = []
for tile_size in (64, 128, 512, 4096):
    streamed = rc.recurrence_plot(ss_pac, th, theiler=1, store="none",
                                  tile_size=tile_size)
    rebuilt = np.zeros(dense.shape, dtype=np.uint8)
    for tile in streamed.tiles():
        rebuilt[tile.i0:tile.i1, tile.j0:tile.j1] = tile.data
    rows.append({"tile_size": tile_size,
                 "identical_to_dense": bool(np.array_equal(rebuilt, dense.matrix)),
                 "recurrence_rate": streamed.recurrence_rate(),
                 "rate_difference": abs(streamed.recurrence_rate()
                                        - dense.recurrence_rate())})
df_streaming = pd.DataFrame(rows)
print(show(df_streaming))
report.export(df_streaming, "streaming_equivalence",
              "tiled streaming reproduces the dense matrix bit for bit")
report.result("streaming identical to dense", bool(df_streaming.identical_to_dense.all()))

print("\n  Memory planning for signals of increasing length:")
rows = []
for n in (5_000, 50_000, 200_000, 1_000_000, 1_800_000):
    p = rc.recurrence.plan(n, n, 3, dtype="uint8", budget_gb=8.0, estimated_rr=0.05)
    rows.append({"n_points": n, "duration_at_500Hz_s": round(n / 500.0, 1),
                 "dense_gb": round(p.matrix_gb, 3),
                 "sparse_gb": round(p.sparse_gb, 3),
                 "tile_size": p.tile_size, "tile_gb": round(p.tile_gb, 4),
                 "chosen_store": p.store})
df_memory = pd.DataFrame(rows)
print(show(df_memory))
report.export(df_memory, "memory_plans",
              "storage strategy chosen per trajectory length, 8 GiB budget")
report.note("an hour of EEG at 500 Hz needs 3.0 TiB dense; the planner streams instead")


# ================================================================== 5
banner("5. RECURRENCE PLOTS BY COUPLING REGIME")
report.section("Recurrence plots")

matrices = {}
rows = []
for name, space in spaces.items():
    rm = rc.recurrence_plot(space, target_rr=0.05, theiler=1, rng=0)
    matrices[name] = rm
    rows.append({"space": name, **rm.describe().iloc[0].to_dict()})
df_plots = pd.DataFrame(rows)
print(show(df_plots.round(5),
           ["space", "kind", "n_rows", "dim", "epsilon", "target_rr", "actual_rr",
            "store", "tile_size"]))
report.export(df_plots, "recurrence_plots",
              "one recurrence plot per state space, all at 5% target rate")

print("\n  Detail:")
print(matrices["PAC alpha=0.9"].summary())


# ================================================================== 6
banner("6. CROSS RECURRENCE  (CRP)")
report.section("Cross recurrence")

circle_theta = sm.phase_circle(rec_pac, "phase_theta").subsample(max_points=2000)
circle_gamma = sm.phase_circle(rec_pac, "phase_gamma").subsample(max_points=2000)
crp = rc.cross_recurrence_plot(circle_theta, circle_gamma, target_rr=0.05, rng=0)
matrices["CRP theta vs gamma"] = crp
print(crp.summary())

lagged = circle_theta.window(0, 1800)
crp_self = rc.cross_recurrence_plot(circle_theta.window(120, 1920), lagged,
                                    target_rr=0.05, rng=0)
matrices["CRP theta vs lagged self"] = crp_self
print()
print(crp_self.summary())

df_crp = pd.concat([crp.describe(), crp_self.describe()])
df_crp.insert(0, "case", ["theta vs gamma", "theta vs lagged self"])
report.export(df_crp, "cross_recurrence", "cross recurrence plots, rectangular")


# ================================================================== 7
banner("7. JOINT RECURRENCE  (JRP) WITH INDEPENDENT THRESHOLDS  [DD-20]")
report.section("Joint recurrence")

phase_only = ss_pac.split(["phase_theta"])          # 2-D
amp_only = ss_pac.split(["amp_gamma"])              # 1-D
jrp = rc.joint_recurrence_plot([phase_only, amp_only], target_rr=0.10,
                               theiler=1, rng=0)
matrices["JRP phase AND amplitude"] = jrp

print(f"  subsystem dimensions : {[p.dim for p in jrp.parts]}   "
      "(different, as the definition allows)")
print(f"  independent epsilons : "
      f"{[round(t.scalar, 5) for t in jrp.thresholds]}")
print(f"  subsystem rates      : "
      f"{[f'{r:.3%}' for r in jrp.subsystem_rates()]}")
print(f"  joint rate           : {jrp.recurrence_rate():.3%}")
print(f"  independence ratio   : {jrp.independence_ratio():.3f}   "
      "(1.0 = subsystems recur independently)")

rows = []
for name, space in [("PAC alpha=0.9", ss_pac), ("PAC alpha=0.3", ss_weak),
                    ("uncoupled", ss_null)]:
    j = rc.joint_recurrence_plot([space.split(["phase_theta"]),
                                  space.split(["amp_gamma"])],
                                 target_rr=0.10, theiler=1, rng=0)
    rates = j.subsystem_rates()
    rows.append({"space": name,
                 "epsilon_phase": j.thresholds[0].scalar,
                 "epsilon_amplitude": j.thresholds[1].scalar,
                 "rr_phase": rates[0], "rr_amplitude": rates[1],
                 "rr_joint": j.recurrence_rate(),
                 "expected_if_independent": rates[0] * rates[1],
                 "independence_ratio": j.independence_ratio()})
df_jrp = pd.DataFrame(rows)
print()
print(show(df_jrp.round(5)))
report.export(df_jrp, "joint_recurrence",
              "joint recurrence with independent per-subsystem thresholds")
report.result("independence ratio, PAC alpha=0.9",
              f"{df_jrp.independence_ratio.iloc[0]:.3f}")
report.result("independence ratio, uncoupled",
              f"{df_jrp.independence_ratio.iloc[-1]:.3f}")


# ================================================================== 8
banner("8. SINGLE FIGURES")
report.section("Figures")

report.save(rc.plot_recurrence(matrices["PAC alpha=0.9"]),
            "fig_recurrence_pac.png", "recurrence plot, PAC alpha=0.9")
report.save(rc.plot_recurrence(matrices["uncoupled"]),
            "fig_recurrence_uncoupled.png", "recurrence plot, uncoupled control")
report.save(rc.plot_recurrence(matrices["Lorenz"]),
            "fig_recurrence_lorenz.png", "recurrence plot, Lorenz attractor")
report.save(rc.plot_recurrence(crp), "fig_cross_recurrence.png",
            "cross recurrence, theta vs gamma phase circles")
report.save(rc.plot_recurrence(jrp), "fig_joint_recurrence.png",
            "joint recurrence, phase AND amplitude subsystems")
report.save(rc.plot_density_profile(matrices["PAC alpha=0.9"]),
            "fig_density_profile_pac.png", "local recurrence density along time")

distances = rc.threshold_sampling.sample_pair_distances(
    ss_pac.weighted_coords, n_samples=80_000, theiler=1, rng=0)
report.save(rc.plot_threshold_diagnostics(thresholds["target_rr 5% (quantile)"],
                                          distances),
            "fig_threshold_diagnostics.png",
            "where epsilon sits in the sampled distance distribution")

for path in [a.path for a in report.artifacts if a.kind.startswith("figure")][-7:]:
    print(f"  {path}")


# ================================================================== 9
banner("9. COMPARISON MODE")

report.save(compare.compare_recurrence(
    {k: matrices[k] for k in ["PAC alpha=0.9", "PAC alpha=0.3", "uncoupled", "Lorenz"]},
    ncols=4, max_size=1300, title="Recurrence plots by regime, all at 5% target rate"),
    "fig_compare_recurrence_regimes.png",
    "recurrence plots across coupling regimes", kind="comparison")

rr_matrices = {f"RR={t:.0%}": rc.recurrence_plot(ss_pac, target_rr=t, theiler=1, rng=0)
               for t in (0.01, 0.05, 0.15, 0.30)}
report.save(compare.compare_recurrence(rr_matrices, ncols=4, max_size=1300,
                                       title="The same trajectory at four target rates"),
            "fig_compare_recurrence_rates.png",
            "effect of the target recurrence rate", kind="comparison")

report.save(compare.compare_recurrence(
    {"RP": matrices["PAC alpha=0.9"], "CRP": crp, "JRP": jrp},
    ncols=3, max_size=1300, title="Recurrence structures"),
    "fig_compare_structures.png", "RP, CRP and JRP side by side", kind="comparison")

report.save(compare.compare_thresholds(thresholds,
                                       title="Threshold modes on one state space"),
            "fig_compare_thresholds.png", "epsilon and achieved rate per mode",
            kind="comparison")

for path in [a.path for a in report.artifacts if a.kind.startswith("figure")][-4:]:
    print(f"  {path}")


# ================================================================== 10
banner("10. PROVENANCE AND RUN REPORT")
report.section("Provenance")

rm = matrices["PAC alpha=0.9"]
print(rm.provenance_frame()[["step", "caveats"]].to_string(index=False,
                                                           max_colwidth=52))
report.export(rm.provenance_frame(), "recurrence_provenance",
              "full operation history behind the PAC recurrence plot")

for step in rm.provenance:
    for caveat in step.caveats:
        report.warn(f"{step.name}: {caveat}")

report_path = report.write()
print(f"\n  Run report: {report_path}")
print(f"\n{SEP}\nDone. See outputs/\n{SEP}")
