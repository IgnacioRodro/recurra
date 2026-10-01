#!/usr/bin/env python3
"""recurra -- every route to a state space (F3).

Run:      python examples/example_02_statespace.py
Writes:   outputs/   CSV tables and figures.

Section 7 draws single figures: one object in, one figure out. Section 8
switches to comparison mode, which takes a mapping of state spaces already
built above. No plotting function in this script constructs data.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

import recurra as rc
from recurra import statespace as ss_mod
from recurra.viz import compare

warnings.simplefilter("ignore", rc.RecurraWarning)
# run_all.py --out sets RECURRA_EXAMPLES_OUT; run alone, the default applies
rc.set_config(output_dir=os.environ.get("RECURRA_EXAMPLES_OUT", "outputs"))
SEP = "=" * 80


def banner(text: str) -> None:
    print(f"\n{SEP}\n{text}\n{SEP}")


def show(df, cols=None) -> str:
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    return d.to_string(index=False)


# ================================================================== 1
banner("1. TEST SIGNALS")

sig_pac = rc.generate_cfc(modality="pac", duration=40.0, fs=500.0, alpha=0.9,
                          preferred_phase=np.pi / 2, concentration=2.0,
                          snr_db=20.0, seed=17)
sig_ppc = rc.generate_cfc(modality="ppc", duration=40.0, fs=500.0, alpha=0.9,
                          nm_ratio=(1, 1), snr_db=20.0, seed=18)
sig_aac = rc.generate_cfc(modality="aac", duration=40.0, fs=500.0, alpha=0.9,
                          snr_db=20.0, seed=19)
sig_ppa = rc.generate_cfc(modality="ppa", duration=40.0, fs=500.0, alpha=0.9,
                          preferred_phase=np.pi / 2, snr_db=20.0, seed=20)
sig_null = rc.generate_cfc(modality="none", duration=40.0, fs=500.0,
                           snr_db=20.0, seed=21)

BANDS = {"theta": (4, 8), "gamma": (50, 70)}


def prepare(sig, subject, bands=BANDS):
    r = rc.filterbank(sig.to_recording(subject), bands)
    return rc.analytic(r, sources=list(bands))


rec_pac = prepare(sig_pac, "pac01")
rec_ppc = prepare(sig_ppc, "ppc01")
rec_aac = prepare(sig_aac, "aac01")
rec_null = prepare(sig_null, "null01")
rec_ppa = prepare(sig_ppa, "ppa01",
                  {"delta": (2, 4), "theta": (8, 12), "gamma": (50, 70)})

print(f"  PAC    : {rec_pac.n_samples} samples, channels {rec_pac.names}")
print(f"  PPA    : {rec_ppa.n_samples} samples, {rec_ppa.n_channels} channels")

X_lorenz = rc.lorenz(n_points=20000, dt=0.01, transient=20.0)
rec_lorenz = rc.ingest(X_lorenz, fs=100.0, labels=["x", "y", "z"], subject="lorenz")
print(f"  Lorenz : {rec_lorenz.n_samples} samples, 3 observable state variables")


# ================================================================== 2
banner("2. EMBEDDING PARAMETER ESTIMATION")

rows = []
for tau_method in ["ami", "acf", "first_zero"]:
    for m_method in ["fnn", "cao"]:
        p = ss_mod.estimate_embedding(X_lorenz[:, 0], tau_method=tau_method,
                                      m_method=m_method, tau_range=(1, 300),
                                      m_range=(1, 10), rng=0)
        rows.append({"signal": "lorenz_x", "tau_method": tau_method,
                     "m_method": m_method, "tau": p.tau, "m": p.m,
                     "m_criterion": p.diagnostics["m_criterion"]})
for tau_method in ["ami", "acf"]:
    p = ss_mod.estimate_embedding(np.asarray(rec_pac.get("signal")),
                                  tau_method=tau_method, m_method="fnn",
                                  tau_range=(1, 200), m_range=(1, 10), rng=0)
    rows.append({"signal": "eeg_like_pac", "tau_method": tau_method,
                 "m_method": "fnn", "tau": p.tau, "m": p.m,
                 "m_criterion": p.diagnostics["m_criterion"]})

df_embedding = pd.DataFrame(rows)
print(show(df_embedding))
print("\n  Lorenz has true dimension 3. AMI + normalised FNN recovers it.")
rc.export_frame(df_embedding, "embedding_parameters")

params_lorenz = ss_mod.estimate_embedding(X_lorenz[:, 0], tau_method="ami",
                                          m_method="fnn", tau_range=(1, 300),
                                          m_range=(1, 10), rng=0)
rc.export_frame(params_lorenz.curve_frame(), "embedding_curves")


# ================================================================== 3
banner("3. ALL CONSTRUCTORS")

spaces: dict[str, object] = {}

spaces["A pac_space (observable)"] = ss_mod.pac_space(rec_pac)
spaces["B pac_space embed m=3 (hybrid)"] = ss_mod.pac_space(rec_pac, embed_amplitude=3)
spaces["C phase_circle (observable)"] = ss_mod.phase_circle(rec_pac, "phase_theta")
spaces["D ppc_space (observable)"] = ss_mod.ppc_space(rec_ppc, "phase_theta",
                                                      "phase_gamma")
spaces["E envelope_space (observable)"] = ss_mod.envelope_space(rec_aac)
spaces["F envelope_space embed (hybrid)"] = ss_mod.envelope_space(rec_aac, embed=2)
spaces["G ppa_space (observable)"] = ss_mod.ppa_space(rec_ppa, "phase_delta",
                                                      "phase_theta", "amp_gamma")
spaces["H nested_space (observable)"] = ss_mod.nested_space(rec_ppa, [
    {"source": "phase_delta", "as": "phase_circle", "label": "delta"},
    {"source": "amp_theta", "as": "amplitude", "label": "theta"},
    {"source": "amp_gamma", "as": "amplitude", "label": "gamma"},
])
spaces["I freq_amp_space (observable)"] = ss_mod.freq_amp_space(rec_pac, "ifreq_theta",
                                                                "amp_gamma")
spaces["J channels (observable)"] = ss_mod.channels(rec_lorenz, ["x", "y", "z"])
spaces["K takens auto (delay)"] = ss_mod.takens(rec_lorenz, "x", m="auto",
                                                tau="auto", rng=0)
spaces["L takens m=3 tau=16 (delay)"] = ss_mod.takens(rec_lorenz, "x", m=3, tau=16)
spaces["M takens_multivariate (delay)"] = ss_mod.takens_multivariate(
    rec_lorenz, ["x", "y"], m=2, tau=16)
spaces["N build custom (hybrid)"] = ss_mod.build(rec_pac, [
    {"source": "phase_theta", "as": "phase_circle", "label": "theta_phase"},
    {"source": "amp_gamma", "as": "delay", "m": 3, "tau": 8, "label": "gamma_env"},
], route="hybrid", builder="build(custom)")
spaces["O from_bands (declarative)"] = ss_mod.from_bands(
    sig_pac.to_recording("pac02"),
    [{"band": (4, 8), "as": "phase", "label": "theta"},
     {"band": (50, 70), "as": "amplitude", "label": "gamma"}])
spaces["P custom callable (external)"] = ss_mod.custom(
    rec_pac,
    lambda r: np.column_stack([np.cos(r.get("phase_theta")),
                               np.sin(r.get("phase_theta")),
                               np.log1p(r.get("amp_gamma"))]),
    scaling="rms_balanced")
spaces["Q pac_space unscaled"] = ss_mod.pac_space(rec_pac, scaling="none")
spaces["R pac_space lambda=0.8"] = ss_mod.pac_space(rec_pac, scaling="weighted",
                                                    lambda_=0.8)
spaces["S pac_space, uncoupled signal"] = ss_mod.pac_space(rec_null)
spaces["T pac_space reduced to PCA-2"] = ss_mod.pac_space(
    rec_ppa, "phase_delta", "amp_gamma").reduce("pca", 2)

df_constructors = pd.DataFrame(
    [{"constructor": name, **s.describe().iloc[0].to_dict()}
     for name, s in spaces.items()])
print(show(df_constructors, ["constructor", "route", "dim", "n_points", "blocks",
                             "scaling", "variance_share", "max_share"]))
rc.export_frame(df_constructors, "statespace_constructors")

print("\n  Detailed summary of constructor B (hybrid):")
print(spaces["B pac_space embed m=3 (hybrid)"].summary())


# ================================================================== 4
banner("4. SCALING POLICY ON THE SAME SPACE")

rows = []
policy_spaces = {}
for label, base, kw in [("none", "none", {}), ("rms_balanced", "rms_balanced", {}),
                        ("lambda=0.00", "weighted", {"lambda_": 0.0}),
                        ("lambda=0.50", "weighted", {"lambda_": 0.5}),
                        ("lambda=1.00", "weighted", {"lambda_": 1.0})]:
    s = ss_mod.pac_space(rec_pac, scaling="none").rescale(base, **kw)
    policy_spaces[label] = s
    frame = s.scale_frame()
    rows.append({"policy": label, "phase_weight": frame.weight[0],
                 "amplitude_weight": frame.weight[1],
                 "phase_share": frame.variance_share[0],
                 "amplitude_share": frame.variance_share[1],
                 **ss_mod.geometry_descriptors(s, rng=0)})
df_policies = pd.DataFrame(rows)
print(show(df_policies.round(4)))
rc.export_frame(df_policies, "scaling_policies")


# ================================================================== 5
banner("5. ROUTE COMPARISON ON THE SAME SIGNAL  (objective O2.1)")

df_routes, route_spaces = ss_mod.compare_routes(rec_pac, rng=0)
cols = ["route_name", "route", "dim", "n_points", "scaling", "max_share",
        "nn_mean", "nn_cv", "corr_slope"]
cols += [c for c in df_routes.columns if c.startswith("nn_agreement")]
print(show(df_routes.round(4), cols))
rc.export_frame(df_routes, "route_comparison")

print("\n  Same exercise on the uncoupled control signal:")
df_routes_null, _ = ss_mod.compare_routes(rec_null, rng=0)
print(show(df_routes_null.round(4), cols))
rc.export_frame(df_routes_null, "route_comparison_control")


# ================================================================== 6
banner("6. LAMBDA SWEEP  (pure phase -> pure amplitude)")

sweep_pac = ss_mod.lambda_sweep(ss_mod.pac_space(rec_pac, scaling="none"),
                                grid=np.linspace(0, 1, 11), rng=0)
sweep_null = ss_mod.lambda_sweep(ss_mod.pac_space(rec_null, scaling="none"),
                                 grid=np.linspace(0, 1, 11), rng=0)
sweep_pac["signal"] = "PAC alpha=0.9"
sweep_null["signal"] = "uncoupled"
cols = ["signal", "lambda", "nn_cv", "corr_slope",
        "nn_agreement_vs_block0", "nn_agreement_vs_block1"]
print(show(pd.concat([sweep_pac, sweep_null]).round(4), cols))
rc.export_frame(pd.concat([sweep_pac, sweep_null]), "lambda_sweep")


# ================================================================== 7
banner("7. SINGLE FIGURES  (one object -> one figure)")

print(rc.list_figures())
print()

single_figures = [
    ("fig_attractor_pac_observable.png", rc.plot_attractor,
     spaces["A pac_space (observable)"], dict(mode="3d")),
    ("fig_attractor_pac_hybrid.png", rc.plot_attractor,
     spaces["B pac_space embed m=3 (hybrid)"], dict(mode="3d")),
    ("fig_attractor_pac_torus.png", rc.plot_attractor,
     spaces["A pac_space (observable)"], dict(mode="torus", cmap="plasma")),
    ("fig_attractor_ppc_pairs.png", rc.plot_attractor,
     spaces["D ppc_space (observable)"], dict(mode="pairs")),
    ("fig_attractor_aac_envelopes.png", rc.plot_attractor,
     spaces["E envelope_space (observable)"], dict(mode="2d")),
    ("fig_attractor_ppa_pairs.png", rc.plot_attractor,
     spaces["G ppa_space (observable)"], dict(mode="pairs")),
    ("fig_attractor_lorenz_observable.png", rc.plot_attractor,
     spaces["J channels (observable)"], dict(mode="3d", line=True, alpha=0.7)),
    ("fig_attractor_lorenz_takens.png", rc.plot_attractor,
     spaces["L takens m=3 tau=16 (delay)"], dict(mode="3d", line=True, alpha=0.7)),
    ("fig_attractor_freq_amp.png", rc.plot_attractor,
     spaces["I freq_amp_space (observable)"], dict(mode="2d")),
    ("fig_attractor_pac_timeseries.png", rc.plot_attractor,
     spaces["A pac_space (observable)"], dict(mode="time")),
    ("fig_embedding_diagnostics_lorenz.png", rc.plot_embedding_diagnostics,
     params_lorenz, {}),
    ("fig_scale_report_pac.png", rc.plot_scale_report,
     spaces["A pac_space (observable)"], {}),
    ("fig_scale_report_pac_unscaled.png", rc.plot_scale_report,
     spaces["Q pac_space unscaled"], {}),
    ("fig_lambda_sweep_pac.png", rc.plot_lambda_sweep, sweep_pac, {}),
    ("fig_phase_amplitude_pac.png", rc.plot_phase_amplitude, rec_pac, {}),
    ("fig_envelope_pair_aac.png", rc.plot_envelope_pair, rec_aac,
     dict(amplitude_a="amp_theta", amplitude_b="amp_gamma")),
]

figure_rows = []
for filename, fn, obj, kwargs in single_figures:
    path = rc.save_figure(fn(obj, **kwargs), filename)
    figure_rows.append({"figure": filename, "kind": "single",
                        "function": fn.__name__, "path": path})
    print(f"  {path}")


# ================================================================== 8
banner("8. COMPARISON MODE  (mapping of built objects -> panels)")

comparison_figures = [
    ("fig_compare_routes_attractors.png",
     compare.compare_attractors,
     {k: spaces[k] for k in ["A pac_space (observable)",
                             "B pac_space embed m=3 (hybrid)",
                             "Q pac_space unscaled",
                             "S pac_space, uncoupled signal",
                             "J channels (observable)",
                             "L takens m=3 tau=16 (delay)"]},
     dict(ncols=3, max_points=3500, title="State space routes")),
    ("fig_compare_lambda_attractors.png",
     compare.compare_attractors,
     {f"lambda={lam:.2f}": ss_mod.pac_space(rec_pac, scaling="none").reweight(lam)
      for lam in [0.0, 0.25, 0.5, 0.75, 1.0]},
     dict(ncols=5, mode="3d", max_points=3000, title="Weighting sweep")),
    ("fig_compare_scale_policies.png",
     compare.compare_scale_policies, policy_spaces,
     dict(title="Scaling policies on the same PAC space")),
    ("fig_compare_route_metrics.png",
     compare.compare_routes, df_routes, dict(title="Route geometry")),
    ("fig_compare_lambda_sweeps.png",
     compare.compare_series, {"PAC alpha=0.9": sweep_pac, "uncoupled": sweep_null},
     dict(x="lambda", y="corr_slope", title="Correlation slope vs lambda",
          xlabel="$\\lambda$")),
]

for filename, fn, obj, kwargs in comparison_figures:
    path = rc.save_figure(fn(obj, **kwargs), filename)
    figure_rows.append({"figure": filename, "kind": "comparison",
                        "function": fn.__name__, "path": path})
    print(f"  {path}")


# ------------------------------------------------------------------
# Coupling regimes: three recordings built here, drawn in comparison mode.
regime_recordings = {}
regime_spaces = {}
for label, kwargs in [
    ("uncoupled (alpha=0)", dict(modality="none", alpha=0.0)),
    ("distributed PAC (alpha=0.95)", dict(modality="pac", alpha=0.95,
                                          preferred_phase=None)),
    ("preferred phase pi/2 (alpha=0.95)", dict(modality="pac", alpha=0.95,
                                               preferred_phase=np.pi / 2,
                                               concentration=3.0)),
]:
    s = rc.generate_cfc(duration=90.0, fs=500.0, snr_db=30.0, f_low=6.0, f_high=60.0,
                        bandwidth_low=3.0, bandwidth_high=20.0, seed=5, **kwargs)
    r = rc.analytic(rc.filterbank(s.to_recording("regime"),
                                  {"theta": (4.5, 7.5), "gamma": (50, 70)}),
                    sources=["theta", "gamma"])
    regime_recordings[label] = r
    regime_spaces[label] = ss_mod.pac_space(r, smooth=8.0)

for filename, fn, obj, kwargs in [
    ("fig_compare_regimes_torus.png", compare.compare_attractors, regime_spaces,
     dict(mode="torus", ncols=3, max_points=7000, point_size=1.8, alpha=0.22,
          cmap="plasma", share_limits=False,
          title="Toroidal view: angle = phase, radius = 1 + normalised amplitude")),
    ("fig_compare_regimes_phase_amplitude.png", compare.compare_phase_amplitude,
     regime_recordings, dict(ncols=3, title="Phase-amplitude distribution by regime")),
]:
    path = rc.save_figure(fn(obj, **kwargs), filename)
    figure_rows.append({"figure": filename, "kind": "comparison",
                        "function": fn.__name__, "path": path})
    print(f"  {path}")

regime_rows = []
for label, r in regime_recordings.items():
    ph, am = np.asarray(r.get("phase_theta")), np.asarray(r.get("amp_gamma"))
    frame = regime_spaces[label].scale_frame()
    regime_rows.append({"regime": label, "mi_tort": rc.modulation_index(ph, am),
                        "mvl": rc.mvl(ph, am),
                        "tail_ratio_phase": frame.tail_ratio[0],
                        "tail_ratio_amplitude": frame.tail_ratio[1]})
print()
print(show(pd.DataFrame(regime_rows).round(4)))
rc.export_frame(pd.DataFrame(regime_rows), "coupling_regimes")
rc.export_frame(pd.DataFrame(figure_rows), "figure_index")


# ================================================================== 9
banner("9. STATE SPACE PROVENANCE")

ss = spaces["B pac_space embed m=3 (hybrid)"]
print(ss.provenance_frame()[["step", "params", "caveats"]].to_string(
    index=False, max_colwidth=50))
rc.export_frame(ss.provenance_frame(), "statespace_provenance")

print(f"\n{SEP}\nDone. {len(spaces)} state spaces, "
      f"{len(figure_rows)} figures. See outputs/\n{SEP}")
