#!/usr/bin/env python3
"""recurra -- ingestion, preprocessing and the synthetic generator (F0-F2).

Run:      python examples/example_01_pipeline.py
Writes:   outputs/   tidy CSV tables with provenance sidecars, plus figures.

Every figure drawn here is a single-figure call: one object in, one figure
out. Nothing in this script is required to reproduce any of them on your own
data -- see docs/EXAMPLES.md.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import recurra as rc

warnings.simplefilter("ignore", rc.RecurraWarning)
# run_all.py --out sets RECURRA_EXAMPLES_OUT; run alone, the default applies
rc.set_config(output_dir=os.environ.get("RECURRA_EXAMPLES_OUT", "outputs"))
SEP = "=" * 76


def banner(text: str) -> None:
    print(f"\n{SEP}\n{text}\n{SEP}")


# ------------------------------------------------------------------ 1
banner("1. ENVIRONMENT")
print(rc.doctor())


# ------------------------------------------------------------------ 2
banner("2. INPUT SHAPES (ingest)")
T, FS = 4000, 500.0
rng = np.random.default_rng(0)

inputs = {
    "I1  1-D array": rc.ingest(rng.standard_normal(T), fs=FS),
    "I2  2-D array": rc.ingest(rng.standard_normal((T, 4)), fs=FS),
    "I4  Hilbert components": rc.ingest(
        {"phase_theta": rng.uniform(-np.pi, np.pi, T), "amp_gamma": rng.gamma(2, 1, T)},
        fs=FS, roles={"phase_theta": "phase", "amp_gamma": "amplitude"}),
    "I5  complex analytic": rc.ingest(
        {"z": np.exp(1j * np.cumsum(rng.normal(0.1, 0.01, T)))}, fs=FS),
    "I6  prebuilt state space": rc.ingest(rng.standard_normal((T, 3)), fs=FS,
                                          kind="statespace"),
}
for label, rec in inputs.items():
    print(f"  {label:26s} -> {rec.n_channels} channels, caps={rec.caps}")

print("\n  Capabilities block invalid compositions:")
try:
    rc.bandpass(inputs["I6  prebuilt state space"], (4, 8))
except rc.CapabilityError as exc:
    print(f"  CapabilityError: {exc}")


# ------------------------------------------------------------------ 3
banner("3. GENERATOR VALIDATION AGAINST CANONICAL INDICES")
rows = []
for modality in ["pac", "ppc", "aac", "none"]:
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sig = rc.generate_cfc(modality=modality, duration=40.0, fs=500.0, alpha=alpha,
                              preferred_phase=np.pi / 2, snr_db=20.0, seed=11)
        row = {"modality": modality, "alpha": alpha}
        comp = sig.components
        if "phase_low" in comp and "amp_high" in comp:
            row["mvl"] = rc.mvl(comp["phase_low"], comp["amp_high"])
            row["mi_tort"] = rc.modulation_index(comp["phase_low"], comp["amp_high"])
        if "phase_high" in comp:
            row["plv"] = rc.plv(comp["phase_low"], comp["phase_high"])
        if "amp_low" in comp:
            row["aac"] = abs(rc.aac_index(comp["amp_low"], comp["amp_high"]))
        rows.append(row)

df_validation = pd.DataFrame(rows)
print(df_validation.round(4).to_string(index=False))

print("\n  Monotonicity in alpha (Spearman rho):")
for modality, column in [("pac", "mvl"), ("pac", "mi_tort"),
                         ("ppc", "plv"), ("aac", "aac")]:
    sub = df_validation[df_validation.modality == modality].dropna(subset=[column])
    if len(sub) > 2:
        print(f"    {modality:4s} {column:8s} rho = "
              f"{spearmanr(sub.alpha, sub[column]).statistic:+.3f}")

print(f"\n  -> {rc.export_frame(df_validation, 'generator_validation')}")


# ------------------------------------------------------------------ 4
banner("4. PREPROCESSING CHAIN WITH PROVENANCE")
sig = rc.generate_cfc(modality="pac", duration=30.0, fs=500.0, alpha=0.85,
                      preferred_phase=np.pi / 2, snr_db=20.0, seed=17)
rec = sig.to_recording("sub01")
rec = rc.filterbank(rec, {"theta": (4, 8), "gamma": (50, 70)})
rec = rc.analytic(rec, sources=["theta", "gamma"])

print(rec.summary())
print("\n  Full history:")
print(rec.provenance_frame()[["step", "caveats"]].to_string(index=False,
                                                            max_colwidth=58))
written = rc.export_recording(rec, "sub01")
print(f"\n  -> {written['channels']}")

rc.save_figure(rc.plot_signal(rec, ["signal", "theta", "gamma", "amp_gamma"],
                              seconds=4.0), "fig_signal_sub01.png")
rc.save_figure(rc.plot_phase_amplitude(rec, "phase_theta", "amp_gamma"),
               "fig_phase_amplitude_sub01.png")
rc.save_figure(rc.plot_quality(rec), "fig_quality_sub01.png")
print("  -> outputs/fig_signal_sub01.png, fig_phase_amplitude_sub01.png, "
      "fig_quality_sub01.png")


# ------------------------------------------------------------------ 5
banner("5. THE HETEROGENEOUS-SCALE PROBLEM  [DD-02]")
from recurra.preprocess.scaling import Scaler, theoretical_rms_pairwise

phi = np.asarray(rec.get("phase_theta"))
amp = np.asarray(rec.get("amp_gamma")).reshape(-1, 1)
circle = np.column_stack([np.cos(phi), np.sin(phi)])

print(f"  RMS pairwise distance, unit circle : "
      f"{theoretical_rms_pairwise(circle):.4f}   (theory sqrt(2) = {np.sqrt(2):.4f})")
print(f"  RMS pairwise distance, envelope    : {theoretical_rms_pairwise(amp):.4f}")

scale_rows = []
print("\n  WITHOUT correction, varying amplifier gain:")
print(f"  {'gain':>9} {'phase share':>12} {'amp share':>11}  dominant")
for gain in [0.01, 0.1, 1.0, 10.0, 100.0]:
    rep = Scaler("none").fit([("phase", circle), ("amplitude", amp * gain)])
    print(f"  {gain:9.2f} {rep.variance_share[0]:12.4f} "
          f"{rep.variance_share[1]:11.4f}  {rep.dominant or '-'}")
    scale_rows.append({"policy": "none", "gain": gain,
                       "phase_share": rep.variance_share[0],
                       "amplitude_share": rep.variance_share[1],
                       "dominant_block": rep.dominant or ""})

print("\n  WITH rms_balanced (default): invariant to gain")
for gain in [0.01, 1.0, 100.0]:
    rep = Scaler("rms_balanced").fit([("phase", circle), ("amplitude", amp * gain)])
    print(f"  {gain:9.2f} {rep.variance_share[0]:12.4f} "
          f"{rep.variance_share[1]:11.4f}  {rep.dominant or '-'}")
    scale_rows.append({"policy": "rms_balanced", "gain": gain,
                       "phase_share": rep.variance_share[0],
                       "amplitude_share": rep.variance_share[1],
                       "dominant_block": rep.dominant or ""})

print("\n  Lambda sweep: the split is exactly lambda")
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rep = Scaler("weighted", lambda_=lam).fit([("phase", circle),
                                               ("amplitude", amp * 37.0)])
    print(f"  lambda={lam:.2f} -> phase={rep.variance_share[0]:.3f}  "
          f"amplitude={rep.variance_share[1]:.3f}")
    scale_rows.append({"policy": f"weighted_lambda_{lam:g}", "gain": 37.0,
                       "phase_share": rep.variance_share[0],
                       "amplitude_share": rep.variance_share[1],
                       "dominant_block": rep.dominant or ""})

print(f"\n  -> {rc.export_frame(pd.DataFrame(scale_rows), 'scale_policy_effect')}")


# ------------------------------------------------------------------ 6
banner("6. MULTI-SUBJECT CORPUS -> ONE TIDY CSV")
exporter = rc.CsvExporter("corpus_coupling_metrics")
for i in range(12):
    alpha = i / 11.0
    sig = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=alpha,
                          preferred_phase=np.pi / 2, snr_db=15.0, seed=100 + i)
    r = rc.analytic(rc.filterbank(sig.to_recording(f"sub{i:02d}"),
                                  {"theta": (4, 8), "gamma": (50, 70)}),
                    sources=["theta", "gamma"])
    for name, value in rc.coupling_indices(r.get("phase_theta"),
                                           r.get("amp_gamma")).items():
        exporter.add(subject=f"sub{i:02d}", alpha_true=alpha, metric=name,
                     value=value, n_samples=r.n_samples, fs_hz=r.fs)

print(exporter.frame.pivot(index="alpha_true", columns="metric",
                           values="value").round(4).to_string())
print(f"\n  {len(exporter)} rows -> "
      f"{exporter.write(note='example 01', n_subjects=12)}")

print(f"\n{SEP}\nDone. See outputs/\n{SEP}")
