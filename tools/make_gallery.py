#!/usr/bin/env python3
"""Regenerate the figures shown in README.md and the documentation.

    python tools/make_gallery.py            # writes docs/_static/gallery/*.png

Synthetic signals with fixed seeds, so the gallery is reproducible and shows
what the library draws, not a particular dataset.
"""
from __future__ import annotations

import pathlib
import warnings

import numpy as np

import recurra as rc
from recurra import statespace as sm

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "_static" / "gallery"
DPI = 90


def main() -> None:
    warnings.simplefilter("ignore", rc.RecurraWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    rc.set_config(output_dir=str(OUT))

    # a phase-amplitude coupled signal: s(t) = (cos phi_theta, sin phi_theta, A_gamma)
    sig = rc.generate_cfc(modality="pac", duration=40.0, fs=250.0, alpha=0.8,
                          preferred_phase=np.pi / 2, snr_db=15.0, rng=1)
    rec = rc.analytic(rc.filterbank(sig.to_recording("demo"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    space = sm.pac_space(rec, smooth=12.0)
    short = space.window(0, 2500).subsample(max_points=900)

    rp = rc.recurrence_plot(short, target_rr=0.05, rng=0)
    rc.save_figure(rc.plot_recurrence_panel(rp, rqa=True, l_min=2, v_min=2),
                   "recurrence_panel.png", dpi=DPI)
    rc.save_figure(rc.plot_rqa_summary(rp, l_min=2, v_min=2), "rqa_summary.png", dpi=DPI)

    phase, amplitude = short.split(["phase_theta"]), short.split(["amp_gamma"])
    jrp = rc.joint_recurrence_plot([phase, amplitude], target_rr=0.10, rng=0)
    rc.save_figure(rc.plot_joint_recurrence(jrp, names=("theta phase", "gamma envelope")),
                   "joint_recurrence.png", dpi=DPI)

    # a cross recurrence plot: Lorenz against a delayed, noisy copy of itself
    X = rc.lorenz(n_points=900, dt=0.02)
    Y = np.roll(X, 25, axis=0) + 0.5 * np.random.default_rng(0).standard_normal(X.shape)
    crp = rc.cross_recurrence_plot(X, Y, target_rr=0.05, rng=0)
    rc.save_figure(rc.plot_recurrence_panel(crp, names=("Lorenz", "delayed copy")),
                   "cross_recurrence_panel.png", dpi=DPI)

    # the meta-RP: recurrence of the RQA measures taken in 2 s windows
    wr = rc.windowed_recurrence(space.subsample(max_points=5000),
                                rc.WindowSpec(size=2.0, overlap=0.5, unit="seconds"),
                                target_rr=0.05, rng=0)
    wr.metrics(rqa=True, l_min=2, v_min=2)
    meta = wr.meta_recurrence(["DET", "LAM", "ENTR"], target_rr=0.2, rng=0)
    rc.save_figure(rc.plot_recurrence_panel(meta, names=("window metrics", "window metrics"),
                                            title="Meta recurrence plot (RQA per window)"),
                   "meta_recurrence_panel.png", dpi=DPI)
    print(f"gallery written to {OUT}")


if __name__ == "__main__":
    main()
