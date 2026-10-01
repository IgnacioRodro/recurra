"""recurra cookbook -- every command and configuration, one recipe at a time.

Written as Spyder cells. Open in Spyder and run a cell with Ctrl+Enter, or the
whole file with F5. Recipes are numbered and mostly independent; cell 0 sets up
the shared objects the later ones reuse.

Sections
--------
  0     setup
  1-12  ingestion, quality, preprocessing
  13-22 synthetic signals and canonical indices
  23-38 state space: every constructor, route, scaling and transform
  39-48 thresholds: every mode and scope
  49-58 recurrence: RP, CRP, JRP, storage, metrics, memory
  59-68 windowed analysis
  69-74 comparability and group separation
  75-84 figures
  85-90 export, reports, configuration
"""

# %% 0 -- setup: run this cell first
import os
import warnings

import numpy as np
import pandas as pd

import recurra as rc
from recurra import statespace as sm
from recurra.viz import compare

warnings.simplefilter("ignore", rc.RecurraWarning)          # caveats are informative, not errors
# run_all.py --out sets RECURRA_EXAMPLES_OUT; run alone, the default applies
rc.set_config(output_dir=os.environ.get("RECURRA_EXAMPLES_OUT", "cookbook_out"))

print(rc.doctor())


def make_signal(modality="pac", alpha=0.8, duration=20.0, seed=0, **kw):
    """A short PAC-style recording, ready for the state-space builders."""
    sig = rc.generate_cfc(modality=modality, duration=duration, fs=500.0,
                          alpha=alpha, preferred_phase=np.pi / 2, snr_db=18.0,
                          seed=seed, **kw)
    rec = rc.filterbank(sig.to_recording(f"sub{seed:02d}"),
                        {"theta": (4, 8), "gamma": (50, 70)})
    return sig, rc.analytic(rec, sources=["theta", "gamma"])


SIG, REC = make_signal(seed=17)
SS = sm.pac_space(REC, smooth=8.0).subsample(max_points=1200)
print(SS.summary())


# %% 1 -- ingest a 1-D array
rec = rc.ingest(np.random.default_rng(5500).standard_normal(4000), fs=500.0)
print(rec.summary())

# %% 2 -- ingest a multichannel array with labels
X = np.random.default_rng(5900).standard_normal((4000, 4))
rec = rc.ingest(X, fs=500.0, labels=["Fz", "Cz", "Pz", "Oz"], subject="s01")
print(rec.names, rec.caps)

# %% 3 -- a transposed array is refused with an explanation
try:
    rc.ingest(np.random.default_rng(6500).standard_normal((4, 4000)), fs=500.0)
except rc.IngestError as e:
    print("IngestError:", e)

# %% 4 -- ingest bands already filtered
rec = rc.ingest({"theta": np.random.default_rng(7000).standard_normal(4000), "gamma": np.random.default_rng(7001).standard_normal(4000)},
                fs=500.0, kind="bands")
print(rec.caps)

# %% 5 -- ingest Hilbert components already computed (roles declared explicitly)
rec = rc.ingest({"phase_theta": np.random.default_rng(7551).uniform(-np.pi, np.pi, 4000),
                 "amp_gamma": np.random.default_rng(7651).gamma(2, 1, 4000)},
                fs=500.0,
                roles={"phase_theta": "phase", "amp_gamma": "amplitude"})
print(rec.caps, "| RAW present:", rc.Capability.RAW in rec.caps)

# %% 6 -- ingest a complex analytic signal (split automatically)
z = np.exp(1j * np.cumsum(np.random.default_rng(8251).normal(0.1, 0.01, 4000)))
rec = rc.ingest({"sig": z}, fs=500.0)
print(rec.names)

# %% 7 -- ingest a (phase, amplitude) pair
rec = rc.ingest((np.random.default_rng(8700).random((4000)), np.random.default_rng(8701).random(4000)), fs=500.0)
print(rec.roles)

# %% 8 -- ingest a prebuilt state space
phi = np.linspace(0, 40 * np.pi, 4000)
rec = rc.ingest(np.column_stack([np.cos(phi), np.sin(phi), np.random.default_rng(9200).random(4000)]),
                fs=500.0, kind="statespace")
print(rec.caps)

# %% 9 -- ingest from a CSV file (fs inferred from a time column)
df = pd.DataFrame({"time": np.arange(4000) / 500.0, "ch0": np.random.default_rng(9700).standard_normal(4000)})
df.to_csv("cookbook_demo.csv", index=False)
rec = rc.ingest("cookbook_demo.csv")
print(rec.fs, rec.names)

# %% 9b -- a corpus on disk: recordings joined to a participants table [DD-89]
# Built here from CSVs so the recipe runs anywhere; with real data the same
# call reads a directory of EDF files under sub-XXX folders.
import pathlib
import tempfile

_root = pathlib.Path(tempfile.mkdtemp()) / "study"
for _i in range(4):
    _sub = f"sub-{_i:03d}"
    _d = _root / _sub
    _d.mkdir(parents=True)
    pd.DataFrame({"time": np.arange(2000) / 250.0,
                  "Cz": np.random.default_rng(_i).standard_normal(2000)}
                 ).to_csv(_d / f"{_sub}_eeg.csv", index=False)
pd.DataFrame({"participant_id": [f"sub-{i:03d}" for i in range(4)] + ["sub-099"],
              "group": ["control", "control", "patient", "patient", "patient"]}
             ).to_csv(_root / "participants.tsv", sep="\t", index=False)

_index = rc.read_corpus(_root, pattern="*.csv")
print(_index.summary())
print("\nis it a CorpusIndex:", isinstance(_index, rc.CorpusIndex))
print("paths, not data:", list(_index.recordings)[:3])
print("labelled subjects:", _index.labelled())
print("\nthe join as a table:")
print(_index.to_frame().to_string(index=False))

# %% 10 -- quality report: a constant channel is an error, not a shrug
rec = rc.ingest({"good": np.random.default_rng(12900).standard_normal(2000), "flat": np.ones(2000)}, fs=500.0)
print(rec.quality.summary())
print(rec.quality.to_frame().to_string(index=False))

# %% 11 -- strict mode turns quality errors into exceptions
try:
    rc.ingest({"flat": np.ones(2000)}, fs=500.0, strict=True)
except rc.QualityError as e:
    print("QualityError:", e)

# %% 12 -- capabilities block invalid compositions before anything is computed
ss_rec = rc.ingest(np.random.default_rng(14000).standard_normal((2000, 3)), fs=500.0, kind="statespace")
try:
    rc.bandpass(ss_rec, (4, 8))
except rc.CapabilityError as e:
    print("CapabilityError:", e)


# %% 13 -- band-pass filtering, FIR with automatic order
rec = rc.ingest(SIG.data[:, 0], fs=500.0, subject="demo")
out = rc.bandpass(rec, (4, 8), name="theta")
step = [s for s in out.provenance if s.name == "bandpass"][0]
print(step.params)
print(step.caveats)

# %% 14 -- IIR instead, and an explicit order
a = rc.bandpass(rec, (4, 8), name="fir", method="fir", order="auto")
b = rc.bandpass(rec, (4, 8), name="iir", method="iir", order=4)
c = rc.bandpass(rec, (4, 8), name="long", method="fir", min_cycles=6.0)
for r, lbl in [(a, "fir auto"), (b, "iir 4"), (c, "fir 6 cycles")]:
    st = [s for s in r.provenance if s.name == "bandpass"][-1]
    print(f"{lbl:14s} order={st.params['order']:5d} edge={st.params['edge_samples']}")

# %% 15 -- a filter bank
rec_b = rc.filterbank(rec, {"delta": (1, 4), "theta": (4, 8),
                            "alpha": (8, 13), "gamma": (30, 80)})
print(rec_b.by_role("band"))

# %% 16 -- Hilbert decomposition, all four edge policies
for policy in ("trim", "mirror", "taper", "none"):
    out = rc.analytic(rc.bandpass(rec, (4, 8), name="th"), sources=["th"],
                      edge_policy=policy)
    print(f"{policy:7s} -> {out.n_samples} samples, t0={out.t0:.3f}")

# %% 17 -- keep only some components, and unwrap the phase
out = rc.analytic(rc.bandpass(rec, (4, 8), name="th"), sources=["th"],
                  keep=("phase", "amplitude"), unwrap=True)
print(out.names, "phase range:", np.ptp(out.get("phase_th")).round(2))

# %% 18 -- notch and resample
n = rc.notch(rec, 50.0, q=30.0)
r = rc.resample(rec, 250.0)
print("notch ok |", "resampled:", r.fs, r.n_samples)

# %% 19 -- block scaling policies on one array
from recurra.preprocess.scaling import scale_array

x = np.random.default_rng(18651).gamma(2.0, 1.0, 5000).reshape(-1, 1)
for policy in ("none", "zscore", "robust", "minmax", "rank"):
    y = scale_array(x, policy=policy)
    print(f"{policy:7s} std={y.std():7.3f}  min={y.min():7.3f}  max={y.max():7.3f}")

# %% 20 -- the quantity the default scaling equalises
from recurra.preprocess.scaling import rms_pairwise_distance, theoretical_rms_pairwise

phi = np.random.default_rng(19451).uniform(0, 2 * np.pi, 5000)
circle = np.column_stack([np.cos(phi), np.sin(phi)])
print("theoretical:", round(theoretical_rms_pairwise(circle), 4),
      "| sampled:", round(rms_pairwise_distance(circle, rng=0), 4),
      "| sqrt(2) =", round(np.sqrt(2), 4))

# %% 21 -- the Scaler directly, and the dominance warning
from recurra.preprocess.scaling import Scaler

amp = (np.random.default_rng(20351).gamma(2, 1, 5000) * 200).reshape(-1, 1)
for policy in ("none", "rms_balanced"):
    rep = Scaler(policy).fit([("phase", circle), ("amplitude", amp)])
    print(f"{policy:13s} share={np.round(rep.variance_share, 4)} "
          f"dominant={rep.dominant}")

# %% 22 -- lambda weighting: the split is exactly lambda
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    rep = Scaler("weighted", lambda_=lam).fit([("p", circle), ("a", amp)])
    print(f"lambda={lam:.2f} -> {np.round(rep.variance_share, 4)}")


# %% 23 -- generator: all five modalities
for modality in ("pac", "ppc", "aac", "ppa", "none"):
    s = rc.generate_cfc(modality=modality, duration=8.0, fs=400.0, seed=1)
    print(f"{modality:5s} data={s.data.shape} components={sorted(s.components)}")

# %% 24 -- generator: modulation shapes and preferred phase
for shape in ("hanning", "vonmises", "sine", "square"):
    s = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=0.9,
                        preferred_phase=np.pi / 2, modulation_shape=shape,
                        concentration=3.0, seed=2)
    mi = rc.modulation_index(s.components["phase_low"], s.components["amp_high"])
    print(f"{shape:9s} MI={mi:.4f}")

# %% 25 -- generator: distributed vs concentrated coupling
for pref, label in [(None, "distributed"), (np.pi / 2, "at pi/2")]:
    s = rc.generate_cfc(modality="pac", duration=30.0, fs=500.0, alpha=0.9,
                        preferred_phase=pref, seed=3)
    print(f"{label:12s} MVL={rc.mvl(s.components['phase_low'], s.components['amp_high']):.4f}")

# %% 26 -- generator: non-stationary coupling
for kind in ("transient", "ramp", "intermittent", "switching"):
    s = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=0.9,
                        nonstationarity=kind, onset=6.0, offset=14.0, seed=4)
    print(f"{kind:13s} constant alpha: {s.truth['alpha_profile_constant']}")

# %% 27 -- generator: an arbitrary alpha profile, as array or callable
n, fs = 10000, 500.0
profile = 0.5 * (1 + np.sin(2 * np.pi * 0.1 * np.arange(n) / fs))
s1 = rc.generate_cfc(modality="pac", duration=n / fs, fs=fs, alpha=profile, seed=5)
s2 = rc.generate_cfc(modality="pac", duration=20.0, fs=fs,
                     alpha=lambda t: np.clip(t / 20.0, 0, 1), seed=5)
print("array profile ok |", "callable profile ok")

# %% 28 -- generator: adverse scenarios
cases = {
    "clean":              dict(snr_db=25.0),
    "low SNR":            dict(snr_db=-3.0),
    "harmonic contamination": dict(snr_db=25.0, harmonic_contamination=1.5),
    "artefacts":          dict(snr_db=25.0, artifacts=["spike", "drift", "blink"]),
}
for label, kw in cases.items():
    s = rc.generate_cfc(modality="pac", duration=30.0, fs=500.0, alpha=0.7,
                        preferred_phase=np.pi / 2, seed=6, **kw)
    mi = rc.modulation_index(s.components["phase_low"], s.components["amp_high"])
    print(f"{label:24s} MI={mi:.4f}")

# %% 28b -- the harmonic artefact: coupling that is not there [DD-64, DD-65]
# A non-sinusoidal slow rhythm puts harmonics in the high band whose envelope
# is locked to the slow phase, so every classical index reports coupling.
print("uncoupled signal, rising harmonic contamination:")
for c in (0.0, 0.5, 1.0, 2.0):
    sg = rc.generate_cfc(modality="none", duration=30.0, fs=500.0, snr_db=20.0,
                         harmonic_contamination=c, seed=11)
    rg = rc.analytic(rc.filterbank(sg.to_recording("h"),
                                   {"theta": (4, 8), "gamma": (50, 70)}),
                     sources=["theta", "gamma"])
    print(f"  contamination {c:.1f}: MVL={rc.mvl(rg.get('phase_theta'), rg.get('amp_gamma')):.4f}"
          f"  MI={rc.modulation_index(rg.get('phase_theta'), rg.get('amp_gamma')):.5f}")
print("\nfor comparison, genuine coupling:")
for a in (0.2, 0.35, 0.65):
    sg = rc.generate_cfc(modality="pac", duration=30.0, fs=500.0, alpha=a,
                         preferred_phase=np.pi / 2, snr_db=20.0, seed=11)
    rg = rc.analytic(rc.filterbank(sg.to_recording("g"),
                                   {"theta": (4, 8), "gamma": (50, 70)}),
                     sources=["theta", "gamma"])
    print(f"  alpha {a:.2f}: MI={rc.modulation_index(rg.get('phase_theta'), rg.get('amp_gamma')):.5f}")
print("\nDD-65: at equal MI, DET is higher for genuine coupling and trapping")
print("time higher for the artefact -- modest effects, and only four of eight")
print("metrics show anything. See docs/DESIGN_DECISIONS.md.")

# %% 28c -- the picture that actually shows coupling [DD-106]
# A scatter of the joint space cannot: at alpha 0.3 the spread of amplitude
# within one phase bin is 4.4 times the movement of the mean across bins.
for _a in (0.0, 0.3, 0.9):
    _sg = rc.generate_cfc(modality="pac" if _a else "none", duration=40.0,
                          fs=500.0, alpha=_a, preferred_phase=np.pi / 2,
                          snr_db=20.0, seed=5)
    _rg = rc.analytic(rc.filterbank(_sg.to_recording("m"),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    _ph, _am = _rg.get("phase_theta"), _rg.get("amp_gamma")
    _prof = rc.modulogram(_ph, _am)
    _peak = _prof.loc[_prof.amplitude.idxmax(), "phase"]
    print(f"  alpha {_a:.1f}: MVL={rc.mvl(_ph, _am):.4f}  "
          f"contrast={_prof.amplitude.max() - _prof.amplitude.min():.5f}  "
          f"peak at {_peak:+.2f} rad")
rc.save_figure(rc.plot_modulogram(_prof), "cb_modulogram.png")

# %% 28d -- which band pair couples, rather than assuming one [DD-106]
_multi = rc.analytic(rc.filterbank(_sg.to_recording("c"),
                                   {"theta": (4, 8), "alpha": (8, 12),
                                    "gamma": (50, 70)}),
                     sources=["theta", "alpha", "gamma"])
_com = rc.comodulogram(_multi)
print(_com.round(5).to_string(index=False))
rc.save_figure(rc.plot_comodulogram(_com), "cb_comodulogram.png")
print("\ncoupling_indices, the reference every recurrence metric is read against:")
print({k: round(v, 5) for k, v in rc.coupling_indices(_ph, _am).items()})

# %% 29 -- generator: n:m phase locking [DD-48]
# n/m must match f_high / f_low, or the locked component falls outside the
# high band and no filter will find it. The generator warns when it does not.
for nm, f_low, f_high in [((10, 1), 6.0, 60.0), ((5, 1), 8.0, 40.0),
                          ((20, 3), 9.0, 60.0)]:
    s = rc.generate_cfc(modality="ppc", duration=40.0, fs=500.0, alpha=0.9,
                        nm_ratio=nm, f_low=f_low, f_high=f_high, seed=7)
    v = rc.plv(s.components["phase_low"], s.components["phase_high"],
               n=nm[0], m=nm[1])
    print(f"n:m = {nm[0]:2d}:{nm[1]}  f_high/f_low = {f_high / f_low:5.2f}  PLV={v:.4f}")

print("\nPLV rises monotonically with alpha:")
for a in (0.0, 0.25, 0.5, 0.75, 1.0):
    s = rc.generate_cfc(modality="ppc", duration=40.0, fs=500.0, alpha=a,
                        nm_ratio=(10, 1), f_low=6.0, f_high=60.0, seed=7)
    print(f"  alpha={a:.2f} PLV={rc.plv(s.components['phase_low'], s.components['phase_high'], n=10, m=1):.4f}")

print("\nan incoherent ratio is flagged:")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    rc.generate_cfc(modality="ppc", duration=10.0, fs=500.0, alpha=0.9,
                    nm_ratio=(1, 1), f_low=6.0, f_high=60.0, seed=7)
print(" ", [c.category.__name__ for c in caught if c.category is rc.ParameterWarning],
      str([c.message for c in caught if c.category is rc.ParameterWarning][0])[:110], "...")

# %% 30 -- coloured noise and reference chaotic systems
from recurra.signals import henon, mackey_glass, rossler, van_der_pol

for beta in (0.0, 1.0, 2.0):
    x = rc.colored_noise(4096, beta=beta, fs=250.0, rng=1)
    print(f"beta={beta}: std={x.std():.3f}")
print("lorenz", rc.lorenz(2000).shape, "| rossler", rossler(2000).shape,
      "| henon", henon(2000).shape, "| van der Pol", van_der_pol(2000).shape,
      "| Mackey-Glass", mackey_glass(2000).shape)

# %% 31 -- a parametric corpus, with ground truth
corpus = rc.generate_corpus({"alpha": [0.0, 0.5, 1.0], "snr_db": [20, 5]},
                            n_per_cell=2, duration=8.0, fs=400.0, seed=0)
truth = pd.DataFrame([s.truth_row() for s in corpus])
print(len(corpus), "signals")
print(truth[["subject", "modality", "alpha_mean", "snr_db", "cell"]].head().to_string(index=False))

# %% 32 -- canonical coupling indices
ph, am = SIG.components["phase_low"], SIG.components["amp_high"]
print(rc.coupling_indices(ph, am))
print("MVL       :", round(rc.mvl(ph, am), 4))
print("MI (Tort) :", round(rc.modulation_index(ph, am, n_bins=18), 4))

# %% 33 -- significance against surrogates
from recurra.metrics.classic import surrogate_significance

for method in ("circular_shift", "shuffle"):
    res = surrogate_significance(rc.mvl, ph, am, n_surrogates=100,
                                 method=method, rng=0)
    print(f"{method:15s} observed={res['observed']:.4f} p={res['p_value']:.4f} "
          f"z={res['z_score']:.2f}")


# %% 34 -- state space: observable route, every constructor
_, rec_ppa = make_signal(seed=20)
rec_ppa = rc.analytic(rc.filterbank(SIG.to_recording("ppa"),
                                    {"delta": (2, 4), "theta": (8, 12),
                                     "gamma": (50, 70)}),
                      sources=["delta", "theta", "gamma"])
builders = {
    "phase_circle":   sm.phase_circle(REC, "phase_theta"),
    "pac_space":      sm.pac_space(REC),
    "ppc_space":      sm.ppc_space(REC, "phase_theta", "phase_gamma"),
    "envelope_space": sm.envelope_space(REC),
    "ppa_space":      sm.ppa_space(rec_ppa, "phase_delta", "phase_theta", "amp_gamma"),
    "freq_amp_space": sm.freq_amp_space(REC, "ifreq_theta", "amp_gamma"),
}
for name, space in builders.items():
    print(f"{name:15s} dim={space.dim} route={space.route} blocks={space.block_labels}")

# %% 35 -- state space: nested hierarchies, and a plain multivariate space
nested = sm.nested_space(rec_ppa, [
    {"source": "phase_delta", "as": "phase_circle", "label": "delta"},
    {"source": "amp_theta", "as": "amplitude", "label": "theta"},
    {"source": "amp_gamma", "as": "amplitude", "label": "gamma"}])
lorenz_rec = rc.ingest(rc.lorenz(4000), fs=100.0, labels=["x", "y", "z"],
                       subject="lorenz")
plain = sm.channels(lorenz_rec)          # no coupling assumed at all
print("nested:", nested.dim, nested.block_labels, "| channels:", plain.dim)

# %% 36 -- state space: delay route, fixed and automatic
fixed = sm.takens(lorenz_rec, "x", m=3, tau=16)
auto = sm.takens(lorenz_rec, "x", m="auto", tau="auto", rng=0)
multi = sm.takens_multivariate(lorenz_rec, ["x", "y"], m=2, tau=16)
print(f"fixed dim={fixed.dim} | auto dim={auto.dim} | multivariate dim={multi.dim}")

# %% 37 -- state space: hybrid route
h1 = sm.pac_space(REC, embed_amplitude=3, tau=8)
h2 = sm.envelope_space(REC, embed=2)
h3 = sm.build(REC, [{"source": "phase_theta", "as": "phase_circle"},
                    {"source": "amp_gamma", "as": "delay", "m": 3, "tau": 8}],
              route="hybrid")
for lbl, s in [("pac+embed", h1), ("envelopes+embed", h2), ("build custom", h3)]:
    print(f"{lbl:16s} route={s.route} dim={s.dim} kinds={[g.kind for g in s.groups]}")

# %% 38 -- state space: declarative, straight from the raw signal
declared = sm.from_bands(SIG.to_recording("decl"), [
    {"band": (4, 8), "as": "phase", "label": "theta"},
    {"band": (50, 70), "as": "amplitude", "label": "gamma", "smooth": 8.0}])
print(declared.summary())

# %% 39 -- state space: the escape hatch
custom = sm.custom(REC, lambda r: np.column_stack([
    np.cos(r.get("phase_theta")), np.sin(r.get("phase_theta")),
    np.log1p(r.get("amp_gamma"))]), scaling="rms_balanced")
print(custom.route, custom.dim)

# %% 40 -- embedding parameters: every method combination
X = rc.lorenz(6000, dt=0.01, transient=10.0)
for tau_method in ("ami", "acf", "first_zero"):
    for m_method in ("fnn", "cao"):
        p = sm.estimate_embedding(X[:, 0], tau_method=tau_method, m_method=m_method,
                                  tau_range=(1, 200), m_range=(1, 10), rng=0)
        print(f"{tau_method:11s}/{m_method:4s} tau={p.tau:4d} m={p.m}  "
              f"({p.diagnostics['m_criterion']})")

# %% 41 -- embedding across records: aggregate, never concatenate
records = [rc.lorenz(3000, dt=0.01, transient=5.0)[:, 0] for _ in range(3)]
combined, per_record = sm.estimate_embedding_multi(
    records, aggregate="median", tau_range=(1, 100), m_range=(1, 8), rng=0)
print(combined, "\n", per_record.to_string(index=False))

# %% 42 -- state space: every scaling policy on the same space
raw = sm.pac_space(REC, scaling="none")
policies = {
    "none":          raw,
    "rms_balanced":  raw.rescale("rms_balanced"),
    "lambda=0.0":    raw.rescale("weighted", lambda_=0.0),
    "lambda=0.5":    raw.rescale("weighted", lambda_=0.5),
    "lambda=1.0":    raw.rescale("weighted", lambda_=1.0),
    "custom":        raw.rescale("custom", weights=[1.5, 0.5]),
}
for label, s in policies.items():
    sf = s.scale_frame()
    print(f"{label:13s} weights={np.round(sf.weight.values, 4)} "
          f"share={np.round(sf.variance_share.values, 3)}")

# %% 43 -- the shape diagnostic that variance balancing cannot supply
sf = sm.pac_space(REC).scale_frame()
print(sf[["block", "rms_pairwise", "weight", "variance_share",
          "tail_ratio", "skewness"]].round(4).to_string(index=False))
print("\nwarning:", sm.pac_space(REC).scale_report.shape_warning)

# %% 44 -- per-block pre-scaling and envelope smoothing
variants = {
    "plain":            sm.pac_space(REC),
    "smoothed 8 Hz":    sm.pac_space(REC, smooth=8.0),
    "robust amplitude": sm.pac_space(REC, scale_amplitude="robust"),
    "ranked amplitude": sm.pac_space(REC, scale_amplitude="rank"),
}
for label, s in variants.items():
    print(f"{label:18s} tail_ratio={np.round(s.scale_frame().tail_ratio.values, 3)}")

# %% 45 -- state space transforms
base = sm.pac_space(REC, smooth=8.0)
print("original      :", base.n_points, base.dim)
print("subsample(4)  :", base.subsample(4).n_points)
print("max_points    :", base.subsample(max_points=1000).n_points)
print("window        :", base.window(500, 2500).n_points)
print("split         :", base.split(["phase_theta"]).block_labels)
print("reduce to PCA2:", base.reduce("pca", 2).dim)

# %% 46 -- comparing routes empirically
df_routes, spaces = sm.compare_routes(REC, rng=0)
cols = ["route_name", "route", "dim", "n_points", "nn_mean", "nn_cv", "corr_slope"]
print(df_routes[[c for c in cols if c in df_routes]].round(4).to_string(index=False))

# %% 47 -- lambda sweep and neighbour agreement
sweep = sm.lambda_sweep(sm.pac_space(REC, scaling="none"),
                        grid=np.linspace(0, 1, 6), rng=0, max_points=800)
print(sweep[["lambda", "nn_cv", "corr_slope",
             "nn_agreement_vs_block0"]].round(4).to_string(index=False))
print("agreement of a space with itself:",
      round(sm.neighbour_agreement(SS, SS, rng=0), 4))

# %% 48 -- geometry descriptors on their own
print(sm.geometry_descriptors(SS, rng=0))


# %% 49 -- thresholds: every mode
modes = [
    ("target_rr 5%",      dict(mode="target_rr", target_rr=0.05)),
    ("target_rr bisect",  dict(mode="target_rr", target_rr=0.05, method="bisect")),
    ("percentile 0.05",   dict(mode="percentile", percentile=0.05)),
    ("fan k=20",          dict(mode="fan", n_neighbors=20)),
    ("std_fraction 0.1",  dict(mode="std_fraction", factor=0.1)),
    ("maxdist 0.2",       dict(mode="maxdist_fraction", factor=0.2)),
    ("per_block 5%",      dict(mode="per_block", target_rr=0.05)),
    ("adaptive_dim 5%",   dict(mode="adaptive_dim", target_rr=0.05)),
    ("fixed 0.45",        dict(mode="fixed", value=0.45)),
]
rows = []
for label, kw in modes:
    th = rc.threshold(SS, theiler=1, rng=0, **kw)
    rows.append({"label": label, **th.to_frame().iloc[0].to_dict()})
print(pd.DataFrame(rows)[["label", "mode", "epsilon", "target_rr", "achieved_rr",
                          "pointwise", "per_block"]].round(5).to_string(index=False))

# %% 50 -- threshold detail, and its diagnostics
th = rc.threshold(SS, mode="target_rr", target_rr=0.05, method="bisect",
                  theiler=1, rng=0)
print(th.summary())
print("\ndiagnostics:", th.diagnostics)

# %% 51 -- FAN is per point, and says so
th_fan = rc.threshold(SS, mode="fan", n_neighbors=20, theiler=1, rng=0)
print("pointwise:", th_fan.is_pointwise, "| epsilon shape:", th_fan.value.shape)
print("warning:", th_fan.diagnostics["warning"])

# %% 52 -- per-block thresholds need no relative weighting at all
th_pb = rc.threshold(SS, mode="per_block", target_rr=0.05, rng=0)
print("per block:", dict(zip(th_pb.block_labels, np.round(th_pb.per_block, 5), strict=False)))

# %% 53 -- a percentage where a fraction was wanted is refused
try:
    rc.threshold(SS, mode="percentile", percentile=5.0, rng=0)
except rc.ParameterError as e:
    print("ParameterError:", e)

# %% 54 -- how accurately the target rate is met
rows = []
for target in (0.01, 0.05, 0.10, 0.20):
    for method in ("quantile", "bisect"):
        t = rc.threshold(SS, target_rr=target, method=method, theiler=1, rng=0)
        rm = rc.recurrence_plot(SS, t, theiler=1)
        rows.append({"target": target, "method": method, "epsilon": t.scalar,
                     "actual_rr": rm.recurrence_rate(),
                     "error": abs(rm.recurrence_rate() - target)})
print(pd.DataFrame(rows).round(5).to_string(index=False))

# %% 55 -- the Theiler window matters to the estimate
for theiler in (0, 1, 10, 50):
    t = rc.threshold(SS, target_rr=0.05, theiler=theiler, rng=0)
    print(f"theiler={theiler:3d} epsilon={t.scalar:.5f}")

# %% 56 -- thresholds under other distance metrics
for metric in ("euclidean", "chebyshev", "manhattan", "cosine"):
    t = rc.threshold(SS, target_rr=0.05, metric=metric, theiler=1, rng=0)
    print(f"{metric:10s} epsilon={t.scalar:.5f}")

# %% 57 -- sampling pairwise distances directly
d = rc.threshold_sampling.sample_pair_distances(SS.weighted_coords,
                                                n_samples=50_000, theiler=1, rng=0)
print(f"{d.size} sampled distances, mean {d.mean():.4f}, "
      f"5th percentile {np.quantile(d, 0.05):.4f}")

# %% 58 -- a cross threshold, with no Theiler exclusion
c1 = sm.phase_circle(REC, "phase_theta").subsample(max_points=1200)
c2 = sm.phase_circle(REC, "phase_gamma").subsample(max_points=1200)
th_x = rc.cross_threshold(c1, c2, target_rr=0.05, rng=0)
print(th_x.summary())


# %% 59 -- a recurrence plot, four ways to give the threshold
a = rc.recurrence_plot(SS, target_rr=0.05, theiler=1, rng=0)   # keywords
b = rc.recurrence_plot(SS, th, theiler=1)                      # a Threshold
c = rc.recurrence_plot(SS, 0.42, theiler=1)                    # a fixed number
d_ = rc.recurrence_plot(SS, "fan", n_neighbors=20, theiler=1, rng=0)  # a mode name
for lbl, rm in [("keywords", a), ("Threshold", b), ("number", c), ("mode name", d_)]:
    print(f"{lbl:10s} eps={rm.threshold.scalar:.4f} RR={rm.recurrence_rate():.4%}")

# %% 60 -- the summary and the exportable description
rm = rc.recurrence_plot(SS, target_rr=0.05, theiler=1, rng=0)
print(rm.summary())
print()
print(rm.describe().T.to_string())

# %% 61 -- the Theiler convention: 0 excludes nothing
for theiler in (0, 1, 5):
    M = rc.recurrence_plot(SS, 0.5, theiler=theiler).matrix
    print(f"theiler={theiler}: main diagonal sum = {np.diag(M).sum():5d}, "
          f"diagonal +1 sum = {np.diag(M, 1).sum()}")

# %% 62 -- storage strategies, all equivalent
th5 = rc.threshold(SS, target_rr=0.05, theiler=1, rng=0)
dense = rc.recurrence_plot(SS, th5, theiler=1, store="memory")
sparse = rc.recurrence_plot(SS, th5, theiler=1, store="sparse")
stream = rc.recurrence_plot(SS, th5, theiler=1, store="none", tile_size=97)
rebuilt = np.zeros(dense.shape, dtype=np.uint8)
for tile in stream.tiles():
    rebuilt[tile.i0:tile.i1, tile.j0:tile.j1] = tile.data
print("sparse == dense :", np.array_equal(sparse.to_sparse().toarray().astype(np.uint8),
                                          dense.matrix))
print("stream == dense :", np.array_equal(rebuilt, dense.matrix))
print("rates equal     :", dense.recurrence_rate() == stream.recurrence_rate())

# %% 63 -- iterating tiles without ever building the matrix
lazy = rc.recurrence_plot(SS, th5, theiler=1, store="none", tile_size=256)
n_tiles = sum(1 for _ in lazy.tiles())
print(f"{n_tiles} tiles, matrix never allocated; RR = {lazy.recurrence_rate():.4%}")

# %% 64 -- memory planning for records of increasing length
rows = []
for n in (5_000, 50_000, 200_000, 1_800_000):
    p = rc.recurrence.plan(n, n, 3, dtype="uint8", budget_gb=8.0, estimated_rr=0.05)
    rows.append({"n_points": n, "seconds_at_500Hz": round(n / 500, 1),
                 "dense_gb": round(p.matrix_gb, 3), "sparse_gb": round(p.sparse_gb, 3),
                 "store": p.store, "tile": p.tile_size})
print(pd.DataFrame(rows).to_string(index=False))

# %% 65 -- the budget refuses rather than thrashes
try:
    rc.recurrence.plan(2_000_000, 2_000_000, 3, store="memory", budget_gb=1.0)
except rc.MemoryBudgetError as e:
    print("MemoryBudgetError:", str(e)[:150], "...")

# %% 66 -- the density profile, streamed
prof = rm.density_profile(n_bins=12)
print(prof[["bin", "start_time_s", "density"]].round(4).to_string(index=False))

# %% 67 -- cross recurrence: rectangular and asymmetric
crp = rc.cross_recurrence_plot(c1, c2, target_rr=0.05, rng=0)
print(crp.summary())
print("symmetric:", crp.is_symmetric, "| shape:", crp.shape)

# %% 68 -- cross recurrence refuses mismatched dimensionality
try:
    rc.cross_recurrence_plot(SS, c1, target_rr=0.05, rng=0)
except rc.ParameterError as e:
    print("ParameterError:", e)

# %% 69 -- joint recurrence: independent thresholds, different dimensions
jrp = rc.joint_recurrence_plot([SS.split(["phase_theta"]), SS.split(["amp_gamma"])],
                               target_rr=0.10, theiler=1, rng=0)
print("subsystem dims  :", [p.dim for p in jrp.parts])
print("epsilons        :", [round(t.scalar, 5) for t in jrp.thresholds])
print("subsystem rates :", [f"{r:.3%}" for r in jrp.subsystem_rates()])
print("joint rate      :", f"{jrp.recurrence_rate():.3%}")
print("independence    :", round(jrp.independence_ratio(), 4))

# %% 70 -- joint recurrence discriminates coupling
for alpha, label in [(0.0, "uncoupled"), (0.4, "weak"), (0.9, "strong")]:
    _, r = make_signal(alpha=alpha, seed=40)
    s = sm.pac_space(r, smooth=8.0).subsample(max_points=1200)
    j = rc.joint_recurrence_plot([s.split(["phase_theta"]), s.split(["amp_gamma"])],
                                 target_rr=0.10, theiler=1, rng=0)
    print(f"alpha={alpha:.1f} ({label:9s}) independence ratio = "
          f"{j.independence_ratio():.4f}")


# %% 70b -- RQA: quantifying a recurrence structure
m = rc.rqa(rm)
for k, v in m.items():
    print(f"  {k:10s} {v:.5f}")

# %% 70b2 -- RQA: choosing the line-length floor from the data [DD-98]
print("DET and LAM at a range of floors, all from one histogram:")
print(rc.line_length_sweep(rm.line_histogram()).round(4).to_string(index=False))
print("\ndefaults are l_min=8, v_min=2 -- not the classical 2 and 2, because")
print("a DET of 0.996 with a spread of 0.0005 cannot separate anything.")

# %% 70c -- RQA: the histograms every metric comes from
metrics, hist = rc.rqa(rm, return_histogram=True)
print(hist.summary())
print(hist.to_frame("diagonal").head().to_string(index=False))
rc.save_figure(rc.plot_line_histogram(hist), "cb_line_histogram.png")

# %% 70d -- RQA: options
print("l_min=2 :", round(rc.rqa(rm, l_min=2)["DET"], 4))
print("l_min=5 :", round(rc.rqa(rm, l_min=5)["DET"], 4))
print("denominator theiler_corrected:",
      round(rc.rqa(rm, denominator="theiler_corrected")["DET"], 5))
print("denominator all              :",
      round(rc.rqa(rm, denominator="all")["DET"], 5))
print("selected metrics:", rc.rqa(rm, metrics=["RR", "DET", "ENTR"]))
print(rc.rqa(rm, as_frame=True).T.to_string())

# %% 70e -- RQA: ground truth on three reference systems
for name, data in [("white noise", np.random.default_rng(0).standard_normal((1500, 3))),
                   ("Lorenz", rc.lorenz(1500, dt=0.01, transient=10.0)),
                   ("sine", np.column_stack([np.sin(2*np.pi*5*np.arange(1500)/200),
                                             np.cos(2*np.pi*5*np.arange(1500)/200)]))]:
    q = rc.rqa(data, target_rr=0.05, theiler=1, rng=0)
    print(f"  {name:12s} DET={q['DET']:.4f}  L={q['L']:6.2f}  "
          f"Lmax={q['Lmax']:7.0f}  ENTR={q['ENTR']:.3f}")

# %% 70f -- CRQA: lag and direction from a cross recurrence plot
Xw = np.cumsum(np.random.default_rng(0).standard_normal((900, 2)), axis=0)
crp_lag = rc.cross_recurrence_plot(Xw[40:], Xw[:-40], target_rr=0.05, rng=0)
out = rc.crqa(crp_lag)
print(f"  peak offset {out['peak_offset']} (the true shift is 40), "
      f"asymmetry {out['profile_asymmetry']:+.3f}")
prof = rc.diagonal_profile(crp_lag)
rc.save_figure(rc.plot_diagonal_profile(prof, in_seconds=False),
               "cb_diagonal_profile.png")

# %% 70g -- invariants: correlation dimension of a known attractor
L_space = sm.channels(rc.ingest(rc.lorenz(10000, dt=0.01, transient=20.0),
                                fs=100.0, labels=["x", "y", "z"]))
d2 = rc.correlation_dimension(L_space, theiler=20, max_points=4000, rng=0)
print(d2.summary())
print("\n  reference for the CORRELATION dimension of Lorenz: 2.05")
print("  (2.06 is the Kaplan-Yorke dimension, a different quantity)")

# %% 70h -- invariants: the raw curve, and fitting a window of your own
curve = rc.correlation_sum(L_space, theiler=20, max_points=3000, rng=0)
print(curve.head().round(4).to_string(index=False))
lo, hi = curve["log_r"].quantile([0.25, 0.75])
print("user region:", rc.correlation_dimension(L_space, theiler=20,
                                               max_points=3000, region=(lo, hi),
                                               rng=0).value.__round__(4))

# %% 70i -- invariants: the largest Lyapunov exponent, with its units [DD-62]
l1 = rc.lyapunov_max(L_space, max_steps=150, theiler=50, max_points=3000, rng=0)
print(l1.summary())
per_sample = rc.lyapunov_max(L_space, max_steps=150, theiler=50,
                             max_points=3000, units="per_sample", rng=0)
print(f"\n  per second {l1.value:.4f}  =  per sample {per_sample.value:.6f} "
      f"x fs {L_space.fs:g}")
print("  reference for Lorenz: 0.906 per second")

# %% 70j -- invariants: the divergence curve behind it
div = rc.divergence_curve(L_space, max_steps=150, theiler=50, max_points=2000,
                          rng=0)
print(div.iloc[::20].round(4).to_string(index=False))
print("\n  the steep first stretch is the transient; the linear part follows")

# %% 70k -- invariants: the guards that fire on data they cannot handle [DD-63]
tt = np.arange(6000) / 200.0
clean_circle = sm.channels(rc.ingest(
    np.column_stack([np.sin(2*np.pi*3*tt), np.cos(2*np.pi*3*tt)]),
    fs=200.0, labels=["a", "b"]))
bad = rc.lyapunov_max(clean_circle, max_steps=200, theiler=100,
                      max_points=1500, rng=0)
print("noiseless circle ->", bad.warning)
print("\nLorenz          -> warning:", rc.lyapunov_max(
    L_space, max_steps=150, theiler=50, max_points=2000, rng=0).warning)

# %% 70l -- invariants: K2 from the diagonal line distribution
k2 = rc.k2_entropy(L_space.subsample(max_points=2500), target_rr=0.05,
                   theiler=20, rng=0)
print(k2.summary())
print(f"  distinct diagonal lengths used: {k2.diagnostics['n_distinct_lengths']}")

# %% 70m -- complexity: scaling of the fluctuations
for beta, ref in ((0.0, 0.5), (1.0, 1.0), (2.0, 1.5)):
    x_noise = rc.colored_noise(8192, beta=beta, rng=1)
    print(f"  beta={beta}: DFA alpha = {rc.detrended_fluctuation(x_noise).value:.3f}"
          f"   Hurst = {rc.hurst_exponent(x_noise).value:.3f}   (reference {ref})")

# %% 70n -- complexity: irregularity of a series
tt = np.arange(4000) / 200.0
series = {"white noise": np.random.default_rng(0).standard_normal(4000),
          "sine": np.sin(2 * np.pi * 5 * tt),
          "Lorenz x": rc.lorenz(4000, dt=0.01, transient=10.0)[:, 0]}
print(f"{'series':>12} {'Higuchi':>9} {'perm ent':>10} {'samp ent':>10} {'spec ent':>10}")
for name, v in series.items():
    print(f"{name:>12} {rc.higuchi_dimension(v).value:9.3f} "
          f"{rc.permutation_entropy(v).value:10.4f} "
          f"{rc.sample_entropy(v).value:10.4f} "
          f"{rc.spectral_entropy(v, fs=200.0).value:10.4f}")

# %% 70n2 -- complexity: the five that complete the standard set
tt2 = np.arange(4000) / 200.0
extra = {"white noise": np.random.default_rng(0).standard_normal(4000),
         "sine": np.sin(2 * np.pi * 5 * tt2)}
print(f"{'series':>12} {'Katz':>7} {'Petros':>8} {'SVDent':>8} {'LZ':>7} {'mobility':>9}")
for name, v in extra.items():
    h = rc.hjorth_parameters(v, fs=200.0)
    print(f"{name:>12} {rc.katz_dimension(v).value:7.3f} "
          f"{rc.petrosian_dimension(v).value:8.4f} {rc.svd_entropy(v).value:8.4f} "
          f"{rc.lempel_ziv_complexity(v).value:7.4f} "
          f"{h['hjorth_mobility'].value:9.4f}")
print(f"\nHjorth mobility of the 5 Hz sine should be 2*pi*f/fs = "
      f"{2 * np.pi * 5 / 200:.4f}")

# %% 70n2b -- fine-grained measures need their own scale [DD-99]
print("samples between turns:  white noise "
      f"{rc.samples_per_turn(np.random.default_rng(0).standard_normal(5000)):.2f}"
      f"   this envelope {rc.samples_per_turn(np.asarray(SS.block('amp_gamma')).ravel()):.2f}")
_flat = rc.dynamics_measures(SS, measures=list(rc.FINE_SCALE_MEASURES),
                             fine_scale=None)
_auto = rc.dynamics_measures(SS, measures=list(rc.FINE_SCALE_MEASURES))
print(f"{'measure':>32} {'as sampled':>11} {'own scale':>10}")
for _k in sorted(k for k in _flat if k.endswith("amp_gamma")):
    print(f"  {_k:>30} {_flat[_k]:11.4f} {_auto[_k]:10.4f}")
print(f"  decimation used: {_auto['fine_scale_amp_gamma']:.0f}"
      f"  (rc.fine_scale_factor gives "
      f"{rc.fine_scale_factor(np.asarray(SS.block('amp_gamma')).ravel())} directly)")

# %% 70n3 -- dynamical measures as a table row [DD-71, DD-72]
print("available:", rc.SCALAR_MEASURES)
print("expensive:", rc.TRAJECTORY_MEASURES)
row = rc.dynamics_measures(SS)
for k, v in row.items():
    print(f"  {k:34s} {v:8.4f}")

# %% 70n4 -- invariants per block as well as per trajectory [DD-93]
print("what can be estimated from a single block:", rc.BLOCK_INVARIANTS)
_blocks = rc.dynamics_measures(SS, measures=["higuchi_fd"],
                               block_invariants=True, max_points=1200, rng=0)
for k, v in _blocks.items():
    print(f"  {k:26s} {v:9.4f}")
print("  these describe each signal on its own; DD-77 says why that is not")
print("  a coupling measure however well it separates.")

# %% 70n5 -- the plot as a picture, for a classifier that takes images
_img, _info = rc.recurrence_image(SS, size=128, target_rr=0.05, theiler=1, rng=0)
print(f"density map {_img.shape}, mean {_img.mean():.4f} "
      f"(= the recurrence rate {_info['recurrence_rate']:.4f})")
print(f"  {_info['cells_per_pixel']} cells behind every pixel, pooling "
      f"{_info['pooling']}")
_meta = rc.export_recurrence_images({"one": SS, "two": SS}, "cookbook_out/rp_images",
                                    size=64, labels={"one": "a", "two": "b"},
                                    fmt="npy", target_rr=0.05, theiler=1, rng=0)
print(_meta[["name", "label", "image_size", "vmin", "vmax", "vmax_kind"]]
      .to_string(index=False))

# %% 70n6 -- invariants from a delay embedding of a signal [DD-96, DD-97]
_lorenz_x = rc.lorenz(12000, dt=0.01, transient=20.0)[:, 0]
_inv = rc.invariants_from_series(_lorenz_x, fs=100.0)
print("Lorenz x, references D2 = 2.05 and lambda_1 = 0.906 per second:")
for k, v in _inv.items():
    print(f"  {k:16s} {v:9.4f}")
print("  a Theiler window of 1 gave D2 = 1.60 here; 'auto' derives it from the")
print("  embedding span and the record length [DD-96].")

# %% 70n6 -- the shape of each block, and why it matters [DD-102]
print("tail ratio per block:", {k: round(v, 2) for k, v in
                                rc.shape_descriptors(SS).items()
                                if k.startswith("tail_ratio")})
print("  a phase circle sits near 1.4; a real gamma envelope 3 to 12.")
print("  It drove 22 of 39 metrics with coupling held constant -- record it")
print("  per subject and adjust for it before believing a group difference.")

# %% 70n7 -- the attractor as an image [DD-103]
_att, _ainfo = rc.attractor_image(SS, size=128, coords=(0, 2))
print(f"occupancy map {_att.shape}, {_ainfo['occupied_fraction']:.1%} of pixels "
      f"visited, peak count {_ainfo['peak_count']:.0f}")
_am = rc.export_attractor_images({"one": SS}, "cookbook_out/attractors",
                                 size=64, fmt="npy", coords=(0, 2))
print(_am[["name", "coords", "kind", "image_size", "vmax_kind"]].to_string(index=False))

# %% 70n8 -- attractor figures that can be read side by side [DD-104]
_pair = {"one": SS, "two": SS.subsample(2)}
_lims = rc.shared_limits(_pair)
print("shared limits:", [tuple(round(v, 3) for v in l) for l in _lims])
rc.save_figure(rc.plot_attractor(SS, mode="3d", axis_limits=_lims,
                                 title="one - shared axes"), "cb_attractor.png")
rc.save_figure(rc.compare_attractors(_pair, mode="3d", share_limits=True),
               "cb_attractors_panel.png")
for _fn in (rc.compare_recurrence, rc.compare_routes, rc.compare_thresholds,
            rc.compare_series, rc.compare_window_series,
            rc.compare_scale_policies, rc.compare_phase_amplitude):
    assert callable(_fn)
print("every comparison figure is reachable at the top level")

# %% 70o -- invariants: the Invariant object itself
print(d2.to_frame().T.to_string())
print("\nisinstance check:", isinstance(d2, rc.Invariant), "| float():", float(d2))

# %% 71 -- windows: the three ways to declare them
for label, spec in [
        ("10 windows, 50% overlap", rc.WindowSpec(n_windows=10, overlap=0.5)),
        ("4 s windows, 50% overlap", rc.WindowSpec(size=4.0, overlap=0.5)),
        ("4 s windows, 0.5 s step", rc.WindowSpec(size=4.0, step=0.5)),
        ("2000 samples, step 1000", rc.WindowSpec(size=2000, step=1000, unit="samples"))]:
    print(f"{label:26s} -> {spec.summary(20000, 500.0).splitlines()[1].strip()}")

# %% 72 -- the resolved grid, and what it leaves out
spec = rc.WindowSpec(size=7.0, step=7.0)
print(spec.describe(20000, 500.0)[["window", "start_sample", "stop_sample",
                                   "t_start_s", "t_stop_s",
                                   "overlap_with_previous"]].to_string(index=False))
print("\ncaveats:", spec.resolve(20000, 500.0)[1])

# %% 73 -- ten overlapping recurrence plots from one record
_, rec_long = make_signal(duration=60.0, seed=17)
ss_long = sm.pac_space(rec_long, smooth=8.0).subsample(max_points=9000)
wr = rc.windowed_recurrence(ss_long, rc.WindowSpec(n_windows=10, overlap=0.5),
                            target_rr=0.05, theiler=1, rng=0)
print(wr.summary())
print("\nwith RQA and dynamical columns side by side:")
_both = wr.metrics(rqa=True, dynamics=True)
print(_both[["window", "DET", "LAM", "higuchi_fd_amp_gamma",
             "permutation_entropy_amp_gamma"]].round(4).to_string(index=False))
print("\nwith RQA columns:")
print(wr.metrics(rqa=True)[["window", "RR", "DET", "L", "ENTR", "LAM"]].round(4).to_string(index=False))
print()
print(wr.metrics()[["window", "t_start_s", "t_stop_s", "n_rows", "epsilon",
                    "recurrence_rate"]].round(4).to_string(index=False))

# %% 74 -- threshold scope: which quantity carries information
for scope in ("per_window", "global"):
    w = rc.windowed_recurrence(ss_long, 8, scope=scope, target_rr=0.05,
                               theiler=1, rng=0)
    m = w.metrics()
    print(f"{scope:11s} epsilon std={m.epsilon.std():.5f}  "
          f"RR std={m.recurrence_rate.std():.5f}  "
          f"RR range={m.recurrence_rate.min():.4f}-{m.recurrence_rate.max():.4f}")

# %% 75 -- windowing works with every threshold mode
for label, kw in [("target_rr", dict(target_rr=0.05)),
                  ("fan", dict(threshold="fan", n_neighbors=20)),
                  ("per_block", dict(threshold="per_block", target_rr=0.05)),
                  ("fixed", dict(threshold=0.45)),
                  ("chebyshev", dict(metric="chebyshev", target_rr=0.05))]:
    w = rc.windowed_recurrence(ss_long, 6, theiler=1, rng=0, **kw)
    print(f"{label:10s} mean RR={w.metrics().recurrence_rate.mean():.4f}")

# %% 76 -- windowed cross and joint recurrence
c1l = sm.phase_circle(rec_long, "phase_theta").subsample(max_points=9000)
c2l = sm.phase_circle(rec_long, "phase_gamma").subsample(max_points=9000)
wc = rc.windowed_cross_recurrence(c1l, c2l, 6, target_rr=0.05, rng=0)
wj = rc.windowed_joint_recurrence([ss_long.split(["phase_theta"]),
                                   ss_long.split(["amp_gamma"])],
                                  6, target_rr=0.10, theiler=1, rng=0)
print("CRP windows:", len(wc), "| JRP windows:", len(wj))
print(wj.metrics()[["window", "rr_subsystem_0", "rr_subsystem_1",
                    "independence_ratio"]].round(4).to_string(index=False))

# %% 77 -- accessing individual windows, and caching them
wr_keep = rc.windowed_recurrence(ss_long, 6, target_rr=0.05, theiler=1,
                                 rng=0, keep=True)
print("window 3 shape:", wr_keep[3].shape)
print("cached:", wr_keep[3] is wr_keep[3])
for window, matrix in wr_keep:
    print(f"  {window.label}: {window.t_start:6.2f}-{window.t_stop:6.2f} s, "
          f"RR={matrix.recurrence_rate():.4f}")

# %% 78 -- non-stationary coupling traced through the windows
series = {}
for kind in ("transient", "ramp", None):
    s = rc.generate_cfc(modality="pac", duration=40.0, fs=500.0, alpha=0.95,
                        preferred_phase=np.pi / 2, snr_db=20.0,
                        nonstationarity=kind, seed=31)
    r = rc.analytic(rc.filterbank(s.to_recording("ns"),
                                  {"theta": (4, 8), "gamma": (50, 70)}),
                    sources=["theta", "gamma"])
    sp = sm.pac_space(r, smooth=8.0).subsample(max_points=6000)
    series[kind or "stationary"] = rc.windowed_joint_recurrence(
        [sp.split(["phase_theta"]), sp.split(["amp_gamma"])],
        rc.WindowSpec(size=4.0, overlap=0.75), target_rr=0.10, theiler=1, rng=0)
for label, w in series.items():
    m = w.metrics().independence_ratio
    print(f"{label:11s} mean={m.mean():.4f} min={m.min():.4f} max={m.max():.4f}")

# %% 79 -- a corpus: subjects times windows
subjects = {}
groups = {}
for i in range(8):
    grp = "control" if i < 4 else "study"
    alpha = 0.1 if grp == "control" else 0.8
    _, r = make_signal(alpha=alpha, duration=30.0, seed=100 + i)
    subjects[f"sub{i:02d}"] = sm.pac_space(r, smooth=8.0).subsample(max_points=4000)
    groups[f"sub{i:02d}"] = grp
batch = rc.batch_windowed_recurrence(subjects, rc.WindowSpec(size=4.0, overlap=0.5),
                                     target_rr=0.05, theiler=1, rng=0)
print(batch.summary())
print("\ncomparable:", batch.report.ok, "| structures:", batch.n_matrices)


# %% 80 -- comparability: the report
print(batch.report.summary())

# %% 81 -- comparability: an explicit contract
contract = rc.contract_from(batch["sub00"],
                            axes=["dim", "fs", "n_points", "metric", "theiler",
                                  "threshold_mode", "target_rr", "scaling",
                                  "window_samples", "window_step"])
print(contract.summary())
print("\nsatisfied:", batch.check_comparability(contract).ok)

# %% 82 -- comparability: a violated contract names the offenders
bad = batch.check_comparability(rc.ComparabilityContract(dim=99))
print("ok:", bad.ok)
for axis in bad.violations:
    print(f"  {axis.axis}: required {axis.required}, offenders {axis.offenders[:3]}")

# %% 83 -- what the matching licenses
print(batch.report.metric_validity().to_string(index=False))

# %% 84 -- the raw parameter table behind the verdict
print(rc.describe_units(batch)[["label", "dim", "fs", "n_points", "metric",
                                "theiler", "window_samples"]].to_string(index=False))

# %% 85 -- separating two groups, window by window
gc = batch.group_comparison(groups, values=["recurrence_rate", "epsilon"])
print(gc[["window", "metric", "n_a", "n_b", "mean_a", "mean_b", "cohens_d",
          "auc", "p_value", "p_bonferroni"]].round(4).head(10).to_string(index=False))

# %% 86 -- ranking metrics by how well they separate
print(rc.rank_metrics(gc).round(4).to_string(index=False))

# %% 87 -- the feature matrix a classifier consumes
X = batch.metrics_by_window("epsilon")
print(f"{X.shape[0]} subjects x {X.shape[1]} windows")
print(X.round(4).to_string())


# %% 87b -- separating two groups, evaluated honestly [DD-83..87]
_labels = {k: ("control" if i < 4 else "study")
           for i, k in enumerate(sorted(subjects))}
_table = batch.metrics(rqa=True, dynamics=True)
for _fam in ("rqa", "dynamics", "both"):
    _rep = rc.classify_groups(_table, _labels, families=_fam, n_repeats=4)
    print(f"  {_fam:9s} {_rep.settings['n_features']:3d} features  "
          f"AUC {_rep.auc:.3f} +- {_rep.scores.auc.std(ddof=1):.3f}")
_rep = rc.classify_groups(_table, _labels, families="rqa", n_repeats=4,
                          permutations=20)
print()
print(_rep.summary())
print("\ntop features:")
print(_rep.importances.head(5).to_string(index=False))
print("\nfamilies available:", list(rc.classify.FAMILIES))
print("selected for 'rqa':", rc.select_features(_table, "rqa")[:6], "...")
_X, _g = rc.build_features(_table, families="rqa", level="window")
print(f"window-level matrix {_X.shape}, {len(set(_g))} subjects behind it")
print("report as one row:", list(_rep.to_frame().columns)[:6], "...")
print("ClassificationReport type:", isinstance(_rep, rc.ClassificationReport))

# %% 88 -- figures: the catalogue
print(rc.list_figures())

# %% 89 -- figures: attractors, every mode
for mode in ("2d", "3d", "pairs", "time", "torus"):
    fig = rc.plot_attractor(SS, mode=mode, max_points=1500)
    rc.save_figure(fig, f"cb_attractor_{mode}.png")
print("saved five attractor views")

# %% 90 -- figures: attractor options
fig = rc.plot_attractor(SS, mode="3d", color_by="amp_gamma", cmap="plasma",
                        point_size=2.0, alpha=0.4, line=False, max_points=2000,
                        title="PAC state space, coloured by envelope")
rc.save_figure(fig, "cb_attractor_options.png")
fig = rc.plot_attractor(SS, mode="3d", weighted=False, max_points=1500,
                        title="raw coordinates (not what the engine sees)")
rc.save_figure(fig, "cb_attractor_raw.png")
print("saved")

# %% 91 -- figures: diagnostics
p = sm.estimate_embedding(rc.lorenz(4000)[:, 0], tau_range=(1, 150),
                          m_range=(1, 8), rng=0)
rc.save_figure(rc.plot_embedding_diagnostics(p), "cb_embedding.png")
rc.save_figure(rc.plot_scale_report(SS), "cb_scale_report.png")
rc.save_figure(rc.plot_lambda_sweep(sweep), "cb_lambda_sweep.png")
rc.save_figure(rc.plot_quality(REC), "cb_quality.png")
print("saved four diagnostic figures")

# %% 92 -- figures: signals and coupling
rc.save_figure(rc.plot_signal(REC, ["signal", "theta", "amp_gamma"], seconds=4.0),
               "cb_signal.png")
rc.save_figure(rc.plot_phase_amplitude(REC, "phase_theta", "amp_gamma"),
               "cb_phase_amplitude.png")
rc.save_figure(rc.plot_envelope_pair(REC, "amp_theta", "amp_gamma"),
               "cb_envelope_pair.png")
print("saved three coupling figures")

# %% 93 -- figures: recurrence structures
rc.save_figure(rc.plot_recurrence(rm), "cb_recurrence.png")
rc.save_figure(rc.plot_recurrence(crp), "cb_cross_recurrence.png")
rc.save_figure(rc.plot_recurrence(jrp), "cb_joint_recurrence.png")
rc.save_figure(rc.plot_density_profile(rm), "cb_density.png")
rc.save_figure(rc.plot_threshold_diagnostics(th5, d), "cb_threshold.png")
print("saved five recurrence figures")

# %% 94 -- figures: pooling control for large matrices
for pooling in ("auto", "max", "mean", "none"):
    fig = rc.plot_recurrence(rm, max_size=300, pooling=pooling,
                             title=f"pooling = {pooling}")
    rc.save_figure(fig, f"cb_pooling_{pooling}.png")
print("saved four pooling variants")

# %% 95 -- figures: windowed
rc.save_figure(rc.plot_window_coverage(wr), "cb_window_coverage.png")
rc.save_figure(rc.plot_window_series(wr, "epsilon"), "cb_window_series.png")
rc.save_figure(rc.plot_window_panel(wr), "cb_window_panel.png")
print("saved three windowed figures")

# %% 96 -- figures: comparability and groups
rc.save_figure(rc.plot_comparability(batch.report), "cb_comparability.png")
rc.save_figure(rc.plot_group_comparison(gc[gc.metric == "epsilon"], "epsilon"),
               "cb_group_comparison.png")
rc.save_figure(rc.plot_feature_matrix(X, groups=groups), "cb_feature_matrix.png")
print("saved three group figures")

# %% 97 -- figures: comparison mode
rc.save_figure(compare.compare_attractors(
    {"observable": sm.pac_space(REC), "hybrid": sm.pac_space(REC, embed_amplitude=3)},
    ncols=2, max_points=1200), "cb_cmp_attractors.png")
rc.save_figure(compare.compare_recurrence(wr_keep.matrices(), ncols=3, max_size=400),
               "cb_cmp_recurrence.png")
rc.save_figure(compare.compare_thresholds(
    {lbl: rc.threshold(SS, theiler=1, rng=0, **kw) for lbl, kw in modes[:5]}),
    "cb_cmp_thresholds.png")
rc.save_figure(compare.compare_window_series(series, "independence_ratio"),
               "cb_cmp_window_series.png")
rc.save_figure(compare.compare_scale_policies(
    {k: v for k, v in list(policies.items())[:4]}), "cb_cmp_scaling.png")
rc.save_figure(compare.compare_phase_amplitude({"one": REC}), "cb_cmp_phaseamp.png")
rc.save_figure(compare.compare_routes(df_routes), "cb_cmp_routes.png")
rc.save_figure(compare.compare_series({"sweep": sweep}, x="lambda", y="corr_slope"),
               "cb_cmp_series.png")
print("saved eight comparison figures")


# %% 98 -- export: tables with provenance sidecars
rc.export_frame(rm.describe(), "cb_recurrence_summary")
rc.export_frame(wr.metrics(), "cb_window_metrics")
rc.export_frame(gc, "cb_group_comparison")
rc.export_recording(REC, "cb_recording")
print("exported; each table has _environment.csv and, where applicable, "
      "_provenance.csv")

# %% 99 -- export: accumulating across subjects into one tidy table
ex = rc.CsvExporter("cb_corpus_metrics")
for name, space in subjects.items():
    t = rc.threshold(space, target_rr=0.05, theiler=1, rng=0)
    ex.add(subject=name, group=groups[name], metric="epsilon", value=t.scalar)
print(ex.frame.to_string(index=False))
print("->", ex.write(note="cookbook"))

# %% 100 -- provenance: what was done, and what was done silently
print(rm.provenance_frame()[["step", "caveats"]].to_string(index=False,
                                                           max_colwidth=60))

# %% 101 -- a run report
report = rc.RunReport("cookbook", title="recurra cookbook run")
report.parameter(fs=500.0, target_rr=0.05, theiler=1)
report.section("Analysis").note("one PAC recording").result("RR", f"{rm.recurrence_rate():.4%}")
report.export(rm.describe(), "cb_report_table", "the recurrence summary")
report.save(rc.plot_recurrence(rm), "cb_report_figure.png", "the recurrence plot")
report.warn("envelope has a heavier tail than the phase block")
print(report.render()[:1200], "...")
print("\nwritten to:", report.write())

# %% 102 -- configuration
print(rc.get_config())
rc.set_config(memory_budget_gb=8.0, warn_on_caveat=True, strict_quality=False)
print("backends:", rc.available_backends())

# %% 102b -- parallelism: the same answer, faster [DD-81, DD-82]
print("workers for n_jobs=-1 on this machine:", rc.resolve_jobs(-1))
squares = rc.parallel_map(lambda v, rng=None: v * v, list(range(10)),
                          n_jobs=2, pass_rng=False)
print("order preserved:", squares)
_subs = {f"s{i}": SS for i in range(4)}
_a = rc.batch_windowed_recurrence(_subs, 4, target_rr=0.05, theiler=1, rng=5,
                                  n_jobs=1).metrics(rqa=True)
_b = rc.batch_windowed_recurrence(_subs, 4, target_rr=0.05, theiler=1, rng=5,
                                  n_jobs=2).metrics(rqa=True)
print("identical across job counts:",
      bool(np.allclose(_a["DET"].to_numpy(), _b["DET"].to_numpy())))

# %% 102c -- single precision for long records [DD-79, DD-80]
import time as _time

for _prec in ("double", "single"):
    _t = _time.time()
    _rm = rc.recurrence_plot(SS, target_rr=0.05, theiler=1, store="none",
                             precision=_prec, rng=0)
    _m = rc.rqa(_rm, metrics=["DET", "LAM"])
    print(f"  {_prec:7s} {_time.time() - _t:5.2f} s  DET={_m['DET']:.6f}  "
          f"LAM={_m['LAM']:.6f}")
print("  decimating by 2 keeps RR, DET and LAM but halves L, TT and Vmax,")
print("  and collapses Lmax -- see DD-80 before decimating.")

# %% 103 -- reproducibility: independent streams, order-independent results
a = rc.spawn_rngs(42, 4)
b = rc.spawn_rngs(42, 4)
print("reproducible:", [round(r.random(), 6) for r in a] ==
      [round(r.random(), 6) for r in b])
print("resolve_rng accepts anything:",
      type(rc.resolve_rng(None)).__name__, type(rc.resolve_rng(7)).__name__)

# %% 104 -- the command line
print("recurra doctor        # available backends")
print("recurra figures       # the figure catalogue")
print("recurra generate --modality pac --n 20 --duration 20 --out corpus/")

# %% 105 -- the remaining public entry points, for completeness
# Types, so `isinstance` checks and type hints work in your own code
print("every metric rqa() can return:", rc.METRIC_NAMES)
print("LineHistogram is a type too:", isinstance(rm.line_histogram(), rc.LineHistogram))
print("types:", [t.__name__ for t in (rc.Recording, rc.StateSpace,
                                      rc.RecurrenceMatrix, rc.Threshold,
                                      rc.WindowedRecurrence,
                                      rc.BatchWindowedRecurrence,
                                      rc.ComparabilityReport, rc.QualityReport,
                                      rc.Step, rc.WindowSpec)])
print("isinstance checks:", isinstance(REC, rc.Recording),
      isinstance(SS, rc.StateSpace), isinstance(rm, rc.RecurrenceMatrix),
      isinstance(wr, rc.WindowedRecurrence), isinstance(batch, rc.BatchWindowedRecurrence))

# The component-level Hilbert helper, when you want the arrays not a Recording
phase, amplitude, ifreq, n_trim = rc.hilbert_components(
    np.asarray(REC.get("theta")), edge_policy="mirror", fs=REC.fs)
print("hilbert_components ->", phase.shape, amplitude.shape, ifreq.shape,
      "trimmed", n_trim)

# Estimating tau and m separately rather than together
t_only = rc.estimate_tau(rc.lorenz(3000)[:, 0], method="ami", tau_range=(1, 120))
m_only = rc.estimate_m(rc.lorenz(3000)[:, 0], tau=t_only.tau, method="fnn",
                       m_range=(1, 8), rng=0)
print(f"estimate_tau -> {t_only.tau}, estimate_m -> {m_only.m}")

# Amplitude-amplitude coupling on its own
print("aac_index:", round(rc.aac_index(REC.get("amp_theta"),
                                       REC.get("amp_gamma")), 4))

# Draw and save an attractor in one call
print("save_attractor ->", rc.save_attractor(SS, "cb_attractor_shortcut.png",
                                             mode="torus", max_points=1500))

# The environment stamp that goes into every export
print("environment_stamp:", rc.environment_stamp())

# The optional-backend error names the exact install command
try:
    raise rc.BackendUnavailable("scikit-learn", "ml")
except rc.BackendUnavailable as e:
    print("BackendUnavailable:", e)

# The quality report object, on demand
print("QualityReport:", REC.quality.summary())

# Warnings have their own hierarchy, so you can quieten the library alone
print("warning classes:", [w.__name__ for w in (rc.RecurraWarning,
                                                rc.CaveatWarning,
                                                rc.InferenceWarning,
                                                rc.GeometryWarning,
                                                rc.ComparabilityWarning,
                                                rc.ParameterWarning)])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sm.pac_space(REC, smooth=8.0)
for c in caught:
    print(f"  {c.category.__name__:22s} {str(c.message)[:58]}...")

# Every library error descends from RecurraError, so one except clause suffices
for bad_call in (lambda: rc.ingest(np.random.default_rng(124900).standard_normal((3, 900)), fs=500.0),
                 lambda: rc.threshold(SS, mode="percentile", percentile=5.0),
                 lambda: rc.ingest({"flat": np.ones(500)}, fs=500.0, strict=True)):
    try:
        bad_call()
    except rc.RecurraError as e:
        print(f"  {type(e).__name__}: {str(e)[:70]}...")


print("\nCookbook finished. Output in:", rc.get_config().output_dir)


# %% Time-resolved separation [DD-114]
# The aggregated comparison says whether the groups differ; these say when.
# Self-contained: a constructed windowed table whose effect starts mid-record.
_tc_rng = np.random.default_rng(0)
_tc_rows = []
for _i in range(30):
    _s, _g = f"s{_i:02d}", ("b" if _i >= 15 else "a")
    for _w in range(10):
        _eff = 0.9 if (_g == "b" and _w >= 5) else 0.0
        _tc_rows.append({"subject": _s, "group": _g, "window": _w,
                         "DET": _tc_rng.normal() + _eff,
                         "LAM": _tc_rng.normal() + 0.5 * _eff})
_tc_table = pd.DataFrame(_tc_rows)
course, tc_summary = rc.group_timecourse(
    _tc_table, "group", values=["DET", "LAM"], time="window",
    n_permutations=200)
print(tc_summary)
tc_curve = rc.classify_timecourse(
    _tc_table, _tc_table.groupby("subject").group.first().to_dict(),
    time="window", families="rqa", n_repeats=2)
print(tc_curve.head())


# %% 106 -- meta-recurrence: the recurrence of the per-window measurements [DD-119]
_meta_ss = sm.pac_space(REC, smooth=12.0).subsample(max_points=3000)
_meta_wr = rc.windowed_recurrence(
    _meta_ss, rc.WindowSpec(n_windows=12, overlap=0.5), target_rr=0.05, rng=0)
_meta_table = _meta_wr.metrics(rqa=True, l_min=2, v_min=2)
print("meta-space columns:", [g.label for g in rc.meta_space(_meta_table).groups])
print("automatic Theiler window:", rc.meta_theiler(_meta_table))
_meta = rc.meta_recurrence(_meta_table, target_rr=0.2, rng=0)
print(_meta.summary())
print(rc.rqa(_meta, l_min=2, v_min=2))


# %% 107 -- figures that explain a recurrence plot [DD-120]
_panel_rm = rc.recurrence_plot(SS, target_rr=0.05, rng=0)
rc.save_figure(rc.plot_recurrence_panel(_panel_rm, rqa=True, l_min=2, v_min=2),
               "cookbook_recurrence_panel.png")
rc.save_figure(rc.plot_rqa_summary(_panel_rm, l_min=2, v_min=2), "cookbook_rqa_summary.png")
_panel_jrp = rc.joint_recurrence_plot([SS.split(["phase_theta"]), SS.split(["amp_gamma"])],
                                      target_rr=0.10, rng=0)
rc.save_figure(rc.plot_joint_recurrence(_panel_jrp), "cookbook_joint_recurrence.png")
rc.save_figure(rc.plot_recurrence_panel(_meta), "cookbook_meta_recurrence_panel.png")
