<picture><source media="(prefers-color-scheme: dark)" srcset="_static/recurra-mark-dark.svg"><img alt="recurra" src="_static/recurra-mark.svg" width="56" align="right"></picture>

# Usage examples — `recurra`

**Living document.** Extended with each phase. Entries marked 📋 describe specified API that is not implemented yet; they are here to fix the shape of the interface before it is written.

Runnable scripts: `examples/example_01_pipeline.py`, `examples/example_02_statespace.py`, `examples/example_03_recurrence.py`

---

## 0. Installation

```bash
# from the repository, pinned (reproducibility)
pip install "git+https://github.com/IgnacioRodro/recurra.git"

# with extras
pip install "recurra[all] @ git+https://github.com/IgnacioRodro/recurra.git"

# development
git clone https://github.com/IgnacioRodro/recurra.git
cd recurra && pip install -e ".[dev]"
```

Check the environment:

```python
import recurra as rc
print(rc.doctor())
```

---

## 1. Input: the eleven doors

### 1.1 Scalar signal

```python
import numpy as np, recurra as rc
rec = rc.ingest(np.random.randn(10000), fs=500.0)
```

### 1.2 Multichannel

```python
rec = rc.ingest(X, fs=500.0, labels=["Fz", "Cz", "Pz", "Oz"], subject="s01")
```

`X` must be `(n_samples, n_channels)`. A transposed array is rejected with a message saying so, rather than failing later.

### 1.3 Bands already filtered

```python
rec = rc.ingest({"theta": x_theta, "gamma": x_gamma}, fs=500.0, kind="bands")
```

### 1.4 Hilbert components already computed

The important case: no need to pretend you have the raw signal.

```python
rec = rc.ingest(
    {"phase_theta": phi, "amp_gamma": env},
    fs=500.0,
    roles={"phase_theta": "phase", "amp_gamma": "amplitude"},
)
print(rec.caps)      # PHASE|AMPLITUDE  -- no RAW
```

Without `roles`, names are matched against patterns (`phase_*`, `amp_*`) and the inference is recorded as a caveat with a warning. **Declaring `roles` is always preferable.**

### 1.5 Complex analytic signal

```python
rec = rc.ingest({"sig": z_complex}, fs=500.0)   # -> sig_phase and sig_amplitude
rec = rc.ingest((phase, envelope), fs=500.0)    # or as a pair
```

### 1.6 State space already built

```python
S = np.column_stack([np.cos(phi), np.sin(phi), A])
rec = rc.ingest(S, fs=500.0, kind="statespace")
```

### 1.7 Files

```python
rec = rc.ingest("record.csv")                # infers fs from a time column
rec = rc.ingest("data.npy", fs=250.0)
rec = rc.ingest("subject.edf")               # needs recurra[io]
rec = rc.ingest("subject.edf", channels=["Fz", "Cz", "Pz"])   # mixed rates
rec = rc.ingest("subject.edf", exclude=["ECG", "EOG"])
# units, patient code, start time and annotations come with it [DD-88]
rec.units, rec.meta["start_datetime"], rec.meta["annotations"]
rec = rc.ingest("corpus.h5", fs=500.0, group="/sub01")
```

---

## 2. Quality and inspection

```python
rec = rc.ingest({"good": x, "flat": np.ones(len(x))}, fs=500.0)
print(rec.quality.summary())          # quality: 1 warning, 1 error
print(rec.quality.to_frame())
```

To abort instead of warning:

```python
rec = rc.ingest(X, fs=500.0, strict=True)      # raises QualityError
rc.set_config(strict_quality=True)             # or globally
```

---

## 3. Preprocessing

```python
rec = rc.bandpass(rec, (4, 8), name="theta")
rec = rc.filterbank(rec, {"delta": (1,4), "theta": (4,8),
                          "alpha": (8,13), "gamma": (30,80)})
rec = rc.analytic(rec, sources=["theta", "gamma"])
rec = rc.notch(rec, 50.0, q=30.0)
rec = rc.resample(rec, 250.0)
```

Filtering: `method="fir"|"iir"`, `order="auto"|int`, `min_cycles=3.0`. Analytic decomposition: `edge_policy="trim"|"mirror"|"taper"|"none"`, `keep=("phase","amplitude","inst_freq")`, `unwrap`.

The edge cost is computed and recorded:

```python
step = [s for s in rec.provenance if s.name == "bandpass"][0]
print(step.params["edge_samples"], step.caveats)
# 375 ('first/last ~375 samples (0.750 s) are filter-transient',)
```

---

## 4. Synthetic signals

```python
sig = rc.generate_cfc(
    modality="pac", duration=30.0, fs=500.0,
    f_low=6.0, f_high=60.0,
    alpha=0.8,                     # coupling strength in [0,1]
    preferred_phase=np.pi/2,       # None = distributed
    concentration=2.0,
    modulation_shape="hanning",    # hanning | vonmises | sine | square
    snr_db=15.0, noise_beta=1.0,   # 1/f noise
    seed=42,
)
rec = sig.to_recording("sub01")
print(sig.truth)                   # complete ground truth
```

`sig.components` holds the **actual** slow phase and fast envelope, so estimators can be validated against the values that generated the data.

Modalities, non-stationarity and adverse scenarios:

```python
rc.generate_cfc(modality="ppc", nm_ratio=(2,1))
rc.generate_cfc(modality="aac", aac_correlation=0.7)
rc.generate_cfc(modality="ppa")
rc.generate_cfc(modality="none")                          # negative control

rc.generate_cfc(modality="pac", nonstationarity="transient", onset=10.0, offset=20.0)
rc.generate_cfc(modality="pac", alpha=lambda t: np.clip(t/30, 0, 1))

rc.generate_cfc(modality="pac", snr_db=-3.0)              # low SNR
rc.generate_cfc(modality="none", harmonic_contamination=1.0)   # fakes PAC
rc.generate_cfc(modality="none", harmonic_contamination=1.0,
                harmonic_sharpness=60.0)     # harmonics reach further up
rc.generate_cfc(modality="pac", artifacts=["spike","drift","blink"])
```

Parametric corpus:

```python
corpus = rc.generate_corpus(
    grid={"alpha": [0.0, 0.25, 0.5, 0.75, 1.0], "snr_db": [20, 10, 0],
          "modality": ["pac", "none"]},
    n_per_cell=20, duration=20.0, fs=500.0, seed=0)
```

Reference chaotic systems:

```python
from recurra.signals import lorenz, rossler, henon, van_der_pol, mackey_glass
X = lorenz(n_points=20000, dt=0.01, transient=10.0)   # D2 ~ 2.06, lambda1 ~ 0.906
```

---

## 5. Canonical coupling indices

```python
rc.mvl(phase, amplitude)                          # Canolty, normalised
rc.modulation_index(phase, amplitude, n_bins=18)  # Tort, in [0,1]
rc.plv(phase1, phase2, n=1, m=1)                  # Lachaux, n:m
rc.aac_index(env1, env2, method="spearman")
rc.coupling_indices(phase, amplitude, phase2=..., amp2=...)

from recurra.metrics.classic import surrogate_significance
surrogate_significance(rc.mvl, phase, amplitude, n_surrogates=200,
                       method="circular_shift", rng=0)
```

---

## 6. State space: the three routes

```python
from recurra import statespace as sm

# OBSERVABLE -- coordinates are physical quantities
ss = sm.pac_space(rec)                       # (cos phi, sin phi, A)
ss = sm.pac_space(rec, smooth=8.0)           # with envelope smoothing [DD-29]
ss = sm.phase_circle(rec, "phase_theta")
ss = sm.ppc_space(rec, "phase_theta", "phase_gamma")
ss = sm.envelope_space(rec)
ss = sm.ppa_space(rec, "phase_delta", "phase_theta", "amp_gamma")
ss = sm.freq_amp_space(rec)
ss = sm.channels(rec, ["x", "y", "z"])       # multivariate, NO coupling assumed

# DELAY -- Takens reconstruction
ss = sm.takens(rec, "signal", m=3, tau=16)
ss = sm.takens(rec, "signal", m="auto", tau="auto", rng=0)
ss = sm.takens_multivariate(rec, ["x", "y"], m=2, tau=16)

# HYBRID -- phase circle intact, envelope embedded
ss = sm.pac_space(rec, embed_amplitude=3, tau=8)
ss = sm.envelope_space(rec, embed=2)
```

Declarative constructor:

```python
ss = sm.build(rec, [
    {"source": "phase_theta", "as": "phase_circle", "label": "phase"},
    {"source": "amp_gamma",   "as": "delay", "m": 3, "tau": 8, "label": "env"},
], route="hybrid", scaling="rms_balanced")
```

Per-entry keys: `source`, `as` (`phase_circle` · `amplitude` · `raw` · `inst_freq` · `delay` · `phase_raw`), `label`, `weight`, `scale` (`zscore`/`robust`/`minmax`/`rank`/`none`), `smooth` (Hz); for `delay`: `m`, `tau`, `tau_method`, `m_method`.

From the raw signal, filtering and applying Hilbert on the way:

```python
ss = sm.from_bands(rec, [
    {"band": (4, 8),   "as": "phase",     "label": "theta"},
    {"band": (50, 70), "as": "amplitude", "label": "gamma", "smooth": 8.0},
])
```

Escape hatch:

```python
ss = sm.custom(rec, lambda r: np.column_stack([...]), scaling="rms_balanced")
```

### 6.1 Embedding parameters

```python
p = sm.estimate_embedding(x, tau_method="ami", m_method="fnn",
                          tau_range=(1, 300), m_range=(1, 10), rng=0)
print(p.tau, p.m, p.diagnostics["m_criterion"])
p.curve_frame()          # AMI(tau) and FNN(m) curves, exportable

sm.estimate_tau(x, method="acf")          # ami | acf | first_zero
sm.estimate_m(x, tau=16, method="cao")    # fnn | cao

# several records: estimate separately, aggregate; never concatenate [DD-24]
p, per_record = sm.estimate_embedding_multi([x1, x2, x3], aggregate="median")
```

On Lorenz-x, AMI + normalised FNN returns **m = 3**, its true dimension.

### 6.2 Scaling and inspection

```python
print(ss.summary())
ss.scale_frame()     # rms, weight, variance share, tail_ratio, skewness
ss.describe()        # one row, ready for CSV

ss.rescale("rms_balanced")
ss.reweight(0.8)                     # lambda: the amplitude's share
ss.rescale("weighted", lambda_=0.3)

ss.scale_report.shape_warning
# "block 'amp_gamma' has a much heavier tail than the others
#  (tail ratio 3.9 vs 1.4)... Consider scale='rank' or 'robust'"
```

`tail_ratio` reference values: circle 1.41 · uniform 2.36 · Gaussian 3.30 · Rayleigh 3.33 · lognormal 8.32.

### 6.3 Transforms and comparison

```python
ss.subsample(4);  ss.subsample(max_points=5000)
ss.window(1000, 5000)
ss.split(["phase_theta"])        # sub-space, for CRP/JRP operands
ss.reduce("pca", n_components=2)

df, spaces = sm.compare_routes(rec, rng=0)
df = sm.lambda_sweep(sm.pac_space(rec, scaling="none"),
                     grid=np.linspace(0, 1, 11), rng=0)
sm.geometry_descriptors(ss, rng=0)
sm.neighbour_agreement(ss_a, ss_b, k=8)
```

---

## 7. Threshold (F4)

```python
th = rc.threshold(ss, mode="target_rr", target_rr=0.05, theiler=1, rng=0)
print(th.summary())
print(th.achieved_rr)     # report THIS, not the target you asked for
th.to_frame()             # exportable
```

### 7.1 All modes

```python
rc.threshold(ss, mode="target_rr", target_rr=0.05)                  # default
rc.threshold(ss, mode="target_rr", target_rr=0.05, method="bisect") # refined
rc.threshold(ss, mode="fixed", value=0.42)
rc.threshold(ss, mode="percentile", percentile=0.05)   # a fraction, not 5
rc.threshold(ss, mode="fan", n_neighbors=25)           # per point, asymmetric
rc.threshold(ss, mode="std_fraction", factor=0.1)
rc.threshold(ss, mode="maxdist_fraction", factor=0.1)
rc.threshold(ss, mode="per_block", target_rr=0.05)     # scale-free
rc.threshold(ss, mode="adaptive_dim", target_rr=0.05)  # + dimension diagnostic
```

Common arguments: `metric`, `theiler`, `scope`, `n_samples`, `tol`, `max_iter`, `rng`.

Measured accuracy across targets from 0.5% to 30%: worst absolute deviation between requested and achieved rate **0.0036**.

`fan` gives every point its own epsilon and therefore an asymmetric matrix — recorded as a warning on the threshold. `per_block` sets one epsilon per coordinate block, combined with Chebyshev, so no relative weighting is needed at all.

### 7.2 Cross threshold

```python
from recurra.threshold import cross_threshold
th = cross_threshold(ss1, ss2, target_rr=0.05, rng=0)
```

No Theiler exclusion: the two trajectories are different systems.

### 7.3 Sampling directly

```python
from recurra.threshold import sample_pair_distances, sample_cross_distances
d = sample_pair_distances(ss.weighted_coords, n_samples=200_000, theiler=1, rng=0)
```

---

## 8. Recurrence structures (F5)

### 8.1 Recurrence plot

```python
rm = rc.recurrence_plot(ss, target_rr=0.05, theiler=1, rng=0)
rm = rc.recurrence_plot(ss, th)              # with a prepared Threshold
rm = rc.recurrence_plot(ss, 0.42)            # fixed epsilon shortcut
rm = rc.recurrence_plot(ss, "fan", n_neighbors=20)

print(rm.summary())
rm.recurrence_rate()        # streamed, exact
rm.describe()               # one row, exportable
```

Arguments: `metric`, `theiler`, `dtype`, `store`, `tile_size`, `budget_gb`, plus anything the threshold takes.

**Theiler convention [DD-115]:** `theiler=w` excludes every pair with `|i - j| < w`, as in the CRP Toolbox, pyunicorn and PyRQA. `theiler=0` excludes nothing; `theiler=1` excludes the line of identity only. Results from versions before 0.25 used `|i - j| <= w`; reproduce them with `theiler=w + 1`.

### 8.2 Storage and streaming

```python
rm = rc.recurrence_plot(ss, target_rr=0.05, store="memory")   # dense array
rm = rc.recurrence_plot(ss, target_rr=0.05, store="sparse")   # CSR
rm = rc.recurrence_plot(ss, target_rr=0.05, store="none")     # never allocated
rm = rc.recurrence_plot(ss, target_rr=0.05, store="auto")     # from the budget
rm = rc.recurrence_plot(ss, target_rr=0.05, precision="single")  # 3.2x faster [DD-79]

for tile in rm.tiles(tile_size=1024):
    process(tile.data, tile.i0, tile.j0)     # the matrix need not exist

rm.matrix          # materialise on demand, budget-checked
rm.to_sparse()
```

Streaming is **exact**: reassembling tiles gives the dense matrix bit for bit, for any tile size.

### 8.3 Memory planning

```python
rc.set_config(memory_budget_gb=8.0)
p = rc.recurrence.plan(1_800_000, 1_800_000, 3, dtype="uint8", estimated_rr=0.05)
print(p.summary())
# dense 3017 GiB, sparse 604 GiB, chosen store 'none'
```

Over budget, the library refuses with the numbers and the options, rather than swapping.

### 8.4 Cross recurrence

```python
crp = rc.cross_recurrence_plot(ss_a, ss_b, target_rr=0.05, rng=0)
crp.shape          # rectangular, not symmetric
```

Both trajectories must have the same dimensionality: a cross recurrence compares them in one common phase space.

### 8.5 Joint recurrence [DD-20]

```python
jrp = rc.joint_recurrence_plot([ss_phase, ss_amplitude],
                               target_rr=0.10, theiler=1, rng=0)

jrp.subsystem_rates()      # each subsystem on its own
jrp.independence_ratio()   # 1.0 = subsystems recur independently
[t.scalar for t in jrp.thresholds]     # independent epsilons
```

Each subsystem gets its **own** threshold in its **own** phase space. Subsystems need a common time base but **not** the same dimensionality — that is the point.

With explicit thresholds:

```python
jrp = rc.joint_recurrence_plot([ss_a, ss_b], thresholds=[th_a, th_b])
```

### 8.6 Density profile

```python
rm.density_profile(n_bins=64)     # local density along time, streamed
```

A collapse in density marks a stretch the threshold does not suit: a regime change, an artefact, or an amplitude excursion.

---

## 8a. Recurrence quantification (F6)

```python
m = rc.rqa(rm)                                   # dict of every metric
rc.line_length_sweep(rm.line_histogram())    # pick l_min and v_min from data
m = rc.rqa(rm, metrics=["RR", "DET", "LAM"])     # a selection
df = rc.rqa(rm, as_frame=True)                   # one row, exportable
m, hist = rc.rqa(rm, return_histogram=True)      # keep the distributions
m = rc.rqa(state_space, target_rr=0.05)          # builds the plot for you
```

Metrics: `RR` · `DET` · `L` · `Lmax` · `DIV` · `ENTR` · `ENTR_norm` · `LAM` ·
`TT` · `Vmax` · `W` · `Wmax` · `ENTW` · `RTE`.

**`l_min` and `v_min` are not the same number.** DET saturates against 1 on a
smooth trajectory and needs a floor near 8; LAM collapses to zero above 2,
because a laminar state lasts a few samples [DD-98]. The defaults stay at the
conventional 2 so published values remain comparable, and a `GeometryWarning`
fires when DET or LAM is against its ceiling.

**Choose the line-length floor from your data** [DD-98]. The defaults are
`l_min=8` and `v_min=2`, which is not the classical 2 and 2: on a smooth
trajectory DET saturates at 0.996 with a floor of 2, and raising the vertical
floor destroys LAM instead. The histogram already holds every threshold, so a
sweep is nearly free:

```python
hist = rm.line_histogram()
rc.line_length_sweep(hist)        # DET and LAM at floors 2..20, plus headroom
```

Saturation and collapse both raise a `GeometryWarning` naming the remedy.

Options: `l_min`, `v_min`, `w_min` (shortest line that counts), `theiler`,
`denominator` (`"theiler_corrected"` by default, or `"all"` for the older
convention [DD-58]), `tile_size`.

### The histograms underneath

Every metric is a function of three distributions, accumulated by streaming so
the matrix never has to exist [DD-56]:

```python
hist = rm.line_histogram()
print(hist.summary())
hist.to_frame("diagonal")      # length, count
hist.frames()                  # all three, tidy
rc.plot_line_histogram(hist)
```

### Cross recurrence: lag and direction

```python
prof = rc.diagonal_profile(crp)          # density per diagonal offset
out = rc.crqa(crp)                       # RQA plus peak_offset, asymmetry
rc.plot_diagonal_profile(prof)
```

The peak of the profile is the lag at which one trajectory best matches the
other; the asymmetry about zero says which leads.

### In the windowed table

```python
wr.metrics(rqa=True)                     # every metric as a column
wr.metrics(rqa=["DET", "LAM"])           # a selection
batch.metrics(rqa=True)                  # across a corpus
```

Asking for RQA does not force any matrix into memory.

---

## 8a2. Dynamical invariants (F9)

```python
d2 = rc.correlation_dimension(ss, theiler=20, max_points=5000)
l1 = rc.lyapunov_max(ss, max_steps=150, theiler=50)      # per_second by default
k2 = rc.k2_entropy(ss, target_rr=0.05, theiler=20)

print(d2.summary())        # value, fitted region, R2, and any warning
d2.to_frame()              # one exportable row
float(l1)                  # just the number
```

Every result is an `Invariant` carrying its **units**, the sampling rate it was
computed at, the method, and the region that was fitted [DD-62]. The region is
located rather than assumed [DD-61]:

```python
rc.correlation_sum(ss)              # the raw C(r) curve
rc.divergence_curve(ss)             # the raw divergence curve
rc.correlation_dimension(ss, region=(lo, hi))    # fit a window of your own
rc.lyapunov_max(ss, method="kantz", radius=0.1)
rc.lyapunov_max(ss, units="per_sample")
rc.lyapunov_max(ss, n_repeats=5)     # more samplings, narrower spread [DD-101]
```

**Read the warnings.** The Lyapunov exponent carries four checks, because four
different things can decide the answer instead of the data: neighbours at
machine precision [DD-63], a fit window that moves with the curve length
[DD-94], a window sitting in the saturated tail [DD-94], and an estimate that
moves with which reference points were drawn [DD-101].

The last one is not optional and not cheap to see. Twelve streams over
identical Lorenz data gave 0.501 to 1.366 against a reference of 0.906, and
eleven came back unflagged. `n_repeats` defaults to 3: the value is the mean
of three samplings and `sampling_cv` says how far they disagreed.

Neither guard works alone. At one sampling a curve run to 1200 steps returns
1.54 and nothing notices; averaged over three it is flagged, while the sound
settings stay clean at 0.91 to 0.98.

Complexity measures take a 1-D series, not a state space:

```python
rc.detrended_fluctuation(x)     # 0.5 white, 1.0 pink, 1.5 Brownian
rc.hurst_exponent(x)
rc.higuchi_dimension(x)         # 1 smooth, 2 very rough
rc.katz_dimension(x)
rc.petrosian_dimension(x)
rc.permutation_entropy(x, order=3)
rc.sample_entropy(x, m=2)
rc.spectral_entropy(x, fs=500.0)
rc.svd_entropy(x, m=5)
rc.lempel_ziv_complexity(x)     # 1 incompressible, lower means structured
rc.hjorth_parameters(x, fs=500.0)   # activity, mobility, complexity
```

### As a row of a table

```python
rc.dynamics_measures(ss)                       # compact set, per block
rc.dynamics_measures(ss, measures="all")       # every scalar measure
rc.dynamics_measures(ss, invariants=True)      # plus D2, lambda_1, K2
rc.dynamics_measures(ss, block_invariants=True)  # lambda_1 and D2 per block
rc.invariants_from_series(signal, fs=500.0)      # D2 and lambda_1 from a delay
                                                 # embedding of the oscillation,
                                                 # which is then discarded [DD-97]

# D2 and lambda_1 from a delay embedding of a signal [DD-97]
rc.invariants_from_series(band_signal, fs=500.0)   # tau and m estimated
rc.invariants_from_series(x, fs=500.0, m=6, tau=12, theiler="auto")
rc.SCALAR_MEASURES, rc.TRAJECTORY_MEASURES, rc.BLOCK_INVARIANTS
rc.FINE_SCALE_MEASURES            # the ones that need their own scale [DD-99]
rc.samples_per_turn(x)            # 1.5 for white noise, 24 for a smooth envelope
rc.fine_scale_factor(x)           # the decimation that brings it back
```

Each block contributes its natural scalar and the columns are named
`measure_block` [DD-71]. A phase circle contributes its **instantaneous
frequency**, not the wrapped angle.

**Fine-grained measures are decimated to their own scale** [DD-99]. Higuchi,
Katz, Petrosian and the ordinal entropies read point-to-point structure, and on
a series sampled far above its content they return constants: across 98 real
recordings `petrosian_fd` had a coefficient of variation of 0.00003. The
factor used and the ratio it came from are recorded as `fine_scale_<block>` and
`samples_per_turn_<block>`; `fine_scale=None` restores the old behaviour. DFA,
Hjorth and the invariants always see the whole series.

### In the windowed table

```python
wr.metrics(dynamics=True)                   # the cheap set, per window
wr.metrics(rqa=True, dynamics=True)         # both families side by side
wr.metrics(dynamics=["higuchi_fd"], invariants=["K2"])
batch.metrics(rqa=True, dynamics=True)      # across a corpus
```

Cheap by default: D2 and the Lyapunov exponent cost seconds per window against
milliseconds for the rest, so they are opt-in [DD-72].

---

## 8b. Windowed recurrence with overlap

The main use case: one long record becomes many overlapping recurrence structures, each with its own threshold, metrics and figure.

### 8b.1 Declaring windows [DD-39]

Three equivalent forms:

```python
rc.WindowSpec(n_windows=10, overlap=0.5)          # ten windows, spanning the record
rc.WindowSpec(size=4.0, overlap=0.5)              # 4 s windows, 50% overlap
rc.WindowSpec(size=4.0, step=0.5)                 # 4 s windows every 0.5 s
rc.WindowSpec(size=2000, step=1000, unit="samples")

spec.summary(n_samples, fs)                       # what it resolves to
spec.describe(n_samples, fs)                      # the full grid as a DataFrame
```

Asking for `n_windows=k` gives **exactly k** windows covering the record from first to last sample. With `size`/`step`, any uncovered tail is reported as a caveat; `drop_last=False` pulls a final window back to the record end and records the extra overlap that creates.

Shorthands: an integer means that many windows, and a list of `(start, stop)` pairs is taken literally.

```python
wr = rc.windowed_recurrence(ss, 10, target_rr=0.05)          # ten windows
wr = rc.windowed_recurrence(ss, [(0, 1000), (500, 1500)])    # explicit
```

### 8b.2 The core call

```python
wr = rc.windowed_recurrence(ss, rc.WindowSpec(n_windows=10, overlap=0.5),
                            target_rr=0.05, theiler=1, rng=0)

len(wr)              # 10
wr[3]                # the RecurrenceMatrix of window 3
for window, rm in wr:
    ...              # (Window, RecurrenceMatrix) pairs
wr.metrics()         # tidy DataFrame, one row per window
wr.series("epsilon") # one metric against window centre time
print(wr.summary())
```

Every option of the unwindowed path is available: threshold mode and its arguments, distance metric, Theiler window, dtype, storage strategy, tile size, memory budget.

```python
rc.windowed_recurrence(ss, spec, threshold="fan", n_neighbors=30)
rc.windowed_recurrence(ss, spec, threshold="per_block", target_rr=0.05)
rc.windowed_recurrence(ss, spec, threshold=0.45)              # fixed epsilon
rc.windowed_recurrence(ss, spec, metric="chebyshev", target_rr=0.05)
```

And every state-space route:

```python
rc.windowed_recurrence(sm.pac_space(rec, smooth=8.0), spec, target_rr=0.05)
rc.windowed_recurrence(sm.pac_space(rec, embed_amplitude=3), spec, target_rr=0.05)
rc.windowed_recurrence(sm.takens(rec, "x", m=3, tau=16), spec, target_rr=0.05)
rc.windowed_recurrence(raw_array, spec, target_rr=0.05)
```

### 8b.2b Choosing a window length [DD-67]

Measured on synthetic PAC at α = 0.7 and 20 dB, the shortest window at which
each measure separates coupled from control:

| measure | shortest window |
|---|---|
| MVL, MI | 1 s |
| LAM, independence ratio | 2 s |
| TT | 4 s |
| ENTR | 8 s |
| DET, L, RTE | 16 s |
| Lmax | never |

**Four seconds is a poor default and eight is defensible.** If DET is the
metric of interest the window must be sixteen seconds, which caps the temporal
resolution. If fine resolution is needed, use LAM or the independence ratio and
say so, rather than reporting DET from windows too short to support it.

`Lmax` must never be compared across windows of different length: it nearly
triples between 1 s and 32 s on the same signal.

### 8b.3 Threshold scope [DD-40]

**The consequential choice.** It decides which quantity carries information.

```python
rc.windowed_recurrence(ss, spec, scope="per_window", target_rr=0.05)
# epsilon adapts per window; the rate is pinned at 5%.
# Structure is comparable across windows; density says nothing.

rc.windowed_recurrence(ss, spec, scope="global", target_rr=0.05)
# one epsilon for all windows; RR(t) becomes a time series.
# Density is the signal; a window whose amplitude drifts really is sparser.
```

The scope is recorded on every row of the metrics table, so a result can never be read under the wrong assumption.

### 8b.4 Windowed cross and joint recurrence

```python
wc = rc.windowed_cross_recurrence(circle_theta, circle_gamma, spec, target_rr=0.05)

wj = rc.windowed_joint_recurrence([ss.split(["phase_theta"]),
                                   ss.split(["amp_gamma"])],
                                  spec, target_rr=0.10, theiler=1)
wj.metrics()[["window", "rr_subsystem_0", "rr_subsystem_1", "independence_ratio"]]
```

Each subsystem keeps its own threshold in its own space [DD-20], per window.

**The independence ratio traces coupling in time** [DD-54]. It is the joint
recurrence rate divided by the product of the subsystem rates, so 1.0 means the
two recur independently:

```python
wj.metrics()[["window", "t_center_s", "independence_ratio"]]
rc.plot_window_series(wj, "independence_ratio")
```

On a transiently coupled signal the curve is flat at 1, rises during the
transient and returns — detecting onset and offset with a latency below the
window step. What makes it worth having is that **the subsystem rates do not
change**: nothing visible to either signal on its own has moved, only the
coincidence between them.

Three cautions before using it as a test. The baseline is not exactly 1 (0.983
measured, from quantile estimation inside short windows), so calibrate against
surrogates rather than assuming the theoretical value. Overlapping windows are
not independent samples. And the transition width on the curve is the window
length, not a property of the signal.

### 8b.5 A corpus: subjects times windows

```python
subjects = {f"sub{i:02d}": build_space(i) for i in range(100)}

batch = rc.batch_windowed_recurrence(
    subjects, rc.WindowSpec(n_windows=10, overlap=0.5),
    target_rr=0.05, theiler=1, rng=0, progress=True,
    n_jobs=-1)                       # every core; the result is unchanged [DD-81]

batch.n_matrices        # 1000
m = batch.metrics()     # one tidy row per subject and window
batch["sub07"]          # that subject's WindowedRecurrence
batch.describe()        # one row per subject
```

Records may have **different lengths**: the spec is resolved against each one [DD-18]. Each record gets an independent random stream, so results do not depend on processing order [DD-05].

For CRP or JRP across a corpus, map each record to its list of trajectories:

```python
batch = rc.batch_windowed_recurrence(
    {k: [v.split(["phase_theta"]), v.split(["amp_gamma"])] for k, v in subjects.items()},
    spec, kind="jrp", target_rr=0.10, theiler=1, rng=0)
```

### 8b.6 Memory [DD-41]

Matrices are built on demand and **not retained**, so a corpus of hundreds of subjects costs one window of memory at a time. Pass `keep=True` when the matrices will be reused, for instance to draw all of them.

```python
wr = rc.windowed_recurrence(ss, spec, target_rr=0.05)              # lazy
wr = rc.windowed_recurrence(ss, spec, target_rr=0.05, keep=True)   # cached
```

---

### 8b.4 Figures that explain a recurrence plot [DD-120]

```python
rc.plot_recurrence_panel(rm)                       # RP with its trajectory along both margins
rc.plot_recurrence_panel(rm, rqa=True)             # ... and the RQA measures beside it
rc.plot_recurrence_panel(crp, names=("Fz", "Cz"))  # CRP: rows on the left, columns on top
rc.plot_recurrence_panel(rm, signals=(raw, raw))   # the raw signal instead of the coordinates
rc.plot_joint_recurrence(jrp)                      # each subsystem, the AND, and the overlay
rc.plot_rqa_summary(rm, l_min=8, v_min=8)          # the line-length histograms behind DET and LAM
rc.save_figure(rc.plot_recurrence_panel(rm), "panel.png")    # drawing never saves by itself
```

The line of identity runs from bottom left to top right in these figures,
the classical orientation, so time increases upward along the left margin.

### 8b.5 Meta-recurrence: the recurrence of the window measurements [DD-119]

A windowed analysis gives one row of measurements per window. That sequence
is itself a multivariate time series, one sample per window, and
`meta_space` turns it into a state space so every tool applies to it:

```python
wr = rc.windowed_recurrence(ss, rc.WindowSpec(size=10, overlap=0.5, unit="seconds"),
                            target_rr=0.05, rng=0)
table = wr.metrics(rqa=True)                 # one row per window

space = rc.meta_space(table)                 # DET, LAM, L, ... z-scored; time = window centres
print(rc.meta_theiler(table))                # 2 at 50% overlap: overlapping windows excluded

meta = rc.meta_recurrence(table, target_rr=0.2, rng=0)    # the meta-RP
print(rc.rqa(meta, l_min=2, v_min=2))        # line lengths in windows
meta2 = wr.meta_recurrence(["DET", "LAM", "TT"], target_rr=0.2, rng=0)   # same, as a method
cross = rc.meta_recurrence(table_fz, kind="crp", other=table_cz, target_rr=0.2)   # two channels
joint = rc.meta_recurrence([table_delta, table_theta], kind="jrp", target_rr=0.2)
```

Bookkeeping columns (window indices, times, thresholds, floors, `qc_*`) are
never used, the recurrence rate is left out when the threshold fixed it, and
each column is rescaled on its own (`scaling="zscore"`, `"rank"`, `"robust"`
or `"none"`). Windows with a missing value are dropped with a warning
(`missing="drop"`) or refused (`missing="error"`). A meta-series has tens of
points, so a higher `target_rr` than for a signal is usually right.

## 8c. Comparability across subjects [DD-42, DD-43, DD-44]

Recurrence structures are only comparable if they were built the same way. Three defaults break that silently, so it is checked rather than assumed.

### 8c.1 What breaks, and how it is caught

```python
batch = rc.batch_windowed_recurrence(subjects, spec, target_rr=0.05, rng=0)
print(batch.report.summary())
```

The check runs by default. It reports axis by axis and, crucially, **which metric families remain valid**:

```
  OK   dim              constant at 3
  FAIL n_points         varies across units: 3 distinct values
  FAIL window_samples   varies across units: 3 distinct values

  comparable metric families : line_ratio, timescale, invariant
  NOT comparable: line_length      (unmatched: n_points)
  NOT comparable: entropy          (unmatched: n_points)
```

The three silent failure modes:

| Failure | Symptom | Fix |
|---|---|---|
| `m="auto"`, `tau="auto"` | subjects in different-dimension spaces | estimate once, then fix `m` and `tau` |
| `WindowSpec(n_windows=k)` | window *k* covers different durations | `WindowSpec(size=..., step=...)` in seconds or samples |
| `subsample(max_points=N)` | different effective sampling rates | resample to a common `fs` instead |

### 8c.2 The fix

```python
SPEC = rc.WindowSpec(size=6.0, overlap=0.5, unit="seconds")   # fixed grid
batch = rc.batch_windowed_recurrence(subjects, SPEC, target_rr=0.05, rng=0)
assert batch.report.ok
```

Records of different length then yield **different numbers of windows**, which is correct, while window *k* covers the same stretch of every record.

### 8c.3 Explicit contracts

```python
contract = rc.ComparabilityContract(
    dim=3, fs=500.0, metric="euclidean", theiler=1,
    threshold_mode="target_rr", target_rr=0.05,
    window_samples=3000, window_step=1500, scaling="rms_balanced",
    notes="PAC space, theta phase x gamma envelope")

batch = rc.batch_windowed_recurrence(subjects, SPEC, contract=contract, ...)
report = batch.check_comparability(contract)
report.metric_validity()          # what this matching licenses
report.raise_if_violated()

rc.export_frame(contract.to_frame(), "contract")     # goes next to the results
```

Derive a contract from an analysis already done:

```python
contract = rc.contract_from(batch["sub00"])
```

With `strict=True` a violation raises instead of reporting.

### 8c.4 Separating two groups

```python
groups = {"sub00": "control", "sub01": "study", ...}

gc = batch.group_comparison(groups, values=["independence_ratio"])
# per window: n_a, n_b, mean_a, mean_b, difference, cohens_d, auc,
#             p_value, n_tests, p_bonferroni

rc.rank_metrics(gc)               # metrics ordered by separation
```

Windows are compared **like with like**: every subject's window 0 against every other's window 0. Pool them instead with `by=None`.

### 8c.5 Feature matrix for a classifier

```python
X = batch.metrics_by_window("independence_ratio")   # subjects x windows
X.shape                                             # (20, 15)
```

Shorter records leave missing entries rather than being silently truncated.

### 8c.6 Figures

```python
rc.plot_comparability(batch.report)          # axes and licensed families
rc.plot_group_comparison(gc, "independence_ratio")
rc.plot_feature_matrix(X, groups=groups)
```

---

## 8c2. A corpus on disk (F1b)

```python
index = rc.read_corpus("study/")            # sub-XXX dirs + participants.tsv
print(index.summary())
index.recordings                            # subject -> path, not loaded
index.labelled()                            # only subjects with file and label
index.unmatched_rows, index.unmatched_files # the join, reported [DD-89]
```

The index holds paths, so an eight-minute corpus is loaded one recording at a
time:

```python
spaces = {}
for subject, path in index.recordings.items():
    rec = rc.analytic(rc.filterbank(rc.ingest(path),
                                    {"theta": (4, 8), "gamma": (50, 70)}),
                      sources=["theta", "gamma"])
    spaces[subject] = sm.pac_space(rec, smooth=8.0).subsample(2)   # 500 -> 250 Hz

batch = rc.batch_windowed_recurrence(spaces, rc.WindowSpec(n_windows=10, overlap=0.5),
                                     target_rr=0.05, theiler=1, rng=0, n_jobs=-1)
report = rc.classify_groups(batch.metrics(rqa=True, dynamics=True),
                            index.labelled(), families="both", permutations=200)
```

---

## 8c3. Recurrence plots as images (F14)

For a classifier that takes pictures rather than features.

```python
image, info = rc.recurrence_image(space, size=256, target_rr=0.05, theiler=1)
# image is a density map in [0, 1]: each pixel is the fraction of recurrent
# cells in the block it covers [DD-90]

meta = rc.export_recurrence_images(
    spaces, "images/", size=256,
    labels=index.labelled(),        # written into the metadata
    decimate="match",               # a second per image instead of a minute
    fmt="png",                      # or "npy", which needs no image library
    target_rr=0.05, theiler=1, rng=0)
```

Three things it does that an image library would not:

**It pools instead of resampling.** Reducing a 34 000-point plot to 256 squared
by nearest neighbour keeps one cell in seventeen thousand [DD-90].

**The brightness scale is shared across the batch.** Every recording is pinned
to the same rate, so what distinguishes subjects is how the density is
distributed, and per-image normalisation rescales that away [DD-91].

**Every image comes out the same size**, even from recordings of different
length, because a network needs that.

---

## 8c4. Attractors as images (F14b)

```python
image, info = rc.attractor_image(space, size=256, coords=(0, 2))
rc.shared_limits(spaces)                 # axes that hold for every space,
                                         # normalised by default [DD-105]
rc.plot_attractor(space, axis_limits=lims)   # so separate files compare [DD-104]
meta = rc.export_attractor_images(spaces, "attractors/", size=256,
                                  coords=(0, 2), labels=labels)
```

Each pixel counts the trajectory points falling in it, so the value is local
occupancy rather than a silhouette [DD-103]. `density=False` gives the
silhouette a scatter plot would show. The scale is shared across the batch, the
limits are robust to one outlying burst, and the projection is recorded --
`(0, 1)` is the phase circle itself, `(0, 2)` crosses phase with amplitude.

### The shape of each block

```python
rc.shape_descriptors(space)     # tail_ratio, skew, max_ratio per block
sm.geometry_descriptors(space)  # includes them
```

The tail ratio is the quantity most likely to be confounding a group
comparison: with coupling held constant it drove 22 of 39 metrics [DD-102].
Record it per subject and adjust for it, as the modulation index was adjusted
for in DD-65.

---

## 8d. Separating two groups (F13)

The whole point of the corpus machinery: given a metrics table and a mapping
from subject to group, how well do they separate, and is it real?

```python
report = rc.classify_groups(
    batch.metrics(rqa=True, dynamics=True),
    labels,                       # {"sub00": "control", "sub01": "study", ...}
    families="rqa",               # "dynamics", "classical", "both", "all", or a list
    level="subject",              # or "window"
    model="logistic",             # or "forest", "svm"
    n_splits=5, n_repeats=10,
    permutations=200,
)
print(report.summary())
report.to_frame()                 # one exportable row
report.per_fold                   # every fold, for a figure
report.importances                # averaged across folds
```

### What it guarantees

**Folds split by subject** [DD-83]. Windows of one subject never straddle a
fold, so the model cannot learn to recognise the subject. A test permutes the
labels and asserts the AUC falls to chance — without that, no other number here
would be worth reading.

**Everything that learns is fitted inside the fold** [DD-84]: imputation,
scaling, and any selection.

**Imbalance is handled and reported** [DD-85]: AUC, balanced accuracy,
sensitivity, specificity, precision, F1 and MCC, next to the majority-class
baseline. Classifiers use balanced class weights.

**The scheme is repeated** [DD-86] and the spread is printed beside every mean.
Small samples, more features than subjects, and a minority group too small for
the requested folds are all flagged in the report.

**Significance comes from permutation** [DD-87], shuffling whole subjects and
rebuilding the entire cross-validation.

### The whole recording, or windows

```python
whole = batch.metrics(rqa=True)                       # with a one-window spec
rc.classify_groups(whole, labels, families="both")    # one plot per subject

rc.classify_groups(windowed, labels, level="subject") # windows averaged
rc.classify_groups(windowed, labels, level="window",  # windows kept
                   aggregate="mean_std_trend")        # + spread and trend
```

### Choosing features by hand

```python
rc.select_features(table, "rqa")            # what a family resolves to
rc.build_features(table, families=["DET", "LAM", "higuchi_fd_amp_gamma"])
```

---

## 8c5. Attractor figures that compare

One file per subject, on axes computed across the whole set, so they can be
read side by side [DD-104]:

```python
limits = rc.shared_limits(spaces)
for name, space in spaces.items():
    fig = rc.plot_attractor(space, mode="3d", axis_limits=limits,
                            title=f"{name} · Fz · PAC space")
    rc.save_figure(fig, f"{name}.png")

rc.compare_attractors(spaces, mode="3d", share_limits=True)   # all in one panel
```

Every comparison figure is reachable at the top level: `compare_attractors`,
`compare_recurrence`, `compare_routes`, `compare_thresholds`,
`compare_series`, `compare_window_series`, `compare_scale_policies`,
`compare_phase_amplitude`.

---

## 9. Figures [DD-30, DD-31]

All figure text and exported file names are in English.

```python
print(rc.list_figures())     # or: recurra figures
```

### 9.1 Single figures: one object, one figure

They behave identically on synthetic and real data because they **generate nothing**.

```python
rc.plot_attractor(ss, mode="3d")             # auto|2d|3d|pairs|time|torus
rc.plot_attractor(ss, mode="torus", cmap="plasma")
rc.plot_attractor(ss, mode="3d", clip=None)     # no robust axis limits [DD-50]
rc.plot_embedding_diagnostics(params)
rc.plot_scale_report(ss)
rc.plot_lambda_sweep(sweep_df)
rc.plot_quality(rec)
rc.plot_phase_amplitude(rec, "phase_theta", "amp_gamma")
rc.plot_signal(rec, ["signal", "theta"], seconds=4.0)
rc.plot_envelope_pair(rec, "amp_theta", "amp_gamma")

rc.plot_recurrence(rm)                       # RP, CRP or JRP
rc.plot_density_profile(rm)
rc.plot_threshold_diagnostics(th, distances)

rc.plot_window_coverage(wr)                  # how the windows tile the record
rc.plot_window_series(wr, "recurrence_rate") # one metric across windows
rc.plot_window_panel(wr, columns=("epsilon", "recurrence_rate"))

rc.plot_comparability(report)                # construction axes and licensing
rc.plot_group_comparison(gc, "metric")       # group means and effect size
rc.plot_feature_matrix(X, groups=groups)     # subject-by-window heat map
```

`plot_recurrence` reduces oversized matrices with `pooling="auto"`, which avoids the saturation that maximum pooling causes at high reduction factors [DD-38]. The method used is written into the caption.

### 9.2 Drawing and saving are separate

```python
fig = rc.plot_recurrence(rm)
rc.save_figure(fig, "recurrence_sub01.png")
rc.save_figure(fig, "recurrence_sub01.png", subdir="subject_01")
```

### 9.3 Comparison mode

```python
from recurra.viz import compare

compare.compare_recurrence({"PAC": rm_pac, "control": rm_null, "Lorenz": rm_lor},
                           ncols=3, title="Recurrence plots by regime")
compare.compare_thresholds({"5%": th_a, "fan k=25": th_b, "fixed": th_c})
compare.compare_attractors({"observable": ss1, "hybrid": ss2})   # share_limits="auto"
compare.compare_phase_amplitude({"control": rec_a, "patient": rec_b})
compare.compare_scale_policies({"none": ss_raw, "balanced": ss_bal})
compare.compare_routes(df_routes)
compare.compare_series({"PAC": sweep_a, "control": sweep_b},
                       x="lambda", y="corr_slope")

# every window of one record, side by side
compare.compare_recurrence(wr.matrices(), ncols=5)

# one window series per record or condition
compare.compare_window_series({"transient": wa, "ramp": wb, "control": wc},
                              column="independence_ratio")
compare.compare_window_series({k: batch[k] for k in ["sub00", "sub07"]},
                              column="epsilon")
```

No panel is exclusive to comparison mode: every view also exists as a single figure.

---

## 10. CSV export

Every table carries provenance and environment sidecars.

```python
rc.set_config(output_dir="results")

rc.export_frame(df, "metrics")
# -> results/metrics.csv
#    results/metrics_environment.csv    (version, platform, date)
#    results/metrics_provenance.csv     (operation history)

rc.export_recording(rec, "sub01")
rc.export_frame(th.to_frame(), "threshold")
rc.export_frame(rm.describe(), "recurrence")
rc.export_frame(rm.density_profile(), "density_profile")
```

Accumulator for large corpora:

```python
ex = rc.CsvExporter("corpus_metrics")
for subject in corpus:
    ...
    ex.add(subject=subject.id, metric="RR", value=rm.recurrence_rate())
ex.write(note="alpha sweep", n_subjects=len(corpus))
```

---

## 11. Run reports [DD-37]

A plain-text account of what a run did and produced.

```python
report = rc.RunReport("analysis", title="PAC recurrence analysis")
report.parameter(fs=500.0, target_rr=0.05, theiler=1)

report.section("Build state spaces").note("12 subjects, PAC space with 8 Hz smoothing")
report.section("Threshold").result("achieved rate", f"{th.achieved_rr:.4%}")

report.export(df, "threshold_modes", "epsilon per mode")   # exports and registers
report.save(rc.plot_recurrence(rm), "fig_rp.png", "recurrence plot, PAC")

report.warn("envelope has a heavier tail than the phase block")
path = report.write()          # -> outputs/analysis_report.txt
```

The report contains: environment stamp, runtime, parameters, what each section did with its timing, an index of every table and figure with sizes and descriptions, and all the collected warnings.

`.export()` and `.save()` wrap the normal export functions, so registering an artefact costs nothing extra.

---

## 12. Global configuration

```python
rc.set_config(
    output_dir="results",
    memory_budget_gb=8.0,
    n_jobs=16,
    strict_quality=False,
    warn_on_caveat=True,
)
```

---

## 13. Command line

```bash
recurra doctor        # available backends
recurra figures       # figure catalogue
recurra generate --modality pac --n 50 --duration 20 --fs 500 --out corpus/
```

---

## 14. The same code on real data

```python
rec = rc.ingest("subject_01.edf")
rec = rc.filterbank(rec, {"theta": (4, 8), "gamma": (30, 80)})
rec = rc.analytic(rec, sources=["theta", "gamma"])

ss = sm.pac_space(rec, smooth=8.0)
th = rc.threshold(ss, target_rr=0.05, theiler=1, rng=0)
rm = rc.recurrence_plot(ss, th, theiler=1)

rc.save_figure(rc.plot_recurrence(rm), "sub01_rp.png")
rc.export_frame(rm.describe(), "sub01_recurrence")

# across subjects
matrices = {sid: rc.recurrence_plot(build_space(sid), target_rr=0.05) 
            for sid in subjects}
rc.save_figure(compare.compare_recurrence(matrices, ncols=4), "cohort_rp.png")
```

---

## 15. Public API index

Everything exported from the top-level namespace. Anything not listed here is
internal and may change without notice.

### Objects

| Name | What it is |
|---|---|
| `Recording` | time-aligned channels with roles, provenance and quality |
| `Capability` | flags recording what a `Recording` holds |
| `QualityReport` | validation findings, exportable |
| `Step` · `provenance_frame` · `environment_stamp` | provenance records and stamps |
| `StateSpace` | a trajectory with coordinate blocks and weights |
| `Threshold` | a recurrence threshold with its achieved rate |
| `RecurrenceMatrix` | a recurrence structure, lazy or materialised |
| `WindowSpec` · `Window` | how a record is cut into windows |
| `WindowedRecurrence` · `BatchWindowedRecurrence` | windowed and multi-record results |
| `ComparabilityContract` · `ComparabilityReport` | what must match, and what did |
| `LineHistogram` | the three line-length distributions |
| `Invariant` | an estimated invariant with its units and fitted region |
| `ClassificationReport` | cross-validated performance, its spread, and what it means |
| `CorpusIndex` | recordings found on disk, joined to a participants table |
| `RunReport` | a plain-text account of a run |
| `Scaler` | block scaling policies |

### Functions

```python
# ingestion and export
rc.ingest(...)                      rc.export_frame(df, name)
rc.export_recording(rec, name)      rc.CsvExporter(name)

# preprocessing
rc.bandpass  rc.filterbank  rc.notch  rc.resample
rc.analytic  rc.hilbert_components   # the component-level Hilbert helper
rc.scale_array(X, policy=...)        # scale one block
rc.rms_pairwise_distance(X)          # the quantity DD-02 equalises

# synthesis
rc.generate_cfc(...)  rc.generate_corpus(...)
rc.colored_noise(n, beta=1.0)        # 1/f^beta noise, unit variance
rc.lorenz(...)  rc.rossler(...)

# state space
rc.statespace.pac_space / takens / build / from_bands / ...
rc.estimate_tau  rc.estimate_m  rc.estimate_embedding

# recurrence quantification
rc.rqa(rm)  rc.crqa(crp)  rc.line_histogram(rm)  rc.diagonal_profile(crp)
rc.METRIC_NAMES                       # every metric rqa() can return

# dynamical invariants
rc.correlation_dimension(ss)   rc.correlation_sum(ss)
rc.lyapunov_max(ss)            rc.divergence_curve(ss)   rc.k2_entropy(ss)
rc.detrended_fluctuation(x)    rc.hurst_exponent(x)      rc.higuchi_dimension(x)
rc.permutation_entropy(x)      rc.sample_entropy(x)      rc.spectral_entropy(x)

# threshold and recurrence
rc.threshold(...)  rc.cross_threshold(...)
rc.threshold_sampling.sample_pair_distances(X, ...)   # the raw sampler
rc.recurrence_plot / cross_recurrence_plot / joint_recurrence_plot

# windowed
rc.windowed_recurrence / windowed_cross_recurrence / windowed_joint_recurrence
rc.batch_windowed_recurrence(sources, window, ...)

# comparability and groups
rc.check_comparability(units, contract)   rc.contract_from(unit)
rc.describe_units(units)                  # the raw parameter table
rc.feature_matrix(metrics, value)         rc.group_comparison(metrics, group)
rc.rank_metrics(comparison)

# coupling indices
rc.mvl  rc.modulation_index  rc.plv  rc.aac_index  rc.coupling_indices

# figures
rc.plot_attractor / plot_recurrence / plot_window_series / ... (see section 9)
rc.save_figure(fig, name)   rc.save_attractor(ss, name, **kw)   # draw and save
rc.list_figures()

# coupling: the reference every recurrence metric is read against
rc.coupling_indices(phase, amplitude)     # MVL, MI, contrast, preferred phase
rc.modulogram(phase, amplitude)           # the picture that shows coupling
rc.comodulogram(recording)                # which band pair couples
rc.plot_modulogram(...)  rc.plot_comodulogram(...)
wr.metrics(rqa=True, classical=True)      # all of it in one table [DD-106]

# recurrence plots as images
rc.recurrence_image(space, size=256)
rc.export_recurrence_images(spaces, "images/", labels=labels)
rc.attractor_image(space, size=256, coords=(0, 2))
rc.shared_limits(spaces)                 # axes that hold for every space,
                                         # normalised by default [DD-105]
rc.plot_attractor(space, axis_limits=lims)   # so separate files compare [DD-104]
rc.export_attractor_images(spaces, "attractors/", labels=labels)
rc.shape_descriptors(space)        # tail_ratio per block [DD-102]

# corpora on disk
rc.read_corpus("study/")            # -> CorpusIndex

# classification
rc.classify_groups(table, labels, families="both", permutations=200)
rc.select_features(table, "rqa")   rc.build_features(table, level="window")

# parallelism
rc.parallel_map(fn, items, n_jobs=-1, rng=0)    # order preserved, streams by position
rc.resolve_jobs(-1)                             # how many workers that means

# configuration and environment
rc.get_config()  rc.set_config(**kw)  rc.available_backends()  rc.doctor()
rc.resolve_rng(seed)                  # anything -> a Generator
rc.spawn_rngs(seed, n)                # n independent streams [DD-05]
```

### Exceptions

`RecurraError` (base) · `CapabilityError` · `IngestError` · `ParameterError` ·
`QualityError` · `BackendUnavailable` · `MemoryBudgetError`

Each carries enough context to act on [DD-09]; `BackendUnavailable` names the
exact install command for the missing extra. One `except` clause covers them
all:

```python
try:
    ...
except rc.RecurraError as e:
    print(type(e).__name__, e)
```

### Warnings [DD-46]

`RecurraWarning` (base) · `CaveatWarning` · `InferenceWarning` ·
`GeometryWarning` · `ComparabilityWarning` · `ParameterWarning`

The library warns when it does something you did not ask for. Quieten it
without going deaf to numpy and scipy:

```python
import warnings
warnings.simplefilter("ignore", rc.RecurraWarning)        # all of ours
warnings.simplefilter("ignore", rc.InferenceWarning)      # only one kind
```

| Category | Raised when |
|---|---|
| `CaveatWarning` | edge trimming, smoothing, decimation, streaming fallback |
| `InferenceWarning` | a role guessed from a name, a channel chosen by frequency |
| `GeometryWarning` | a dominant block, a heavy-tailed block [DD-02, DD-28] |
| `ComparabilityWarning` | records that are not built the same way [DD-42] |
| `ParameterWarning` | a valid parameter combination unlikely to do what was meant, such as an `nm_ratio` whose locked component falls outside the high band [DD-48] |

Silencing a warning does **not** lose the record: every caveat is also in the
provenance trail and in any run report.

---

## 16. API in preparation 📋

Shape fixed, implementation pending.

### F6 · RQA

```python
metrics = rc.rqa(rm, l_min=2, v_min=2, theiler=1)      # tidy DataFrame
hist = rc.line_histogram(rm)                            # streamed, exact
rc.crqa(crp)                                            # + diagonal profile, lag
```

### F7 · Corpus and parallelism

```python
corpus = rc.Corpus.from_directory("data/", pattern="*.edf", meta_from="bids")
pipe = rc.Pipeline([
    ("bands",     {"bands": {"theta": (4,8), "gamma": (30,80)}}),
    ("analytic",  {}),
    ("space",     {"route": "observable", "spec": "pac"}),
    ("threshold", {"mode": "target_rr", "target_rr": 0.05}),
    ("windows",   {"size": 4.0, "step": 0.5, "unit": "seconds"}),
    ("rqa",       {"metrics": ["RR","DET","LAM","L","ENTR"]}),
])
results = corpus.map(pipe, n_jobs=32, backend="process", resume=True)
```

### F10 · Meta-recurrence (windowing itself is implemented, see section 8b)

```python
series = wr.metrics()                       # already available
meta = rc.meta_recurrence(series, metrics=("DET","LAM","L","RR","ENTR"))
rc.meta_rqa(meta)
rc.detect_transitions(series, method="meta_texture")
```


## Does the separation evolve along the recording?

Aligned windows allow a time-resolved comparison, with the subject kept as the
unit of inference [DD-114]:

```python
course, summary = rc.group_timecourse(
    windows, "group", values=["DET", "LAM", "TT"], by="channel",
    keep="kept", n_permutations=2000)
# summary: auc_peak, peak_window, p_peak (max-statistic), trend, p_trend

curve = rc.classify_timecourse(windows, labels, time="window", keep="kept",
                               families="both")     # one honest CV per window
```

`p_peak` is already corrected for reading the best window of the curve; the
aggregated `classify_groups` remains the primary analysis, this is the *when*.
