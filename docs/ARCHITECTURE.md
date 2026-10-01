<picture><source media="(prefers-color-scheme: dark)" srcset="_static/recurra-mark-dark.svg"><img alt="recurra" src="_static/recurra-mark.svg" width="56" align="right"></picture>

# Architecture — `recurra`

**Living document.** Taxonomies, objects, contracts, implementation status.

Status key: ✅ implemented and tested · 🔨 in progress · 📋 specified, not written

---

## 1. Module map

```
src/recurra/
├─ __init__.py            ✅  public API, doctor()
├─ _version.py            ✅
├─ capabilities.py        ✅  Capability, require()                    [DD-03]
├─ config.py              ✅  Config, RNG policy, available backends   [DD-05]
├─ diagnostics.py         ✅  QualityReport, check_channels            [DD-06]
├─ exceptions.py          ✅  error hierarchy                          [DD-09]
├─ provenance.py          ✅  Step, digests, environment stamp         [DD-04]
├─ report.py              ✅  RunReport -> plain-text run summary      [DD-37]
├─ comparability.py       ✅  contracts, checks, group tests    [DD-42..44]
├─ cli.py                 ✅  recurra doctor | figures | generate
│
├─ core/
│   └─ recording.py       ✅  Recording, copy-on-write                 [DD-07]
│
├─ io/
│   ├─ ingest.py          ✅  ingest() -- 11 input shapes              [DD-08]
│   ├─ readers.py         ✅  csv, npy, npz, hdf5, edf              [DD-88]
│   ├─ corpus.py          ✅  read_corpus, CorpusIndex             [DD-89]
│   └─ export.py          ✅  CSV with sidecars, CsvExporter           [DD-10]
│
├─ preprocess/
│   ├─ filters.py         ✅  bandpass, filterbank, notch, resample    [DD-11]
│   ├─ analytic.py        ✅  Hilbert -> phase / envelope / freq       [DD-12]
│   └─ scaling.py         ✅  Scaler, ScaleReport               [DD-02, DD-28]
│
├─ signals/
│   ├─ noise.py           ✅  coloured noise, target SNR
│   ├─ cfc.py             ✅  generate_cfc, generate_corpus            [DD-13]
│   └─ systems.py         ✅  Lorenz, Rossler, Henon, van der Pol, Mackey-Glass
│
├─ statespace/
│   ├─ core.py            ✅  StateSpace, CoordGroup                   [DD-21]
│   ├─ embedding.py       ✅  tau (AMI/ACF/first-zero), m (FNN/Cao) [DD-22..24]
│   ├─ builders.py        ✅  16 constructors, three routes            [DD-25]
│   └─ compare.py         ✅  compare_routes, lambda_sweep             [DD-26]
│
├─ threshold/
│   ├─ core.py            ✅  Threshold
│   ├─ sampling.py        ✅  Theiler-aware pair sampling              [DD-32]
│   └─ estimate.py        ✅  8 modes, quantile and bisection          [DD-35]
│
├─ recurrence/
│   ├─ distance.py        ✅  BLAS squared-norm kernels                [DD-33]
│   ├─ memory.py          ✅  MemoryPlan, budget enforcement           [DD-34]
│   ├─ core.py            ✅  RecurrenceMatrix, tile streaming         [DD-36]
│   └─ builders.py        ✅  RP, CRP, JRP                       [DD-19, DD-20]
│
├─ metrics/
│   └─ classic.py         ✅  MVL, MI (Tort), PLV, AAC, surrogates
│
├─ viz/
│   ├─ style.py           ✅  palette, sizes, save_figure        [DD-30, DD-31]
│   ├─ attractor.py       ✅  2d/3d/pairs/time/torus                   [DD-27]
│   ├─ diagnostics.py     ✅  embedding, scales, lambda sweep, quality
│   ├─ coupling.py        ✅  phase-amplitude, signal, envelopes
│   ├─ recurrence.py      ✅  recurrence plot, density, threshold      [DD-38]
│   ├─ panels.py          ✅  plot with its signals, joint decomposed, RQA summary [DD-120]
│   ├─ rqa.py             ✅  line histograms, diagonal profile
│   ├─ windowed.py        ✅  coverage, window series, panel
│   ├─ groups.py          ✅  group comparison, feature matrix, comparability
│   └─ compare.py         ✅  comparison mode                          [DD-31]
│
├─ windowed/
│   ├─ spec.py            ✅  WindowSpec, Window, resolution       [DD-39]
│   ├─ recurrence.py      ✅  WindowedRecurrence, batch      [DD-40, DD-41]
│   └─ meta.py            ✅  meta-space and meta-RP/CRP/JRP of window metrics [DD-119]
│
├─ parallel/
│   └─ runner.py          ✅  parallel_map, resolve_jobs      [DD-81, DD-82]
│
├─ images/
│   └─ export.py          ✅  recurrence and attractor images [DD-90..92, DD-103]
│
├─ rqa/
│   ├─ histogram.py       ✅  LineHistogram, streaming carry      [DD-56, DD-57]
│   └─ metrics.py         ✅  rqa, crqa, diagonal_profile         [DD-58, DD-59]
├─ dynamics/
│   ├─ scaling.py         ✅  find_scaling_region, local slopes       [DD-61]
│   ├─ invariants.py      ✅  D2, lambda_1, K2, Invariant       [DD-62, DD-63]
│   ├─ complexity.py      ✅  DFA, Hurst, fractals, entropies, Hjorth, LZ
│   └─ measures.py        ✅  dynamics_measures -> a table row  [DD-71, DD-72]
├─ costas/                📋  F12
├─ classify/
│   ├─ features.py        ✅  family selection, aggregation      [DD-83]
│   └─ evaluate.py        ✅  classify_groups, classify_timecourse [DD-84..87, DD-114]
```

---

## 2. Taxonomies

### 2.1 Capabilities (`Capability`) ✅

| Flag | Meaning | Granted by |
|---|---|---|
| `RAW` | raw time-domain samples | signal ingestion |
| `BANDS` | band-limited components | `bandpass`, `filterbank` |
| `PHASE` | instantaneous phase | `analytic` |
| `AMPLITUDE` | envelope | `analytic` |
| `INST_FREQ` | instantaneous frequency | `analytic` |
| `STATESPACE` | explicit point cloud | `build_statespace` |
| `THRESHOLD` | threshold fixed | `threshold` |
| `RECURRENCE` | recurrence structure available | `recurrence_plot` |

### 2.2 Channel roles ✅

`raw` · `band` · `phase` · `amplitude` · `inst_freq` · `state`

Name-based inference only when `roles` is not declared; every inference is recorded as a caveat and warned about.

### 2.3 Input shapes ✅

| # | Input | Capabilities | Tested |
|---|---|---|---|
| I1 | `ndarray (T,)` | RAW | ✅ |
| I2 | `ndarray (T,D)` | RAW | ✅ |
| I3 | dict of bands | RAW·BANDS | ✅ |
| I4 | dict of Hilbert components | PHASE·AMPLITUDE | ✅ |
| I5 | complex analytic signal | PHASE·AMPLITUDE | ✅ |
| I6 | `ndarray (T,K)` as state space | STATESPACE | ✅ |
| I7 | precomputed recurrence matrix | RECURRENCE | 📋 |
| I8 | file (csv/npy/npz/hdf5/edf) | from content | ✅ all, EDF verified against real files |
| I9 | BIDS-style directory + participants.tsv | from content | ✅ read_corpus |
| I10 | `Corpus` object passed to `ingest` | per record | 📋 |
| I11 | memmap / lazy | RAW | 📋 |
| — | `Recording` (passthrough) | unchanged | ✅ |

### 2.4 Coupling modalities ✅

| Modality | Couples | State space | Preferred structure |
|---|---|---|---|
| `pac` | slow phase → fast amplitude | (cos φ, sin φ, A) | RP |
| `ppc` | phase ↔ phase (n:m) | two circles | CRP |
| `aac` | envelope ↔ envelope | independent spaces | JRP |
| `ppa` | phase-phase-amplitude | 5-D | tensor (F11) |
| `none` | negative control | — | — |

### 2.5 State-space routes ✅ [DD-25]

`observable` · `delay` · `hybrid` · `external`

### 2.6 Scaling policies ✅ [DD-02]

`rms_balanced` (default) · `empirical_balanced` · `weighted`+λ · `per_block_threshold` · `rank` · `zscore` · `robust` · `minmax` · `custom` · `none`

### 2.6b RQA metrics ✅ [DD-58, DD-59]

`RR` · `DET` · `L` · `Lmax` · `DIV` · `ENTR` · `ENTR_norm` · `LAM` · `TT` ·
`Vmax` · `W` · `Wmax` · `ENTW` · `RTE`

Denominator conventions: `theiler_corrected` (default) · `all`

### 2.6c Dynamical and complexity measures ✅ [DD-61, DD-62, DD-71..73]

**Trajectory** (expensive, opt-in): `D2` · `lambda_1` · `K2`

**Scalar, per coordinate block**: `higuchi_fd` · `katz_fd` · `petrosian_fd` ·
`svd_entropy` · `permutation_entropy` · `sample_entropy` · `spectral_entropy` ·
`DFA_alpha` · `hurst` · `lempel_ziv` · `hjorth_activity` · `hjorth_mobility` ·
`hjorth_complexity`

A phase-circle block contributes its instantaneous frequency, an amplitude
block its amplitude [DD-71].

Units: `per_second` (default for rates) · `per_sample` · `dimensionless`

### 2.7 Threshold modes ✅ [DD-35]

| Mode | Epsilon from | Notes |
|---|---|---|
| `target_rr` *(default)* | quantile or bisection to a target rate | reports achieved rate |
| `fixed` | user value | still reports what it achieves |
| `percentile` | distance quantile | rejects values outside [0,1] |
| `fan` | distance to k-th neighbour | per point, asymmetric matrix |
| `std_fraction` | fraction of attractor RMS radius | |
| `maxdist_fraction` | fraction of maximum distance | |
| `per_block` | one epsilon per block, Chebyshev | scale-free |
| `adaptive_dim` | target rate + dimension diagnostic | warns when epsilon swallows the cloud |

Scopes: `global` · `per_record` · `per_window` · `per_block`

### 2.8 Distance metrics ✅

`euclidean` (BLAS-accelerated) · `chebyshev` · `manhattan` · `cosine`

### 2.9 Storage strategies ✅ [DD-36]

| Store | Holds | When |
|---|---|---|
| `memory` | the full matrix | fits in half the budget |
| `sparse` | recurrent pairs (CSR) | low rate, dense too large |
| `none` | nothing; streams tiles | anything larger |
| `auto` | chooses from the budget | default |

### 2.10 Non-stationarity ✅

`transient` · `ramp` · `intermittent` · `switching` · `None`, or a per-sample array/callable of α.

### 2.11 Reference systems ✅

Lorenz (D₂ ≈ 2.06, λ₁ ≈ 0.906) · Rossler · Henon · van der Pol · Mackey-Glass. Integrated at `rtol=1e-9` with the transient discarded and success verified.

---

## 3. Object contracts

### `Recording` ✅

```python
Recording(channels, fs, roles, t0, units, meta, provenance, quality, caps)
```
Frozen, shared buffers, read-only arrays.

`.names` · `.n_channels` · `.n_samples` · `.duration` · `.subject` · `.get(name)` · `.by_role(role)` · `.select(names)` · `.as_array(names)` · `.with_channels(...)` · `.with_meta(...)` · `.with_step(...)` · `.channel_frame()` · `.provenance_frame()` · `.to_frame()` · `.summary()`

### `StateSpace` ✅

```python
StateSpace(coords, groups, fs, t0, route, scaling, scale_report, provenance, meta)
CoordGroup(label, start, stop, kind, metric, weight, source, params)
```

Combined distance `d² = Σ_g w_g · d_g²`, which by [DD-21] is Euclidean on `.weighted_coords`.

`.n_points` · `.dim` · `.labels` · `.block(label)` · `.blocks()` · `.weights` · `.weighted_coords` · `.rescale(policy, lambda_)` · `.reweight(λ)` · `.subsample(...)` · `.window(a,b)` · `.split(labels)` · `.reduce(pca|umap)` · `.scale_frame()` · `.describe()` · `.summary()`

### `Threshold` ✅

```python
Threshold(value, mode, metric, scope, theiler, target, achieved_rr,
          n_samples, per_block, block_labels, diagnostics)
```

`value` is a scalar, a per-point array (`fan`), or accompanied by a per-block tuple. `.is_pointwise` · `.is_per_block` · `.scalar` · `.to_frame()` · `.summary()`

**`achieved_rr` is the number a paper should report**, not the requested target.

### `RecurrenceMatrix` ✅

```python
RecurrenceMatrix(coords, threshold, coords2, kind, metric, theiler, dtype,
                 tile_size, store, groups, fs, t0, meta, provenance, memory_plan)
```

| Method | Returns |
|---|---|
| `.shape` · `.n_rows` · `.n_cols` · `.dim` · `.is_symmetric` | shape |
| `.tiles(tile_size)` | generator of `Tile`, never allocates the whole matrix |
| `.materialize()` / `.matrix` | dense array, budget-checked |
| `.to_sparse()` | CSR of recurrent pairs |
| `.recurrence_rate()` | streamed, exact, Theiler-corrected denominator |
| `.density_profile(n_bins)` | local density along the trajectory |
| `.describe()` · `.provenance_frame()` · `.summary()` | tables |

`_JointRecurrenceMatrix` adds `.parts`, `.thresholds`, `.subsystem_rates()` and `.independence_ratio()`.

### `MemoryPlan` ✅

`n_rows` · `n_cols` · `dim` · `dtype` · `tile_size` · `backend` · `store` · `matrix_gb` · `tile_gb` · `budget_gb` · `sparse_gb` · `note`

### `WindowSpec` and `Window` ✅ [DD-39]

```python
WindowSpec(size, step, overlap, n_windows, unit, align, drop_last, min_samples)
Window(index, start, stop, fs, t0)
```

Three equivalent forms: `size`+`step`, `size`+`overlap`, or `n_windows`+`overlap`. `.resolve(n_samples, fs)` returns `(windows, caveats)`; `.describe()` and `.summary()` report the grid and anything left uncovered.

### `WindowedRecurrence` ✅ [DD-40, DD-41]

```python
WindowedRecurrence(sources, windows, thresholds, kind, scope,
                   build_kwargs, subject, caveats, provenance, keep)
```

| Method | Returns |
|---|---|
| `len()` · `[i]` · iteration | window count, matrix on demand, `(Window, RecurrenceMatrix)` pairs |
| `.matrices()` | all matrices keyed by window label |
| `.metrics()` | tidy DataFrame, one row per window |
| `.series(column)` | one metric against window centre time |
| `.window_frame()` · `.describe()` · `.provenance_frame()` · `.summary()` | tables |

`BatchWindowedRecurrence` holds one per record and concatenates their metrics; `.n_matrices` counts subjects times windows.

### `ComparabilityContract` and `ComparabilityReport` ✅ [DD-42, DD-43]

```python
ComparabilityContract(dim, route, scaling, blocks, m, tau, fs, n_points,
                      metric, theiler, threshold_mode, target_rr, epsilon,
                      window_samples, window_step, tolerance, strict, notes)
```

`check_comparability(units, contract)` accepts a mapping, a sequence or a `BatchWindowedRecurrence`. The report exposes `.ok`, `.violations`, `.matched()`, `.metric_validity()`, `.to_frame()`, `.raise_if_violated()` and `.summary()`.

`contract_from(reference_unit)` derives a contract from an analysis already done, to apply to the rest.

Group tools: `feature_matrix()`, `group_comparison()`, `rank_metrics()`, and on the batch itself `.check_comparability()`, `.metrics_by_window()`, `.group_comparison()`.

### `Invariant` and `ScalingRegion` ✅ [DD-61, DD-62]

```python
Invariant(name, value, units, fs, method, region, curve, diagnostics)
ScalingRegion(start, stop, slope, intercept, r_squared, slope_std,
              x_lo, x_hi, n_points, criterion, warning)
```

`.to_frame()` flattens both into one exportable row; `.summary()` prints the
value with the region and any warning; `float(invariant)` gives the number.

### `RunReport` ✅ [DD-37]

`.section(title)` · `.note(text)` · `.result(label, value)` · `.parameter(**kw)` · `.warn(text)` · `.export(frame, name, description)` · `.save(fig, filename, description)` · `.table(...)` · `.figure(...)` · `.artifact_frame()` · `.render()` · `.write()`

### Others ✅

`Step` · `QualityReport` · `FilterDesign` · `ScaleReport` · `SyntheticSignal` · `EmbeddingParams` · `Tile`

---

## 4. Code conventions

- **Immutability.** No library function mutates its input. Every transformation returns a new object.
- **RNG.** Never global `np.random.*`. Always explicit `rng` via `resolve_rng`; parallel via `spawn_rngs`.
- **Errors.** Specific type plus enough context to fix it. Never bare `except:`; always `except Exception`.
- **Units.** Explicit wherever ambiguous.
- **Optional dependencies.** Lazy import inside the function, with `BackendUnavailable` naming the exact install command.
- **Docstrings.** NumPy style. Design decisions referenced by ID (`[DD-02]`) in the module docstring.
- **Typing.** `from __future__ import annotations` everywhere; strict mypy on the core.
- **Language.** English throughout, enforced by tests [DD-30].

---

## 5. Test suite

**509 test functions, all green** (more cases once parametrised).

| File | Covers |
|---|---|
| `test_capabilities.py` | flag composition, refusal on missing capability |
| `test_recording.py` | copy-on-write, read-only arrays, provenance |
| `test_ingest.py` | input shapes, roles, quality, CSV round trip |
| `test_preprocess.py` | filter order, band isolation, edges, full chain |
| `test_scaling.py` | **DD-02**: circle √2, balancing, dominance, λ, gain invariance |
| `test_signals.py` | spectral slopes, monotonicity in α, reproducibility, parallel RNG |
| `test_export.py` | sidecars, environment stamp, accumulator |
| `test_statespace.py` | **DD-21** identity, 16 constructors, three routes, transforms |
| `test_threshold.py` | **DD-32** Theiler sampling, all 8 modes, target accuracy |
| `test_recurrence.py` | **DD-33** kernel equivalence, **streaming = dense bit for bit**, Theiler bands, RP/CRP/JRP, memory refusal |
| `test_viz.py` | figure registry, input immutability, shared limits, **DD-38** pooling, **DD-30** language guards |
| `test_report.py` | sections, artefact registration, rendering, language |
| `test_comparability.py` | **DD-42** the three failure modes detected, contracts and violations, **DD-43** metric-family licensing, **DD-44** window-by-window group separation, feature matrices |
| `test_rqa.py` | **DD-56** carry logic and streaming equivalence over 48 combinations, **DD-58/59** conventions, ground truth on noise, sine and Lorenz, CRQA lag recovery, integration with the windowed table |
| `test_dynamics.py` | **DD-61** scaling-region location, **DD-62** unit conversion, **DD-63** the guards for numerically degenerate curves, plus ground truth for D2, lambda_1, K2, DFA, Higuchi and three entropies |
| `test_validation.py` | **DD-64/65/66/67** the harmonic artefact exists and is monotone; at equal modulation index DET and the diagonal family distinguish genuine coupling from it, the vertical family points the other way; every signal has a unique seed |
| `test_parallel.py` | **DD-81** results identical across job counts, streams derived by position, the worker is picklable; **DD-82** the serial path never needs joblib |
| `test_classify.py` | **DD-83** folds never straddle a subject and shuffled labels score at chance, **DD-85** imbalance reported with a baseline, **DD-86** repeats give a spread, **DD-87** the permutation null sits at chance |
| `test_realdata.py` | **DD-88** genuine EDF round trips, units, metadata and annotations kept, mixed rates named; **DD-89** the corpus join is reported, orphan rows and duplicate subjects caught |
| `test_images.py` | **DD-90** pooling preserves the rate where resampling would not, fixed output size across unequal records; **DD-91** the scale is shared and per-image normalisation is marked; **DD-92** the decimation route is recorded |
| `test_docs.py` | **DD-45**: the living documents stay in step -- decision index, citations, module map, figure catalogue, test count, example scripts, public API coverage |
| `test_windowed.py` | **DD-39** three spec forms and exact window counts, **DD-40** scope behaviour, **DD-41** laziness, every threshold mode and metric windowed, RP/CRP/JRP, multi-subject batch with unequal lengths |

Per-test index: [`docs/TESTS.md`](TESTS.md), generated by
`python tools/generate_test_index.py`.
Per-example index of what each script produces:
[`docs/EXAMPLE_INDEX.md`](EXAMPLE_INDEX.md).

### Ground-truth tests already in place

| Test | Expected | Obtained |
|---|---|---|
| RMS pairwise distance of the unit circle | √2 = 1.4142 | 1.4142 |
| `tail_ratio` of the unit circle | √2 | 1.411 |
| Coloured-noise spectral slope | −β | within 0.25 |
| Generator monotonicity α → MVL, MI | ρ = 1 | 1.00 |
| Lorenz embedding dimension (AMI + FNN) | 3 | 3 |
| Sine recurrence diagonals | multiples of fs/f₀ | exact, empty between |
| Tiled vs dense matrix | identical | bit for bit |
| Recurrence rate vs target | ≈ target | worst deviation 0.0036 |
| Line histograms, streaming vs dense | identical | 48/48 combinations |
| White noise DET at l_min=2 | 1−(1−RR)² | within 15% |
| Lorenz DET, LAM | > 0.99 | 0.9999, 0.998 |
| DET ordering: noise < sine < Lorenz | strict | 0.10 < 0.97 < 1.00 |
| CRQA peak offset vs true lag | exact | 40 of 40 |
| Lorenz correlation dimension, delay embedding | 2.05 | 1.996, unflagged |
| Rossler correlation dimension | 1.81 | 1.766 |
| Lorenz largest Lyapunov exponent | 0.906 /s | 0.91 /s, 3 samplings |
| Same exponent, one sampling, 12 streams | should agree | 0.50 to 1.37 |
| Higuchi of an oversampled envelope | 1 (smooth) | 1.04, and 1.92 at its own scale |
| D2 of a built PAC space | the construction | 1.98 on all 98, flagged |
| DFA exponent, white/pink/Brownian | 0.5 / 1.0 / 1.5 | 0.512 / 1.003 / 1.489 |
| Higuchi dimension, noise / sine | 2 / 1 | 2.001 / 1.050 |
| Hjorth mobility of a sinusoid | 2·pi·f/fs | 0.1569 vs 0.1571 |
| Lempel-Ziv, noise / sine | 1 / low | 1.04 / 0.08 |
| Harmonic artefact fakes PAC | MI comparable to real coupling | 0.0027 at contamination 1.0 |
| DET separates artefact at equal MI | detectable | t = 3.31, p = 0.011 |
| Detection floor, MI vs DET | MI more sensitive | 0.30 vs 0.90 at 20 dB |
| Noise robustness, dynamics vs RQA | dynamics hold at -5 dB | 0.50 vs 0.70 or never |
| Combining families, cross-validated | does not help | 0.776 vs 0.824 at alpha 0.15 |
| Univariate measure, mismatched pairing | must not separate | separates at 1.00 -- it is not a coupling measure |
| Single-window floor, RQA | ~2-4 s | LAM AUC 0.90 at 2 s |
| Single-window floor, SVD entropy | ~1 s | AUC 0.92 at 1 s |
| PPC: L vs PLV | recurrence ahead | AUC 1.00 vs 0.77 at alpha 0.3 |
| PPA: RQA vs modulation index | only RQA works | AUC 1.00 vs 0.61 |
| Shortest working window, LAM vs DET | LAM shorter | 2 s vs 16 s |
| Length invariance of DET, LAM, L | invariant | drift under 2% |
| Metrics stable against window length | few | only RR, DET, LAM |
| Shortest window that detects coupling | short | 1 s (TT, AUC 1.000) |
| Windowed joint ratio vs true alpha | monotone | Spearman +1.000 |
| Transient coupling onset/offset | 19.8 / 39.6 s | detected 19.8 / 39.8 s |
| `n_windows=k` | exactly k, spanning the record | exact |
| Group separation, 20 subjects | detectable | AUC 0.972, d 2.25 |
| Recurrence index vs true coupling | monotone | Spearman +0.937 |

Still to come: DET and LAM checked against published reference values for the standard systems.

---

## 6. Figure catalogue [DD-31]

**Single** — one object, one figure. In `recurra.viz`:

| Function | Takes |
|---|---|
| `plot_attractor` | `StateSpace` — modes `auto`/`2d`/`3d`/`pairs`/`time`/`torus` |
| `plot_embedding_diagnostics` | `EmbeddingParams` — τ and m curves with the criterion marked |
| `plot_scale_report` | `StateSpace` — variance share and tail ratio per block |
| `plot_lambda_sweep` | DataFrame from `lambda_sweep()` |
| `plot_quality` | `Recording` — per-channel scale and issues |
| `plot_phase_amplitude` | `Recording` — the distribution behind Tort's MI |
| `plot_signal` | `Recording` — stacked time series |
| `plot_envelope_pair` | `Recording` — two envelopes against each other |
| `plot_recurrence` | `RecurrenceMatrix` — RP, CRP or JRP |
| `plot_window_coverage` · `plot_window_series` · `plot_window_panel` | `WindowedRecurrence` |
| `plot_group_comparison` | DataFrame from `group_comparison()` |
| `plot_feature_matrix` | DataFrame from `feature_matrix()` |
| `plot_comparability` | `ComparabilityReport` |
| `plot_line_histogram` | `LineHistogram` — the three distributions |
| `plot_recurrence_panel` | `RecurrenceMatrix` — RP, CRP, JRP or meta-RP with its trajectories in the margins and, optionally, its RQA measures [DD-120] |
| `plot_joint_recurrence` | joint `RecurrenceMatrix` — each subsystem, the conjunction and a coloured overlay, with the JRR ratio [DD-120] |
| `plot_rqa_summary` | `RecurrenceMatrix` — the plot, both line-length distributions with their floors, and the measures [DD-120] |
| `plot_diagonal_profile` | DataFrame from `diagonal_profile()` |
| `plot_density_profile` | `RecurrenceMatrix` — local density along time |
| `plot_threshold_diagnostics` | `Threshold` — epsilon in the distance distribution |

**Comparison** — a mapping of already-built objects. In `recurra.viz.compare`:

| Function | Takes |
|---|---|
| `compare_attractors` | `Mapping[str, StateSpace]`, `share_limits=True` |
| `compare_recurrence` | `Mapping[str, RecurrenceMatrix]` |
| `compare_thresholds` | `Mapping[str, Threshold]` |
| `compare_phase_amplitude` | `Mapping[str, Recording]` |
| `compare_scale_policies` | `Mapping[str, StateSpace]` |
| `compare_routes` | DataFrame from `compare_routes()` |
| `compare_window_series` | `Mapping[str, WindowedRecurrence]` |
| `compare_series` | `Mapping[str, DataFrame]` — generic |

`recurra.viz.list_figures()` and `recurra figures` print the catalogue. `save_figure(fig, name)` is a separate step: drawing does not save.

---

## 7. Status and what is next

The analytical core is complete and has carried one real study end to end
(49 children, 32 channels, two recording blocks): ingestion of precomputed
Hilbert arrays, windowed recurrence and RQA per channel and band, dynamical
measures, parallel batches, subject-level classification with permutation
tests and timecourses.

Still specified and not written: F11 higher-order recurrence tensors (the
natural structure for `ppa`), F12 Costas arrays, input shapes I7 (a
precomputed recurrence matrix), I10 (a `Corpus` object passed to `ingest`)
and I11 (memory-mapped, lazy raw data), the GPU backend of DD-16, and the
RQA measures planned in F6 but never written (T1, T2, TREND, CLEAR and
recurrence-network measures).

The study also wrote, outside the library, several pieces that are general
and belong inside it. They are listed, with their reasoning and a proposed
API, in [`docs/ROADMAP.md`](ROADMAP.md).
