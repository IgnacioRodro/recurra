<picture><source media="(prefers-color-scheme: dark)" srcset="_static/recurra-mark-dark.svg"><img alt="recurra" src="_static/recurra-mark.svg" width="56" align="right"></picture>

# Example index

What each example script demonstrates, what data it uses, and every file it
produces. Ordered as `python examples/run_all.py` runs them.

Nothing here is generated until you run something: the package ships with no
output directory.

```bash
python examples/run_all.py              # all six, into ./outputs
python examples/run_all.py --only 3     # just one
```

---

## Summary

| # | Script | Demonstrates | Seconds | Figures | Tables |
|---|---|---|---|---|---|
| 1 | `example_01_pipeline.py` | ingestion, preprocessing, the generator, the scaling problem | 3 | 3 | 5 |
| 2 | `example_02_statespace.py` | every state-space constructor and route | 29 | 23 | 9 |
| 3 | `example_03_recurrence.py` | thresholds and recurrence structures | 15 | 11 | 8 |
| 4 | `example_04_windowed.py` | windowed analysis with overlap | 20 | 12 | 11 |
| 5 | `example_05_comparability.py` | cross-subject comparability, group separation | 155 | 5 | 12 |
| 6 | `example_06_cookbook.py` | 106 recipes covering the whole API | 29 | 39 | 8 |

Every table is written with two sidecars, `*_environment.csv` (versions,
platform, date) and where applicable `*_provenance.csv` (the operation
history), so the file counts above are of *distinct results*, not of files on
disk.

Examples 3, 4 and 5 also write a plain-text run report
(`*_demo_report.txt`) summarising what the run did and produced.

---

## 1. `example_01_pipeline.py` — ingestion, preprocessing, the generator

**Data.** Synthetic throughout. Random arrays for the ingestion shapes; a
30-second PAC signal at 500 Hz with α = 0.85 and a preferred phase of π/2 for
the pipeline; a corpus of twelve subjects with α from 0 to 1 for the last
section.

**Sections.** The environment check; the input shapes and the capability
system refusing an invalid composition; validation of the generator against
the canonical indices; the preprocessing chain with its provenance; the
heterogeneous-scale problem; a multi-subject corpus into one tidy table.

### Figures

| File | Object | What it shows |
|---|---|---|
| `fig_signal_sub01.png` | `Recording` | Four stacked time series: the raw signal, the theta and gamma bands, and the gamma envelope. The vertical scale is per channel, so the envelope's slow modulation is visible next to the fast carrier. |
| `fig_phase_amplitude_sub01.png` | `Recording` | Mean gamma amplitude per theta-phase bin, normalised to sum to one, with the uniform level marked. The distribution behind Tort's modulation index; a flat profile means no coupling. Title carries MI and MVL. |
| `fig_quality_sub01.png` | `Recording` | Left, the standard deviation of every channel, which makes scale differences between raw signal, bands and envelopes obvious. Right, the six quality checks with their verdicts. |

### Tables

| File | One row per | Columns of interest |
|---|---|---|
| `generator_validation.csv` | modality × α (20 rows) | `mvl`, `mi_tort`, `plv`, `aac` — the canonical indices against the α that generated the data. This is the table behind the claim that the generator is monotone in coupling strength. |
| `scale_policy_effect.csv` | policy × amplifier gain | `phase_share`, `amplitude_share`, `dominant_block`. Shows the phase block taking 99.99% of the distance variance at gain 0.01 and the amplitude taking it at gain 100, and both settling at 0.5 under `rms_balanced`. |
| `corpus_coupling_metrics.csv` | subject × metric (24 rows) | tidy long format: `subject`, `alpha_true`, `metric`, `value`. The shape every corpus-level table in the library uses. |
| `sub01_channels.csv` | channel | role, units, sample count, statistics and a digest per channel, so a figure can be traced to the exact data behind it. |
| `sub01_quality.csv` | quality issue | empty when the recording is clean. |

---

## 2. `example_02_statespace.py` — every constructor and route

**Data.** Five synthetic signals, one per coupling modality (PAC, PPC, AAC,
PPA, and an uncoupled control), 40 s at 500 Hz. Plus a Lorenz system of 20 000
points, which serves as ground truth for the delay route: its embedding
dimension is known to be 3.

**Sections.** Embedding parameter estimation across every method combination;
the twenty constructors; the effect of each scaling policy on one space;
comparison of routes on the same signal, with the uncoupled control as a
foil; the λ sweep; the toroidal view; single figures; comparison mode.

### Figures — attractors

| File | Object | What is plotted |
|---|---|---|
| `fig_attractor_pac_observable.png` | `StateSpace`, PAC, 3-D | The joint space (cos φ_theta, sin φ_theta, A_gamma) as a 3-D point cloud, coloured by time. The ring in the horizontal plane is the phase; height is the envelope. |
| `fig_attractor_pac_torus.png` | same space, `mode="torus"` | The same data in polar form: angle is the phase, radius is 1 + normalised amplitude. The ring thickens where the envelope is large, which is what makes coupling visible at a glance. |
| `fig_attractor_pac_hybrid.png` | `StateSpace`, hybrid, 5-D | Phase circle kept intact, envelope delay-embedded in three dimensions. First three coordinates shown. |
| `fig_attractor_pac_timeseries.png` | same space, `mode="time"` | Each coordinate against time, which is how to see that cos and sin oscillate while the envelope drifts. |
| `fig_attractor_ppc_pairs.png` | `StateSpace`, PPC, 4-D | Scatter matrix of two phase circles. Off-diagonal cos-vs-sin panels are perfect circles by construction; the cross panels between the two rhythms show locking as structure and its absence as a filled square. Diagonals show the phase distribution, not the cosine's arcsine shape. |
| `fig_attractor_ppa_pairs.png` | `StateSpace`, PPA, 5-D | The same for the trivariate space: two phase circles plus an envelope. |
| `fig_attractor_aac_envelopes.png` | `StateSpace`, envelopes, 2-D | Theta envelope against gamma envelope. Amplitude-amplitude coupling appears as elongation along the diagonal. |
| `fig_attractor_freq_amp.png` | `StateSpace`, 2-D | Instantaneous frequency against amplitude. Axis limits are robust, because phase slips throw the frequency to tens of hertz; the subtitle says when clipping happened. |
| `fig_attractor_lorenz_observable.png` | `StateSpace`, channels | The Lorenz attractor from its three observable state variables. The reference picture. |
| `fig_attractor_lorenz_takens.png` | `StateSpace`, delay | The same attractor reconstructed from the x coordinate alone by delay embedding, m = 3, τ = 16. Comparing it with the previous figure is the visual proof that the delay route works. |

### Figures — diagnostics

| File | Object | What it shows |
|---|---|---|
| `fig_embedding_diagnostics_lorenz.png` | `EmbeddingParams` | Left, average mutual information against lag with the chosen τ marked. Right, the false-nearest-neighbour fraction against m with the chosen dimension marked. The figure that justifies an embedding choice in a paper rather than asserting it. |
| `fig_scale_report_pac.png` | `StateSpace` | Left, share of distance variance per block with the balanced level marked. Right, tail ratio per block against the unit circle's √2 reference, in red when a block's tail is much heavier. |
| `fig_scale_report_pac_unscaled.png` | same, `scaling="none"` | The same diagnostic without a policy applied, which is where it matters most. |
| `fig_lambda_sweep_pac.png` | DataFrame from `lambda_sweep()` | Geometric descriptors against λ, with the straight line joining the endpoints drawn. Both endpoints are single-block projections that carry no information about coupling; the interior is what matters. Note the endpoints are degenerate — one block has weight exactly zero — so descriptors there measure a lower-dimensional object. |
| `fig_phase_amplitude_pac.png` | `Recording` | As in example 1, for reference beside the recurrence-based views. |
| `fig_envelope_pair_aac.png` | `Recording` | Theta against gamma envelope with the Spearman correlation in the title. |

### Figures — comparison mode

| File | Takes | Panels |
|---|---|---|
| `fig_compare_routes_attractors.png` | six `StateSpace` objects | PAC observable, PAC hybrid, PAC unscaled, PAC on an uncoupled signal, Lorenz observable, Lorenz by delays. Axes are **not** shared here, and the subtitle says why: the spaces have different coordinate blocks. |
| `fig_compare_lambda_attractors.png` | five `StateSpace` objects | The same PAC space at λ = 0, 0.25, 0.5, 0.75, 1. At λ=0 the amplitude weight is zero and the cloud is a flat ring; at λ=1 the phase weight is zero and it collapses to a vertical line. The morph between them is the point. |
| `fig_compare_scale_policies.png` | four `StateSpace` objects | Stacked bars of variance share per block under each policy. |
| `fig_compare_route_metrics.png` | DataFrame from `compare_routes()` | Per route: the coefficient of variation of nearest-neighbour distance (homogeneity of the cloud), the correlation-sum slope (effective dimension), and the fraction of nearest neighbours shared with the reference route. Scale-dependent descriptors such as `nn_mean` are excluded by default because they are expressed in the units of each space [DD-53]. Routes that did not apply to the recording appear in the caption with their reason rather than being dropped. The shared-neighbour panel is the one to read first: it shows that routes over the same data barely agree on who is near whom. |
| `fig_compare_lambda_sweeps.png` | two DataFrames | The λ sweep for a coupled and an uncoupled signal overlaid. The two agree at both endpoints, as they must, and separate in the interior — with the **uncoupled** signal showing the higher effective dimension, because independent blocks form a cartesian product while coupling constrains the joint space onto a lower-dimensional set [DD-02]. |
| `fig_compare_regimes_torus.png` | three `StateSpace` objects | Toroidal view for no coupling, distributed coupling and coupling at a preferred phase. The uniform ring, the smoothly thickened ring and the localised lobe. |
| `fig_compare_regimes_phase_amplitude.png` | three `Recording` objects | The classical phase-amplitude histogram for the same three regimes, on shared axes, with MI and MVL per panel. Put beside the previous figure, it is the classical and the geometric view of the same fact. |

### Tables

| File | One row per | Notes |
|---|---|---|
| `embedding_parameters.csv` | method combination | τ and m for each of `ami`/`acf`/`first_zero` crossed with `fnn`/`cao`, on Lorenz and on an EEG-like signal, with the criterion that fired. AMI+FNN recovers m = 3 for Lorenz. |
| `embedding_curves.csv` | lag or dimension | the raw curves behind the previous table, for replotting. |
| `statespace_constructors.csv` | constructor (20 rows) | route, dimension, point count, block structure, scaling, per-block variance share. The catalogue of what each constructor produces. |
| `scaling_policies.csv` | policy | weights, shares and geometric descriptors for the same space under each policy. |
| `route_comparison.csv` | route | dimension, extent, nearest-neighbour statistics, correlation slope, and agreement of k-neighbours against a reference route. |
| `route_comparison_control.csv` | route | the same on an uncoupled signal, so route differences can be told apart from coupling. |
| `lambda_sweep.csv` | λ value × signal | shares and geometry across the sweep, for coupled and uncoupled. |
| `coupling_regimes.csv` | regime | MI, MVL and tail ratios for the three toroidal panels. |
| `figure_index.csv` | figure | filename, kind (single or comparison), the function that drew it, and its path. |
| `statespace_provenance.csv` | operation | the full history behind one hybrid state space: ingestion, each band-pass with its edge cost, the Hilbert decomposition, the trim, the embedding and the scaling. |

---

## 3. `example_03_recurrence.py` — thresholds and recurrence structures

**Data.** Four state spaces at 2500 points each: PAC at α = 0.9, PAC at α =
0.3, an uncoupled control, and Lorenz. All PAC spaces are built with 8 Hz
envelope smoothing.

**Sections.** Eleven threshold configurations; how accurately the target rate
is met; streaming equivalence and memory planning; recurrence plots per
regime; cross recurrence; joint recurrence with independent thresholds;
figures; provenance and the run report.

### Figures

| File | Object | What is plotted |
|---|---|---|
| `fig_recurrence_pac.png` | `RecurrenceMatrix` | The recurrence plot of the PAC space at a 5% target rate. Black is a recurrent pair. Both axes are time in seconds. Title carries the achieved rate, ε, its mode, and the Theiler window. |
| `fig_recurrence_uncoupled.png` | `RecurrenceMatrix` | The same for the uncoupled control at the same rate. Comparing textures at matched density is the whole reason for fixing the rate rather than ε. |
| `fig_recurrence_lorenz.png` | `RecurrenceMatrix` | Lorenz at the same rate: the block-and-diagonal texture of a deterministic system, quite unlike the two above. |
| `fig_cross_recurrence.png` | `RecurrenceMatrix`, CRP | Theta phase circle against gamma phase circle. Rectangular and not symmetric; asymmetry about the line of identity is what encodes lag and direction. |
| `fig_joint_recurrence.png` | `RecurrenceMatrix`, JRP | The elementwise AND of a 2-D phase subsystem and a 1-D envelope subsystem, each at its own threshold. Sparser than either part by construction. |
| `fig_density_profile_pac.png` | `RecurrenceMatrix` | Local recurrence density along the trajectory with the overall rate marked. A collapse marks a stretch the threshold does not suit. Computed by streaming, so it works for matrices too large to hold. |
| `fig_threshold_diagnostics.png` | `Threshold` | Histogram of sampled pairwise distances with ε drawn on it. Where the threshold sits in the distance distribution, which is the thing a target rate is really choosing. |

### Comparison figures

| File | Panels |
|---|---|
| `fig_compare_recurrence_regimes.png` | PAC α=0.9, PAC α=0.3, uncoupled, Lorenz, all at a 5% target rate. The comparison the whole rate-matching apparatus exists to make honest. |
| `fig_compare_recurrence_rates.png` | The same trajectory at 1%, 5%, 15% and 30%. Shows how much of the apparent structure is a choice of density. |
| `fig_compare_structures.png` | RP, CRP and JRP side by side on the same recording. |
| `fig_compare_thresholds.png` | ε and achieved rate for each of the eleven threshold modes, with the target marked as a tick. |

### Tables

| File | One row per | Notes |
|---|---|---|
| `threshold_modes.csv` | mode (11 rows) | ε, target and achieved rate, sample count, and whether the threshold is per point or per block. |
| `target_rate_accuracy.csv` | target × method (14 rows) | requested against actually achieved rate for `quantile` and `bisect`, from 0.5% to 30%. The worst deviation across the table is the number to quote. |
| `streaming_equivalence.csv` | tile size | whether the reassembled matrix is identical to the dense one, and the rate difference. All identical, difference exactly zero. |
| `memory_plans.csv` | trajectory length | dense and sparse size, chosen storage, tile size, for lengths from 5 000 to 1 800 000 points. An hour of EEG is 3017 GiB dense. |
| `recurrence_plots.csv` | state space | one row per recurrence plot: dimension, ε, target and actual rate, storage. |
| `cross_recurrence.csv` | case | the two cross-recurrence plots, including one against a lagged copy of itself. |
| `joint_recurrence.csv` | signal | independent ε per subsystem, each subsystem's rate, the joint rate, the rate expected if they were independent, and the ratio. |
| `recurrence_provenance.csv` | operation | every step behind the PAC recurrence plot, with the caveats each one raised. |

---

## 4. `example_04_windowed.py` — windowed analysis with overlap

**Data.** A 60-second PAC recording for the main demonstration; three
non-stationary signals (transient, ramped and stationary coupling) for the
time-resolved section; a corpus of eight subjects with α from 0 to 1.

**Sections.** The three ways to declare windows; ten overlapping recurrence
plots from one record; the effect of threshold scope; non-stationary coupling
seen through the windows; every mode windowed; a corpus of subjects × windows.

### Figures

| File | Object | What is plotted |
|---|---|---|
| `fig_window_coverage.png` | `WindowedRecurrence` | One horizontal bar per window against time, with the overlap with the previous window in red. Shows immediately whether the end of the record is analysed and how much consecutive windows share. |
| `fig_window_rate_global.png` | `WindowedRecurrence` | Recurrence rate at each window centre, under a **global** threshold, with each window's extent drawn as a short horizontal segment. Here the rate is informative because ε is fixed. |
| `fig_window_epsilon_per_window.png` | `WindowedRecurrence` | ε at each window centre under a **per-window** threshold. Here ε is the informative series, because the rate is pinned by construction. The two figures together are the argument of DD-40. |
| `fig_window_panel_pac.png` | `WindowedRecurrence` | ε and rate stacked on a shared time axis. |
| `fig_window_independence_transient.png` | `WindowedRecurrence`, JRP | The joint independence ratio across windows for a transiently coupled signal. The ratio is the joint recurrence rate divided by the product of the subsystem rates, so 1.0 means the two recur independently. Ground truth is coupling between 19.8 s and 39.6 s; the curve rises and falls at those times with a latency below the 1 s window step. The subsystem rates are identical inside and outside the transient, so nothing visible to either signal alone has changed — only the coincidence [DD-54]. Grey segments show each window's extent, a reminder that at 75% overlap adjacent points are not independent. |
| `fig_recurrence_window_00/04/09.png` | `RecurrenceMatrix` | Three individual windows drawn alone, so a single window can be inspected without the comparison grid. |

### Comparison figures

| File | Panels |
|---|---|
| `fig_compare_windows_pac.png` | All ten windows of one 60-second record, at a 5% rate each. |
| `fig_compare_nonstationary.png` | Independence ratio against time for transient, ramped and absent coupling. A bell, a monotone rise, and a flat line at 1.0. |
| `fig_compare_scope.png` | Recurrence rate under per-window and global thresholds, overlaid. |
| `fig_compare_subjects_epsilon.png` | ε across windows for three subjects. |

### Tables

| File | One row per | Notes |
|---|---|---|
| `window_specifications.csv` | specification (5 rows) | the same 60 s record cut five ways, with the resulting window count, size, step, and whether the grid spans the record. |
| `window_grid.csv` | window | start and stop in samples and seconds, and the overlap with the previous window. |
| `windowed_metrics_pac.csv` | window | the core output: geometry, threshold, achieved rate, times. This is the table the RQA metrics will slot into. |
| `threshold_scope_comparison.csv` | scope | ε and rate statistics under each scope, the numbers behind DD-40. |
| `windowed_mode_coverage.csv` | configuration (9 rows) | every threshold mode, distance metric, state-space route and structure kind, windowed. Evidence that windowing composes with everything. |
| `nonstationary_windowed_metrics.csv` | window × signal | independence ratio through time for the three regimes. |
| `nonstationary_summary.csv` | signal | mean, minimum and maximum of the ratio. |
| `corpus_windowed_metrics.csv` | subject × window | the corpus in tidy form. |
| `corpus_windowed_summary.csv` | subject | one row per record. |
| `corpus_windowed_joint.csv` | subject × window | the joint-recurrence version, carrying the independence ratio per window. |
| `windowed_provenance.csv` | operation | the history behind the windowed analysis, including the resolved window specification. |

---

## 5. `example_05_comparability.py` — comparability and group separation

**Data.** Three subjects with record lengths of 60, 45 and 72 seconds for the
failure modes. Then twenty subjects, ten per group: controls with α uniform in
[0, 0.25], the study group in [0.55, 0.85], record lengths drawn from 40, 45
and 50 seconds. Unequal lengths are the point, not an accident.

**Sections.** The three ways comparability breaks silently; the library
detecting it; the fix with a fixed window grid and an explicit contract; the
two-group corpus; separation window by window; the feature matrix.

### Figures

| File | Object | What is plotted |
|---|---|---|
| `fig_comparability_bad.png` | `ComparabilityReport` | Left, one bar per construction axis coloured by verdict, with a legend naming the four statuses; right, the metric families the observed matching licenses, each withheld one labelled with the requirement it lacks. For a batch built with `n_windows` on unequal records, `n_points` and the window axes are red, so `line_length` and `entropy` are withdrawn while `line_ratio` survives — DET and LAM are ratios and N cancels. Note `epsilon` is grey rather than red: with per-window thresholds a record has no single epsilon, so the axis is *absent* rather than violated, and the recurrence-rate family is withheld for that reason [DD-55]. |
| `fig_comparability_good.png` | `ComparabilityReport` | The same after fixing the window grid: every axis green, every family licensed. |
| `fig_group_comparison.png` | DataFrame from `group_comparison()` | Top, the mean of each group per window with a one-standard-deviation band. Bottom, the AUC per window, dark where it survives Bonferroni correction. The x axis is the window index, because window k is compared only with window k. |
| `fig_feature_matrix.png` | DataFrame from `feature_matrix()` | Subjects by windows as a heat map, ordered by group with the boundary drawn. This is literally the classifier's input. |
| `fig_compare_group_series.png` | four `WindowedRecurrence` objects | The independence ratio through time for two controls and two study subjects. |

### Tables

| File | One row per | Notes |
|---|---|---|
| `comparability_failure_modes.csv` | failure × subject | the same request producing different dimensions, window durations and effective sampling rates per subject. |
| `comparability_report_bad.csv` | axis | verdict, required value, distinct values observed, and which subjects are the offenders. |
| `comparability_report_good.csv` | axis | the same after the fix. |
| `metric_validity_bad.csv` | metric family | what each family requires and whether the observed matching supplies it. The table that says DET and LAM survive while L and Lmax do not. |
| `metric_validity_good.csv` | metric family | the same after the fix. |
| `comparability_contract.csv` | axis | the parameters held fixed. This belongs next to any published result. |
| `corpus_group_metrics.csv` | subject × window | the corpus with group labels and true α. |
| `group_comparison.csv` | window × metric | group sizes, means, standard deviations, difference, Cohen's d, AUC, raw p-value, the number of tests, and the Bonferroni-corrected p-value. |
| `metric_ranking.csv` | metric | metrics ordered by how well they separate the groups. |
| `feature_matrix.csv` | subject | the pivot, one column per window. |
| `subject_level_summary.csv` | subject | group, true α, and the mean recurrence-based index. The Spearman correlation between the last two columns is the parallel the project is looking for. |

---

## 6. `example_06_cookbook.py` — 106 recipes

**Different in kind from the others.** Split into Spyder cells with `# %%`
markers, meant to be run one at a time with Ctrl+Enter rather than as a
script. Cell 0 builds the shared objects; the rest are largely independent.

**Data.** Short synthetic signals throughout, deliberately small so that a
cell returns in a second or two.

**Coverage.** Every command and configuration in the library: the eleven
ingestion shapes and their errors, quality and capabilities, filtering and
Hilbert options, the five generator modalities with every parameter, the
canonical indices with surrogates, the twenty state-space constructors across
three routes, embedding estimation, the scaling policies, transforms, route
comparison and λ sweeps, the nine threshold modes with both scopes and four
distance metrics, RP/CRP/JRP with three storage strategies, memory planning,
the three ways to declare windows, windowing across every mode, the
multi-subject batch, comparability contracts and group separation, all
seventeen single figures and eight comparison figures, CSV export, run
reports, configuration and reproducibility.

**Output.** 39 figures named `cb_*.png` and 8 tables, written to
`cookbook_out/` rather than `outputs/` so they do not mix with the other
examples. The tables are `cb_recurrence_summary` (one recurrence plot
described), `cb_window_metrics` (per-window metrics), `cb_group_comparison`
(two-group statistics), `cb_corpus_metrics` (the accumulator across subjects),
`cb_recording_*` (channel and quality tables), `cb_report_table` (registered
through a run report) and `cb_cookbook_artifact_index`. Every public name in `recurra.__all__` is exercised, which the test
suite checks.

---

## Where each figure function is documented

The figures above are produced by the functions catalogued in
[`ARCHITECTURE.md`](ARCHITECTURE.md) and shown in use in
[`EXAMPLES.md`](EXAMPLES.md) section 9. Every single figure takes one object
and every comparison figure takes a mapping of already-built objects; neither
kind generates data, so any of them can be reproduced on real recordings with
the same call.
