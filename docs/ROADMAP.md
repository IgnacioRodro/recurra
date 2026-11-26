<picture><source media="(prefers-color-scheme: dark)" srcset="_static/recurra-mark-dark.svg"><img alt="recurra" src="_static/recurra-mark.svg" width="56" align="right"></picture>

# Roadmap — `recurra`

**Living document.** What is specified and not written, what the first real
study needed and had to write outside the library, and what has to happen
before a public release. Decisions referenced by ID live in
[`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md); the module map and the status of
every module live in [`ARCHITECTURE.md`](ARCHITECTURE.md).

State at the 0.25.0 release: every phase except F11 and F12 implemented and
tested (the test count lives in [`ARCHITECTURE.md`](ARCHITECTURE.md), where a
test keeps it true); one complete real study carried end to end (49 children,
32-channel EEG at 500 Hz, two recording blocks, delta and theta envelope
dynamics, subject-level classification).

---

## 1. Specified, not written

| Item | What | Notes |
|---|---|---|
| F11 | Higher-order recurrence tensors | The natural structure for the `ppa` modality (phase-phase-amplitude), today handled as a 5-D point cloud |
| F12 | Costas arrays | Module slot reserved in the architecture map |
| I7 | `ingest` a precomputed recurrence matrix | Capability `RECURRENCE` without a state space; RQA must run on it |
| I10 | `ingest` a `Corpus` object | `read_corpus` already returns a `CorpusIndex`; the missing piece is passing it straight to the batch layer |
| I11 | Memory-mapped, lazy raw data | For corpora that do not fit in memory; the recurrence layer is already lazy |
| DD-16 | GPU backend (PyTorch) | Replaced by DD-125 in 0.26: the torch engine for recurrence plots, `engine="auto"`, the same answer as the CPU by construction. Pending: the CUDA battery and the benchmark on the target GPUs; cross and joint plots, FAN and per-block thresholds on the device; thresholds of `windowed_recurrence` on the device |
| F6 leftovers | RQA measures `T1`, `T2`, `TREND`, `CLEAR`; recurrence-network measures (transitivity, clustering, average path length) | Planned with the line histograms and never written |
| Validation | DET and LAM against published reference values for the standard systems | D2 and lambda_1 of Lorenz are already checked |
| DD-54, DD-60 | Two decisions still `provisional` | The joint independence ratio as a coupling detector in time; RQA against the geometric index. Both measured on synthetic data only |

---

## 2. Written outside the library by the first real study

The study pipeline (about 15 000 lines of scripts around the library) wrote
the pieces below by hand. Each is general, was needed for a defensible result,
and was implemented at least twice in slightly different forms. They are
listed in the order in which they would pay off.

### 2.1 Level and dispersion of every windowed metric

**What.** For every subject, channel and metric of a windowed table, two
numbers: the *level* (median over the retained windows) and the *dispersion*
(standard deviation over the retained windows, requiring at least three).
The dispersion does not depend on the order of the windows.

**Why.** In the study, the dispersion family was the one that replicated
across recording blocks, while several level effects did not. Every script
recomputed it with its own suffix (`_bw_sd`) and its own minimum-window rule.

**Proposed API.** `recurra.windowed.aggregate(table, by=("subject", "channel"),
stats=("median", "sd"), min_windows=3)` returning a tidy frame with one column
per metric and statistic, plus a `n_windows` column. `classify.select_features`
learns a `kind="level" | "dispersion" | "both"` switch.

### 2.2 Features of the analytic signal per window

**What.** Per window: mean and SD of the instantaneous frequency; envelope
shape descriptors (q99/median, max/median, fraction above median + 6 MAD,
coefficient of variation, skewness) computed on the unscaled envelope.

**Why.** The instantaneous frequency of the delta band was the first
replicated result of the study; it was computed outside the library from the
`inst_freq` channel. `statespace.shape_descriptors` covers part of the
envelope shape per block [DD-102] but not per window and not on the raw scale.

**Proposed API.** A `family="analytic"` option in `windowed_recurrence(...).metrics()`
and `batch_windowed_recurrence(...).metrics()`, reading the `INST_FREQ` and
`AMPLITUDE` channels the recording already carries. Guard: the envelope shape
must be taken before any scaling policy, or the descriptors measure the
policy [DD-105].

### 2.3 Regions of channels, aggregated at the feature level

**What.** A mapping region -> channels; the regional value of a subject is
the median, over the region's channels, of its per-channel feature. Never
average the signals.

**Proposed API.** `recurra.regions.aggregate(table, regions, how="median")`
and a `"global"` pseudo-region (median over all channels).

### 2.4 Group inference by region and montage

**What.** Per channel, the Mann-Whitney AUC between groups after regressing a
covariate out (artefact load in the study); per region, the mean over its
channels of `AUC - 0.5`; significance by permuting subject labels with the
*same* shuffle for every channel, so the channel correlation is kept in the
null. Folded separation `A* = max(AUC, 1 - AUC)` for screening, with the
direction stored apart. One-sided tests in a stated direction for
confirmation. Benjamini-Hochberg q-values, descriptive.

**Why.** `comparability.group_comparison` and `rank_metrics` work per window
and per metric; the study needed per region and per montage, with a covariate
and with shared shuffles. This was written four times.

**Proposed API.** `recurra.inference` module: `auc_by_channel`, `region_test`,
`montage_test` (shared shuffles, covariate, `n_jobs`), `folded_auc`,
`bh_fdr`. Every p-value reproducible at any worker count [DD-81].

### 2.5 Discovery and replication screening

**What.** A fixed rule applied to a discovery set: a candidate passes through a
*strong-cell* door (A* >= 0.70 and p < 0.05 in some region) or a *diffuse*
door (A* >= 0.60 in every region, same direction, montage p < 0.05); each
candidate gets a pattern (diffuse / regional / mixed); candidates are grouped
into *islands* (band, family, level or dispersion, direction, pattern); the
representative of an island is its largest A*; representatives correlating
with |Spearman rho| > 0.7 across subjects are merged. Replication: the same
statistic in the other set, one-sided in the discovery direction, with a
stated threshold. Nothing is discarded by hand.

**Why.** It is what made the study defensible: of eleven candidates six
replicated, and the two strongest discovery cells were among those that did
not. A single-block analysis would have reported them.

**Proposed API.** `recurra.inference.screen(table, rule=Rule(...))` returning
the candidates with their door, pattern, island and representative flag, and
`recurra.inference.replicate(candidates, other_table, threshold=0.64)`.
Thresholds are parameters with these values as documented defaults.

### 2.6 Transfer across recordings without an identity leak

**What.** When the same subjects appear in two recordings, a model fitted on
all of them in recording A and applied to recording B can recognise each
subject through stable individual traits and return its label. With 17 to
312 features the study measured transfer AUCs of 0.8 for models whose
within-recording cross-validation was at chance. The fix: for each subject,
fit on the *other* subjects in A and apply to that subject in B.

**Proposed API.** `classify.transfer(table_a, table_b, labels, features)`,
returning the leave-one-subject-out transfer AUC, the per-subject scores and,
on request, a permutation p. `ClassificationReport` should state which of the
two numbers it holds. A test must pin the leak: a pure subject fingerprint
with no group effect has to give chance (the study's check: 0.36, where the
naive transfer gave about 1.0).

### 2.7 Metrics at a decision threshold

**What.** Accuracy, balanced accuracy, sensitivity, specificity, precision,
F1 and MCC at a probability threshold, cross-validated (mean and SD over
repeats of the out-of-fold predictions) and on transfer scores.

**Why.** With a 15 / 34 imbalance precision and F1 are bounded by prevalence;
the report should say so next to the numbers [DD-85].

**Proposed API.** `ClassificationReport.at_threshold(0.5)` and a
`threshold="youden" | "balanced" | float` option.

### 2.8 Temporal tools

- **Cluster-based permutation over continuous traces** (Maris and Oostenveld):
  instant-by-instant group contrast of a resampled trace, clusters of
  supra-threshold samples, max-cluster null. Needed to show that a difference
  is a trait rather than a state of the session.
- **Sliding-block dispersion**: the between-window SD over a moving block of k
  windows, per subject, so that the stability of a metric can be followed
  along the recording.
- **Window-scale sweep**: recompute the windowed pipeline at several window
  lengths and tabulate the separation per scale. The study found
  scale-invariant measures, measures with a characteristic scale that average
  out in long windows, and measures that only separate at one scale.

`comparability.group_timecourse` already gives per-window AUC curves with a
max-statistic [DD-114]; these three extend it.

### 2.9 Topographic display

A zone map (values per scalp region drawn on a head outline) was the most
used figure of the study. It is montage-specific, so it belongs in `viz` only
with a user-supplied layout: `viz.plot_zone_map(values, layout)` where the
layout maps zone names to channel positions.

---

## 3. Guards learned from real data

Each of these cost time in the study and should become a warning or a default.

| Guard | Where | Status |
|---|---|---|
| `RR` is not a feature when the threshold fixes the rate: `select_features("rqa")` should drop it under `target_rr` | `classify/features.py` (`FAMILIES["rqa"]` includes `RR`) | to do |
| `Lmax` and `DIV` saturate in short windows; flag saturation per column rather than per call | `rqa` | partly done [DD-98] |
| Book-keeping columns (`seconds` kept, `n_windows`, `qc_*`) must never enter a feature set; in the study the seconds of clean signal per window leaked in and correlated with the diagnosis | `classify/features.py` | to do |
| Inside a joint (phase, amplitude) space, measures of each component alone are univariate; label them as such so they are not read as coupling [DD-77] | `dynamics_measures`, family names | to do |
| `ingest(meta={"bands": ...})` for arrays decomposed elsewhere [DD-112] | done | done |

---

## 4. Housekeeping before a public repository

Done for 0.25.0:

- **The nine failing slow tests.** Seven validation tests now pin the line
  floors and the Theiler window their findings were measured under, in one
  place at the top of `tests/test_validation.py` [DD-98]. Fine-scale
  decimation keeps at least 64 samples [DD-99]. The Lyapunov tests are
  contracts over several trajectories rather than facts about one [DD-101].
- **Lint.** `ruff check src tests examples tools` is clean with the ruff
  version pinned in the `dev` extra.
- **Licence** (MIT), **citation** (`CITATION.cff`), **logo**.
- **Contributing.** [`CONTRIBUTING.md`](../CONTRIBUTING.md) restates the rules
  the project follows.
- **Repository URL.** `github.com/IgnacioRodro/recurra` is the final name.
- **Counts that rot.** Test counts are stated only where a test checks them:
  `ARCHITECTURE.md` and the status line of `README.md`.

Still open:

- **PyPI.** The wheel builds and installs in a clean environment, and CI
  checks that on every push. Before publishing, the images in `README.md`
  need absolute URLs (PyPI does not resolve relative paths), then TestPyPI
  before PyPI. Planned after the GPU backend.

---

## 5. Release checklist

Before tagging `vX.Y.Z`:

1. **Version.** `src/recurra/_version.py` says `X.Y.Z`; the newest heading of
   `CHANGELOG.md` is `## [X.Y.Z] — YYYY-MM-DD`; `CITATION.cff` has the same
   `version` and `date-released`. `tests/test_docs.py` checks all three.
2. **Fast suite, lint, documents.** Every CI job green on the last push: lint,
   the fast suite on Linux, macOS and Windows for every supported Python,
   core-only, minimum-versions, coverage and build.
3. **Slow suite.** Run *Slow validation* by hand (Actions → Slow validation →
   Run workflow) and wait for it to pass. Not optional: nine slow tests broke
   unnoticed between 0.22 and 0.24 because they were never re-run after a
   default changed.
4. **Examples, all of them.** `python examples/run_all.py`, without `--skip`.
   The cookbook is the slowest and the one that finds bugs: an odd-sized
   matrix (1157 points) exposed a figure failure that every test, built on
   round sizes, had missed.
5. **Test index.** `python tools/generate_test_index.py`, and commit
   `docs/TESTS.md` if it changed.
6. **Tag.** GitHub → Releases → Draft a new release, tag `vX.Y.Z`, the
   `CHANGELOG.md` entry as the description.
7. **Move `main` on.** Straight after the tag, set the version to the next
   `.dev0` with a new `unreleased` heading, so that nothing installed from
   `main` can be mistaken for the release.
