<picture><source media="(prefers-color-scheme: dark)" srcset="_static/recurra-mark-dark.svg"><img alt="recurra" src="_static/recurra-mark.svg" width="56" align="right"></picture>

# Design decisions — `recurra`

**Living document.** Every decision with consequences is recorded here with its reasoning, the alternatives rejected, and why. ADR format (Architecture Decision Record).

Status: `accepted` · `provisional` · `revised` · `rejected`

| ID | Decision | Status | Phase |
|---|---|---|---|
| [DD-01](#dd-01) | Package name: `recurra` | accepted | F0 |
| [DD-02](#dd-02) | **Scaling policy for heterogeneous blocks** | accepted | F0 |
| [DD-03](#dd-03) | Capability system | accepted | F0 |
| [DD-04](#dd-04) | Mandatory provenance | accepted | F0 |
| [DD-05](#dd-05) | Explicit RNG, never global | accepted | F0 |
| [DD-06](#dd-06) | Validation reports, it does not silence | accepted | F0 |
| [DD-07](#dd-07) | Immutable `Recording` with copy-on-write | accepted | F1 |
| [DD-08](#dd-08) | One entry point: `ingest()` | accepted | F1 |
| [DD-09](#dd-09) | Actionable exception hierarchy | accepted | F0 |
| [DD-10](#dd-10) | CSV export with sidecars | accepted | F1 |
| [DD-11](#dd-11) | Filter order from a minimum cycle count | accepted | F2 |
| [DD-12](#dd-12) | Hilbert edge policy | accepted | F2 |
| [DD-13](#dd-13) | Alpha preserves the marginal spectrum | accepted | F8 |
| [DD-14](#dd-14) | Channels as a mapping, not a matrix | accepted | F1 |
| [DD-15](#dd-15) | Metric output: tidy DataFrame | accepted | F0 |
| [DD-16](#dd-16) | GPU backend: PyTorch | provisional | F5 |
| [DD-17](#dd-17) | Numba optional, numpy always present | accepted | F0 |
| [DD-18](#dd-18) | Unequal record lengths allowed | accepted | F1 |
| [DD-19](#dd-19) | Corrected Theiler semantics | revised | F5 |
| [DD-20](#dd-20) | **JRP with independent thresholds** | accepted | F5 |
| [DD-21](#dd-21) | **Weights fold into the coordinates** | accepted | F3 |
| [DD-22](#dd-22) | Autocorrelation by FFT | accepted | F3 |
| [DD-23](#dd-23) | FNN criterion A must be normalised | accepted | F3 |
| [DD-24](#dd-24) | Aggregate across records, never concatenate | accepted | F3 |
| [DD-25](#dd-25) | Three routes, one path downstream | accepted | F3 |
| [DD-26](#dd-26) | Route comparison is a first-class operation | accepted | F3 |
| [DD-27](#dd-27) | Plots show the weighted coordinates | accepted | F3 |
| [DD-28](#dd-28) | **Balancing variance is not enough** | accepted | F3 |
| [DD-29](#dd-29) | Envelope smoothing is explicit | accepted | F3 |
| [DD-30](#dd-30) | **The whole repository is in English** | accepted | F3 |
| [DD-31](#dd-31) | **Single figures vs comparison mode** | accepted | F3 |
| [DD-32](#dd-32) | Sample pairs from the region RR is defined over | accepted | F4 |
| [DD-33](#dd-33) | Squared-norm expansion, not broadcasting | accepted | F5 |
| [DD-34](#dd-34) | The library refuses rather than thrashes | accepted | F5 |
| [DD-35](#dd-35) | **The default is never a fixed epsilon** | accepted | F4 |
| [DD-36](#dd-36) | **The matrix is an implementation detail** | accepted | F5 |
| [DD-37](#dd-37) | A run should explain itself | accepted | F5 |
| [DD-38](#dd-38) | Pooling for display, watching for saturation | accepted | F5 |
| [DD-39](#dd-39) | One window spec, three ways to state it | accepted | F10 |
| [DD-40](#dd-40) | **Threshold scope decides what windows can say** | accepted | F10 |
| [DD-41](#dd-41) | Windows are lazy | accepted | F10 |
| [DD-42](#dd-42) | **Comparability is a contract, not an accident** | accepted | F10 |
| [DD-43](#dd-43) | **What matched decides what may be compared** | accepted | F10 |
| [DD-44](#dd-44) | Groups are compared like with like | accepted | F10 |
| [DD-45](#dd-45) | **The documents are kept in step by tests** | accepted | F10 |
| [DD-46](#dd-46) | Warnings carry their own category | accepted | F10 |
| [DD-47](#dd-47) | **Decimation changes the sampling rate** | accepted | F10 |
| [DD-48](#dd-48) | **Phase locking is a bounded deviation, not a mixture** | accepted | F8 |
| [DD-49](#dd-49) | Shared axes only between comparable spaces | accepted | F10 |
| [DD-50](#dd-50) | Robust axis limits for heavy-tailed coordinates | accepted | F10 |
| [DD-51](#dd-51) | Circle blocks show the phase, not its cosine | accepted | F10 |
| [DD-52](#dd-52) | A clean check reports what it checked | accepted | F10 |
| [DD-53](#dd-53) | **Route comparison: what may be compared, and which route to pick** | accepted | F10 |
| [DD-54](#dd-54) | **The joint independence ratio detects coupling in time** | provisional | F10 |
| [DD-55](#dd-55) | Four statuses cannot live in colour alone | accepted | F10 |
| [DD-56](#dd-56) | **Line histograms carry runs across tiles** | accepted | F6 |
| [DD-57](#dd-57) | White lines and the Theiler band | accepted | F6 |
| [DD-58](#dd-58) | The DET denominator is a choice, and it is stated | accepted | F6 |
| [DD-59](#dd-59) | Entropy needs its normalisation stated | accepted | F6 |
| [DD-60](#dd-60) | **RQA does not beat the geometric index yet** | provisional | F6 |
| [DD-61](#dd-61) | **The scaling region is located, not guessed** | accepted | F9 |
| [DD-62](#dd-62) | Rates carry their units | accepted | F9 |
| [DD-63](#dd-63) | **Noiseless data breaks the Lyapunov estimator** | accepted | F9 |
| [DD-64](#dd-64) | The harmonic artefact did not exist | accepted | F8 |
| [DD-65](#dd-65) | **Recurrence measures carry artefact information** | accepted | validation |
| [DD-66](#dd-66) | **The classical indices win on detection** | accepted | validation |
| [DD-67](#dd-67) | **Only three metrics are stable against window length** | accepted | validation |
| [DD-68](#dd-68) | Window length with a fixed sampling rate | accepted | validation |
| [DD-69](#dd-69) | **Many short windows beat few long ones** | accepted | validation |
| [DD-70](#dd-70) | **The other three modalities, and where recurrence wins** | accepted | validation |
| [DD-71](#dd-71) | **Each block contributes its natural scalar** | accepted | F9b |
| [DD-72](#dd-72) | Cheap by default, invariants on request | accepted | F9b |
| [DD-73](#dd-73) | The Higuchi fit is anchored at k = 1 | accepted | F9b |
| [DD-74](#dd-74) | **Dynamical measures on the harmonic artefact** | accepted | validation |
| [DD-75](#dd-75) | **Detection with both families, and why combining hurts** | accepted | validation |
| [DD-76](#dd-76) | **SVD entropy is the short-window measure** | accepted | validation |
| [DD-77](#dd-77) | **A univariate measure cannot be a coupling measure** | accepted | validation |
| [DD-78](#dd-78) | Most tiles never touch the Theiler band | accepted | F7 |
| [DD-79](#dd-79) | Single precision is a choice, not a default | revised | F7 |
| [DD-80](#dd-80) | **Decimation changes the metrics, not just the cost** | accepted | F7 |
| [DD-81](#dd-81) | **Parallelism must not change the answer** | accepted | F7 |
| [DD-82](#dd-82) | Parallelise across units, not inside one | accepted | F7 |
| [DD-83](#dd-83) | **Windows of one subject are not independent samples** | accepted | F13 |
| [DD-84](#dd-84) | Everything that learns goes inside the fold | accepted | F13 |
| [DD-85](#dd-85) | Imbalanced-aware scores, with a baseline | accepted | F13 |
| [DD-86](#dd-86) | One split is an anecdote | accepted | F13 |
| [DD-87](#dd-87) | **Significance from permutation, at the subject level** | accepted | F13 |
| [DD-88](#dd-88) | **A recording carries more than its samples** | accepted | F1b |
| [DD-89](#dd-89) | **A corpus is files plus a table, and the join is checked** | accepted | F1b |
| [DD-90](#dd-90) | **Resizing a recurrence plot is not resampling a photo** | accepted | F14 |
| [DD-91](#dd-91) | **The brightness scale belongs to the batch** | accepted | F14 |
| [DD-92](#dd-92) | Past some reduction, decimate rather than pool | accepted | F14 |
| [DD-93](#dd-93) | Invariants per block as well as per trajectory | accepted | F9c |
| [DD-94](#dd-94) | **The Lyapunov fit window is the estimate** | accepted | F9c |
| [DD-95](#dd-95) | Two estimators, two option sets | accepted | F9c |
| [DD-96](#dd-96) | **A delay embedding needs a large Theiler window** | accepted | F9c |
| [DD-97](#dd-97) | **Embed the oscillation, not its description** | accepted | F9c |
| [DD-98](#dd-98) | **DET and LAM need different floors** | accepted | F6b |
| [DD-99](#dd-99) | **A fine-scale measure needs its own scale** | accepted | F9d |
| [DD-100](#dd-100) | **A dimension with no scaling region measures the construction** | accepted | F9d |
| [DD-101](#dd-101) | **The exponent is averaged over samplings, not drawn once** | accepted | F9d |
| [DD-102](#dd-102) | **The tail was a warning and not a column** | accepted | F3b |
| [DD-103](#dd-103) | An attractor image counts rather than marks | accepted | F14b |
| [DD-104](#dd-104) | **Figures in separate files still have to share their axes** | accepted | F14b |
| [DD-105](#dd-105) | **The absolute scale of a weighted space is not information** | accepted | F14b |
| [DD-106](#dd-106) | **The reference index belongs in the same table** | accepted | F2b |
| [DD-107](#dd-107) | A circular-shift surrogate saturates its own z-score | accepted | F2b |
| [DD-108](#dd-108) | **The line histogram is kept, so exploring floors is free** | accepted | F10b |
| [DD-109](#dd-109) | **The metrics are the work, so they parallelise too** | accepted | F10b |
| [DD-110](#dd-110) | **Centre the coordinates before expanding the norm** | accepted | F15 |
| [DD-111](#dd-111) | Normalised entropies count admissible lengths, not lengths present | accepted | F15 |
| [DD-112](#dd-112) | **The envelope low-pass has to pass the phase band** | accepted | F15 |
| [DD-113](#dd-113) | **The amplitude band has to hold the sidebands** | accepted | F15 |
| [DD-114](#dd-114) | **The unit of inference is the subject; the peak of a curve is a max-statistic** | accepted | F15 |
| [DD-115](#dd-115) | **The Theiler window follows the standard convention** | accepted | F16 |
| [DD-116](#dd-116) | Canonical coupling indices need an envelope | accepted | F16 |
| [DD-117](#dd-117) | **Envelope smoothing preserves the sign** | accepted | F16 |
| [DD-118](#dd-118) | No counter-intuitive calls: one name per idea, defaults that agree | accepted | F16 |
| [DD-119](#dd-119) | **Meta-recurrence of the window measurements** | accepted | F16 |
| [DD-120](#dd-120) | A recurrence plot is drawn with its signals | accepted | F16 |
| [DD-121](#dd-121) | **D2 is a small-radius limit: the search stays below the saturating scale** | accepted | F16 |

---

<a id="dd-01"></a>
## DD-01 · Package name: `recurra`

**Status:** accepted

**Context.** The draft was called `multiv_RQA`. It mixes capitals and an underscore, against PEP 8 and against PyPI convention, and it describes the scope badly: the project is about recurrence over arbitrary state spaces, not only about multivariate RQA.

**Decision.** `recurra`. Short, lowercase, pronounceable, evokes recurrence without locking the package into either RQA or CFC.

---

<a id="dd-02"></a>
## DD-02 · Scaling policy for heterogeneous blocks

**Status:** accepted · **The central design decision**

### The problem

In a state space such as the one the project proposes for PAC,

$$s(t) = (\cos\phi_{low},\ \sin\phi_{low},\ A_{high})$$

with the Euclidean metric:

$$d^2(s_i,s_j) = \underbrace{4\sin^2(\Delta\phi/2)}_{\text{phase block}} + \underbrace{(\Delta A)^2}_{\text{amplitude block}}$$

The phase block has a **geometrically fixed** extent: by construction it lives on the unit circle and the chord distance is bounded in [0, 2], whatever happens. The amplitude block does not: it depends on microvolts, electrode impedance, subject, amplifier gain.

Two failure modes:

- **σ_A ≫ 1** → the recurrence plot is the amplitude's. The toroidal geometry that motivates Hypothesis 1 disappears.
- **σ_A ≪ 1** → the recurrence plot is the phase's. This is the dangerous one: a slow oscillator is deterministic by itself, so **DET comes out high whether or not there is any coupling**. A false positive that looks like a result.

Verified numerically: at σ_A = 0.01 the phase block takes 99.99% of the distance variance; at σ_A = 100, the amplitude takes 99.99%.

### The default

Equalise the mean-square contribution of each block. Two analytic results:

- Scalar with standard deviation σ: `E[(A_i − A_j)²] = 2σ²` ⟹ RMS pairwise = σ√2
- Unit circle, uniform phase: `E[‖s_i − s_j‖²] = E[2 − 2cos Δφ] = 2` ⟹ RMS pairwise = √2

So **dividing the amplitude by its standard deviation matches the two contributions exactly**. Not an arbitrary convention: it is the solution of σ_A → √2, which is the circle's own value.

Verified: `theoretical_rms_pairwise(circle) = 1.4142`, and with `policy="rms_balanced"` the split is exactly 0.5/0.5 for σ_A between 0.001 and 1000.

**Decisive side benefit:** the phase block was already invariant to signal gain (it lives on the unit circle by construction). Z-scoring the amplitude makes **the whole joint space invariant to a global gain change**, a necessary condition for comparing across subjects and across electrodes with different impedances.

### Policies implemented

| `policy` | What it does | When |
|---|---|---|
| `rms_balanced` *(default)* | equalises RMS pairwise distance per block, closed form | general use |
| `empirical_balanced` | the same, using the sampled RMS | strongly non-uniform phase |
| `weighted` + `lambda_` | `d² = (1−λ)d²_A + λd²_B` | sensitivity sweep |
| `per_block_threshold` | Chebyshev per block with its own epsilon | scale-free |
| `rank` / `copula` | map to the empirical CDF | monotone invariance |
| `custom` | user weights | replicating other work |
| `none` | raw | diagnosing the problem |

### Lambda as an observable, not a nuisance

With `policy="weighted"`, λ ∈ [0,1] sweeps from a pure phase plot (λ=0) to a pure amplitude one (λ=1). Verified that the variance split equals λ exactly, **independently of σ_A**.

The testable hypothesis this opens is stated, and corrected, below.

### The hypothesis as first written, and why it was wrong

The original text of this decision read:

> under genuine coupling, DET should show an **interior maximum** in λ — the
> joint space captures structure that neither projection alone contains.
> Without coupling it should interpolate monotonically.

Half of that survives. The endpoints do carry no information about coupling:
at λ=0 the space is the phase circle alone and at λ=1 the envelope alone, both
one-dimensional objects whatever the coupling. A projection onto a single
block cannot know anything about the relation between blocks, and the measured
curves confirm it — coupled and uncoupled signals coincide at both ends.

**The sign was wrong.** Measured on the correlation-sum slope, which estimates
the effective dimension of the cloud:

| λ | 0.0 | 0.3 | 0.5 | 0.7 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| uncoupled | 1.01 | 1.90 | 1.97 | 2.05 | 2.16 | 0.98 |
| PAC α=0.9 | 1.02 | 1.48 | 1.52 | 1.63 | 1.76 | 0.94 |

The **uncoupled** signal has the larger interior value, not the coupled one.

The reasoning behind the original guess was simply back to front. When phase
and amplitude are independent, the joint space is their cartesian product: the
object fills both directions and the effective dimension approaches the sum of
the parts, near 2. Coupling is a *constraint* — in the deterministic limit the
amplitude is a function of the phase and the joint space collapses onto a
one-dimensional curve lifted over the circle. A constraint removes dimension;
it does not add structure in the sense of adding degrees of freedom.

So the discriminating quantity is a **dimension deficit relative to the
independent case**, not an interior excess.

### The corrected hypothesis, and what supports it

*Under coupling, the effective dimension of the joint space falls below the
sum of its blocks' dimensions, by an amount that grows with coupling
strength.*

Same seed, same spectrum, only α varied; the figure is the mean
correlation-sum slope over λ ∈ {0.3, 0.5, 0.7, 0.9}:

| α | 0.00 | 0.20 | 0.40 | 0.60 | 0.80 | 0.95 |
|---|---|---|---|---|---|---|
| interior slope | 2.020 | 2.021 | 2.009 | 1.963 | 1.812 | 1.452 |

Spearman(α, slope) = **−0.943**. Across three independent seeds the gap
between α=0 and α=0.95 is 0.567, 0.554 and 0.578 — consistent to within 4%.

### The correction was itself over-generalised

With the RQA layer in place the hypothesis can be tested with DET, which is
what it was originally phrased about. Same recording, same sweep:

| λ | 0.00 | 0.30 | 0.50 | 0.70 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|
| DET, uncoupled | 0.917 | 0.850 | 0.784 | 0.702 | 0.560 | 0.215 |
| DET, coupled | 0.917 | **0.921** | **0.888** | **0.831** | **0.702** | 0.196 |

The endpoints coincide, as they must. In the interior the **coupled** signal
has the higher DET — which is what the original prediction said, and the
opposite of what the correlation-sum slope does.

Both results are right, and the mistake was mine twice over: first predicting
without measuring, then correcting the prediction in terms of a different
descriptor as though it were the same claim. It is not.

**Coupling is a constraint, and a constraint does two things at once.** It
removes degrees of freedom, so the effective dimension falls — which is what
`corr_slope` measures, and why the deficit is negative. And it makes the
trajectory more predictable, so more recurrent points lie on diagonal lines —
which is what DET measures, and why its interior excess is positive.

The corrected statement, in full:

*In the interior of the sweep, coupling lowers the effective dimension of the
joint space and raises its determinism. Both effects vanish at the endpoints,
where a single-block projection can say nothing about the relation between
blocks.*

### What this does not yet establish

Two limitations that must be settled before the deficit is proposed as a
coupling index:

- **It is insensitive to weak coupling.** Between α=0 and α=0.4 the slope
  barely moves, from 2.020 to 2.009. The effect only takes off above α≈0.6,
  precisely where MVL and the modulation index are already saturating. A
  detection curve against the canonical indices is needed before claiming
  anything about sensitivity.
- **It is uncontrolled for noise and record length.** All of the above is at
  20 dB over 40 seconds. The correlation-sum slope is notoriously sensitive to
  both, and the deficit has not been shown to survive realistic noise or
  shorter records.

The measurement is a candidate for a purely geometric coupling index, obtained
before any RQA metric exists. It is not one yet.

### A note on how this was found

Not by a test, and not by reasoning. The hypothesis sat in this document for
several phases, stated with the wrong sign, and was contradicted the first
time somebody looked at the comparison figure it predicted. The lesson is
narrow and worth keeping: a prediction written into a design document is not
evidence, and phrasing it confidently does not make it more likely to be
right.

### Mandatory diagnostic

`ScaleReport` returns, per block, the fraction of distance variance it contributes, and warns when a block exceeds 90%. That number belongs in the tables of any paper using this.

### Rejected alternatives

- **Global Mahalanobis whitening.** It would mix cos φ with sin φ and destroy the circular geometry the whole framework rests on. Rejected.
- **Do nothing and document it.** It hands the user a decision almost nobody will take consciously, and the silent failure mode (high DET with no coupling) is exactly the one that produces wrong papers.
- **Normalise everything to [0,1] (minmax).** Sensitive to outliers and not gain-invariant. Available as an option, not as the default.

---

<a id="dd-03"></a>
## DD-03 · Capability system

**Status:** accepted

**Context.** Data can enter at any point of the chain: raw signal, filtered bands, already-computed Hilbert components, or a point cloud in state space.

**Decision.** A `Flag` (`Capability`) recording what information a `Recording` holds. Each stage declares `requires` and `provides`; `require()` validates before computing.

**Consequences.**
1. Entering with preprocessed data does not force pretending to have the raw signal.
2. Stages already covered are skipped rather than recomputed.
3. A composition error fails when the plan is built, naming the missing capability, instead of blowing up three stages later with a shape error.

---

<a id="dd-04"></a>
## DD-04 · Mandatory provenance

**Status:** accepted

Principle P3 (no silent simplification) is only enforceable if every operation leaves a trace that travels with the data. Each `Recording` carries a tuple of `Step` with name, parameters, inputs, outputs, timestamp and **caveats**: what the stage did without being asked, and the user needs to know.

A real example from the preprocessing chain: `analytic` records *"375 samples trimmed from each edge (1.500 s total)"*. Without it, the user believes they have 20 s of signal when they have 18.5.

---

<a id="dd-05"></a>
## DD-05 · Explicit RNG, never global

**Status:** accepted

**Context.** The draft called `np.random.seed()` inside library functions. That mutates global interpreter state and silently correlates draws between callers — in `synthetic.py` it was in fact called twice with the same seed, producing correlated components by accident.

**Decision.** Every stochastic operation takes an explicit `rng`. `resolve_rng()` is the single funnel. For parallelism, `spawn_rngs()` uses `SeedSequence.spawn`, which guarantees statistical independence between streams.

**Critical consequence:** results **do not depend on `n_jobs`**. Verified in tests.

---

<a id="dd-06"></a>
## DD-06 · Validation reports, it does not silence

**Status:** accepted

A constant channel, a saturated ADC or a signal too short produce plausible-looking numbers downstream. `check_channels` detects them, grades them by severity and attaches them to the `Recording` as an exportable `QualityReport`. `strict=True` turns errors into exceptions; by default they report.

---

<a id="dd-07"></a>
## DD-07 · Immutable `Recording` with copy-on-write

**Status:** accepted

**The dilemma.** Strict immutability gives predictable semantics and exact provenance, but duplicating an hour of EEG is unaffordable. Mutability is cheap but lets one step corrupt data another process is using.

**Decision.** Immutable semantics at near-zero cost. Each operation returns a new `Recording` that **shares by reference** the arrays that did not change; memory is allocated only for what is new. Arrays are marked `writeable=False` so a shared buffer cannot be mutated behind another object's back.

Verified in tests: after adding a channel, `out.get(x) is rec.get(x)` is `True`, and writing to the array raises `ValueError`.

---

<a id="dd-08"></a>
## DD-08 · One entry point: `ingest()`

**Status:** accepted

Eleven input shapes, one function. The alternative — a constructor per case — multiplies the API surface and fragments validation.

The `roles` argument removes the otherwise fatal ambiguity: a `(T, 3)` array can be three channels, a `(cos, sin, A)` state space, or three bands. Only the caller knows. When it is not declared, names are matched against patterns (`phase_*`, `amp_*`) and **every inference is recorded as a caveat and warned about**.

---

<a id="dd-09"></a>
## DD-09 · Actionable exception hierarchy

**Status:** accepted

Every failure a user can fix has its own type and carries enough context to fix it. Bare `ValueError` is reserved for programming errors.

Example: passing a transposed matrix produces *"input shape (3, 800) has more columns than rows. recurra expects (n_samples, n_channels); transpose it if that is wrong"* instead of a broadcasting error two hundred lines later.

---

<a id="dd-10"></a>
## DD-10 · CSV export with sidecars

**Status:** accepted

Project requirement: intermediate results must reach CSV so they can go into a paper. Two rules:

1. Every exported table carries `*_provenance.csv` and `*_environment.csv`. A published number must be reconstructible from the code version, the parameters and the digest of the input that produced it.
2. **Tidy** format (long, one observation per row). Wide tables are pleasant to look at and painful to aggregate over subjects and windows.

---

<a id="dd-11"></a>
## DD-11 · Filter order from a minimum cycle count

**Status:** accepted

CFC is notoriously sensitive to filtering: too narrow a band manufactures a sinusoid where none exists, and a filter with too few cycles smears the phase. Aru et al. (2015) name filtering as a main source of spurious PAC.

`order="auto"` sets the FIR length from the **low** edge of the band, guaranteeing a minimum number of cycles in the impulse response (3 by default, the standard EEG recommendation). Odd length is forced for exact linear phase. Zero-phase filtering by default: a phase-distorting filter invalidates any phase-based coupling measure.

---

<a id="dd-12"></a>
## DD-12 · Hilbert edge policy

**Status:** accepted

The Hilbert transform is non-local: the analytic signal near the edges of a finite record is contaminated. Combined with the filter transient this is a classic source of spurious coupling at the start and end of every epoch.

`edge_policy ∈ {trim, mirror, taper, none}`, default `trim`, **reusing the transient length recorded by the preceding `bandpass`** — so the two edge effects are handled together instead of being counted twice or forgotten.

---

<a id="dd-13"></a>
## DD-13 · Alpha preserves the marginal spectrum

**Status:** accepted

In the generator, the modulation is normalised so that the mean envelope power **does not change with alpha**. Otherwise a coupling index would rise with alpha simply because the spectrum changed — precisely the pseudo-coupling artefact the generator should let us *study*, not produce by accident.

Validated: Spearman(α, MVL) = 1.00 and Spearman(α, MI) = 1.00, with α=0 giving MVL ≈ 0.0025.

---

<a id="dd-14"></a>
## DD-14 · Channels as a mapping, not a matrix

**Status:** accepted

A `Recording` routinely holds heterogeneous components: `signal`, `theta`, `phase_theta`, `amp_gamma`, each with its own role and units. A `(T, D)` matrix would force them into one anonymous axis and lose exactly the information the state-space builders need.

`as_array(names)` returns the matrix when one is wanted.

---

<a id="dd-15"></a>
## DD-15 · Metric output: tidy DataFrame

**Status:** accepted

Decided with the user. `pandas.DataFrame` in long format. `xarray` would be cleaner for `subject × window × component × metric`, but adds a core dependency, and pandas is already needed for CSV export.

---

<a id="dd-16"></a>
## DD-16 · GPU backend: PyTorch

**Status:** provisional (to confirm when the GPU backend lands)

Against CuPy: PyTorch will be needed anyway for the CNN work, so reusing it avoids a dependency. The available server has 4 GPUs with 11 GB, enough for 8192×8192 float32 blocks (268 MB).

The tile abstraction of [DD-36] is what makes this a backend swap rather than a rewrite: only `sq_distance_block` needs a device-aware version.

**Note (0.25).** Not implemented. The `gpu` extra installed PyTorch for a backend that does not exist, so it is withdrawn until the backend is written; `available_backends()` still reports whether torch is present.

---

<a id="dd-17"></a>
## DD-17 · Numba optional, numpy always present

**Status:** accepted

Numba speeds up line-length histograms considerably, but complicates installation on older HPC systems. It goes in the `[fast]` extra, with a numpy reference implementation **always present and always tested**. The backend is chosen at runtime.

**Note (0.25).** The numpy path is the only one written: no module imports numba. The `fast` extra installed it for nothing and is withdrawn until a numba kernel exists and is tested against the numpy reference.

---

<a id="dd-18"></a>
## DD-18 · Unequal record lengths allowed

**Status:** accepted

The draft required identical T and D across subjects. That was implementation convenience, not a requirement of the method, and real public datasets do not satisfy it. It is allowed from the start; homogeneity, where genuinely needed, is required at the specific point that needs it — for instance a joint recurrence plot, which does need a common time base and says so explicitly.

---

<a id="dd-19"></a>
## DD-19 · Corrected Theiler semantics

**Status:** revised by [DD-115](#dd-115) in 0.25.0. What follows is the
original record; the text below said `theiler=1` excludes the line of
identity, while the code excluded `|i - j| <= theiler`, which at
`theiler=1` is three diagonals. DD-115 makes the code say what this entry
said.

In the draft the condition was `abs(k) <= theiler`, so `theiler=0` already excluded the main diagonal. Convention adopted: `theiler=0` ⟹ **no exclusion**; `theiler=1` ⟹ excludes the line of identity.

The recurrence rate denominator excludes the same band, so the reported rate is directly comparable with the threshold's own estimate. Tested exactly: for `theiler` in {1, 5, 20}, every diagonal inside the band is empty and the first diagonal outside it is not.

---

<a id="dd-20"></a>
## DD-20 · JRP with independent thresholds

**Status:** accepted (implemented in F5)

Romano et al. (2004) define the joint recurrence plot as the elementwise product of recurrence plots computed in **independent** phase spaces, each with its own threshold fixed to its own recurrence rate.

The draft forced a single shared epsilon and required both subsystems to have equal dimensionality. Both were wrong:

- A shared epsilon makes the joint rate depend on the arbitrary relative scaling of two different systems.
- The equal-dimension requirement belongs to the **cross** recurrence plot, where the two trajectories must live in one common space to be compared at all.

This matters more than usual here, because Hypothesis 2 of the project rests on the joint plot working **between attractors of different dimension**.

**Measured on synthetic data.** With each subsystem held at a 10% rate, the ratio of the joint rate to the product of the individual rates — 1.0 when the subsystems recur independently — comes out:

| Signal | Joint rate | Expected if independent | Ratio |
|---|---|---|---|
| PAC α=0.9 | 1.600% | 0.997% | **1.604** |
| PAC α=0.3 | 1.014% | 1.005% | 1.008 |
| uncoupled | 1.001% | 1.003% | 0.998 |

The subsystems were a 2-D phase circle and a 1-D envelope. Under the draft's shared-epsilon rule this comparison could not have been made at all.

`independence_ratio()` is exposed on the joint matrix as a first-class quantity.

---

<a id="dd-21"></a>
## DD-21 · Weights fold into the coordinates

**Status:** accepted

The combined distance across heterogeneous blocks is

$$d^2(x,y) = \sum_g w_g \lVert x_g - y_g \rVert^2$$

Rather than teach every downstream engine about blocks and weights, note that this is exactly the plain Euclidean distance on rescaled coordinates:

$$d^2 = \lVert \sqrt{w_g}\,(x_g - y_g) \rVert^2$$

So `StateSpace.weighted_coords` returns the array on which ordinary Euclidean distance **already is** the grouped weighted distance. The whole scaling apparatus [DD-02] collapses into one array transform, and the recurrence engine stays ignorant of phases, amplitudes and blocks.

Verified in tests: block-wise distance matches Euclidean distance on `weighted_coords` to relative tolerance 1e-10.

**Limit:** the identity holds for Euclidean block metrics. Non-Euclidean block metrics need the explicit grouped path, which the recurrence layer provides for the `per_block` threshold mode.

---

<a id="dd-22"></a>
## DD-22 · Autocorrelation by FFT

**Status:** accepted

The draft recomputed the correlation from scratch for every candidate lag: O(τ_max · N · D). Here the autocorrelation comes from one FFT, O(N log N), which makes τ_max = 2000 as cheap as τ_max = 20.

---

<a id="dd-23"></a>
## DD-23 · FNN criterion A must be normalised

**Status:** accepted

The standard false-nearest-neighbour test has two criteria: the relative jump `R = |d_{m+1} − d_m| / d_m` against `Rtol`, and the absolute one `d_{m+1} / R_A` against `Atol`, where `R_A` is the size of the attractor.

The draft compared `d_{m+1}` against `Atol` directly. Since distances grow like √(m·D), that criterion becomes progressively harder to satisfy as m rises, the FNN fraction never falls below threshold, and **m saturates at `m_max`**. Normalising by `R_A` fixes it.

**Validated:** with AMI + normalised FNN, Lorenz-x gives **m = 3**, its true dimension. Neighbour search by KD-tree rather than the draft's brute-force Python loop.

---

<a id="dd-24"></a>
## DD-24 · Aggregate across records, never concatenate

**Status:** accepted

The draft stacked subjects along the time axis before computing ACF/AMI, which inserts an artificial discontinuity at every join. `estimate_embedding_multi` estimates per subject and aggregates (median by default, or max/mean/mode), also reporting the spread across records.

---

<a id="dd-25"></a>
## DD-25 · Three routes, one path downstream

**Status:** accepted

The project states the two ways of building the phase space: observable state variables, or Takens reconstruction. `recurra` supports both plus their combination, with identical code downstream.

- **`observable`** — coordinates are physical quantities. This is where the CFC state spaces live.
- **`delay`** — Takens reconstruction from one or several channels.
- **`hybrid`** — observable coordinates *plus* delays applied to a subset. The canonical case: keep the phase circle intact (its geometry is already right) but embed the envelope in m dimensions, because the envelope is a scalar observable of a subsystem with its own dynamics.

Nothing forces the coupling reading: `channels()` builds an ordinary multivariate state space and `custom()` takes any callable.

---

<a id="dd-26"></a>
## DD-26 · Route comparison is a first-class operation

**Status:** accepted

The project asks for the choice of state space to be justified. The only honest justification is empirical, so `compare_routes()` is a library function, not something the user must improvise.

Until the RQA layer exists, the geometric descriptors are threshold-free and cheap: mean nearest-neighbour distance, its coefficient of variation, the local slope of the correlation sum, and `neighbour_agreement` (fraction of k-neighbours shared between two spaces). They already discriminate the routes, and DET and LAM will join them — not replace them — in F6.

---

<a id="dd-27"></a>
## DD-27 · Plots show the weighted coordinates

**Status:** accepted

The geometry that matters is the one the recurrence engine will see, and that is the weighted one [DD-21]. Drawing the raw coordinates would show a picture the analysis never uses: typically one where the amplitude axis flattens the phase circle, exactly the artefact the scaling policy removes. `weighted=False` is available for diagnosing the problem.

Modes: `2d`, `3d`, `pairs`, `time`, `torus`.

**`torus`** is specific to the CFC framework: it maps angle = φ and radius = 1 + normalised A. It is the natural view of Hypothesis 1 and makes visible at a glance the difference between no coupling (uniform ring), distributed coupling (smoothly thickened ring) and coupling at a preferred phase (localised lobe).

---

<a id="dd-28"></a>
## DD-28 · Balancing variance is not enough

**Status:** accepted · **Found by running F3, not anticipated in the design**

Equalising RMS pairwise distance [DD-02] balances **second moments**. It does not balance distribution **shape**.

Measured on real pipeline data: a gamma envelope has skewness ≈ 1.7 and tail/RMS ratio ≈ 3.9, while the phase circle is bounded with ratio 1.41. After balancing, the rare high-amplitude excursions still sit far from everything else, and under a fixed threshold they become non-recurrent — producing empty bands in the recurrence plot that reflect **the tail of the amplitude distribution, not the coupling**.

The library therefore reports `tail_ratio` and `skewness` per block alongside the variance share, and warns when they differ sharply.

**Definition of `tail_ratio`:** the 99.9th percentile of **pairwise** distance divided by the RMS pairwise distance. Pairwise, not distance-to-centroid: a phase circle is a shell whose radius barely varies, so a centroid-based dispersion is degenerate for it (the first implementation returned 431), whereas pairwise distance is exactly the quantity a recurrence threshold is compared against.

Measured reference values: uniform unit circle **1.41** (theory 2/√2 = 1.414), 1-D uniform 2.36, Gaussian 3.30, Rayleigh 3.33, lognormal 8.32.

The fix, when it matters, is a shape-normalising per-block transform: `scale="rank"` removes skew entirely, `scale="robust"` limits the influence of outliers. The scale-free alternative is `per_block` thresholding [DD-35], which sidesteps relative weighting altogether.

---

<a id="dd-29"></a>
## DD-29 · Envelope smoothing is explicit

**Status:** accepted

The envelope of a band centred at f_high carries meaningful variation only up to roughly half the band's width; faster fluctuation is estimation noise that fills the state space and hides the modulation geometry.

Building the PAC attractors showed that without smoothing the cloud is visually unreadable and the toroidal structure is buried. Smoothing is standard practice in PAC work, but it **discards information**, so it is exposed as an explicit parameter (`smooth=<Hz>`) and recorded as a caveat. Never by default.

---

<a id="dd-30"></a>
## DD-30 · The whole repository is in English

**Status:** accepted

**Context.** Development discussion happens in Spanish, but the repository is public, the library will accompany papers in international journals, and its outputs will be pasted into the figures and tables of those papers.

**Decision.** In English, without exception: code and identifiers, comments and docstrings, figure titles, axis labels, legends and annotations, exported file names, column names of every exportable DataFrame, example scripts including their console output, run reports, README, CHANGELOG, and these design documents.

**Automatic enforcement.** Four tests verify it rather than trusting discipline:

- `test_source_has_no_spanish_identifiers` walks all of `src/recurra/**.py` looking for accented characters
- `test_figure_text_is_english` inspects titles, labels and legends of rendered figures
- `test_dataframe_columns_are_english` checks the columns of every exportable table
- `test_report_text_is_english` checks the rendered run report

**Output naming.** CSV files carry descriptive `snake_case` names (`route_comparison`, `threshold_modes`, `joint_recurrence`) with no numeric prefixes: generation order is a property of the script, not of the data. Figures follow `fig_<what>_<about_what>.png`, and comparison figures carry the `fig_compare_` prefix.

---

<a id="dd-31"></a>
## DD-31 · Single figures vs comparison mode

**Status:** accepted

**The problem found.** In the first version of F3, the multi-panel figures were assembled with raw matplotlib inside the example script. Consequence: **those views existed only inside the demo**. A user with real data could not reproduce them without copying and adapting the script, and some panels had no standalone equivalent anywhere.

**Decision.** Every plotting function falls into exactly one of two categories, and the signature enforces it:

| | Takes | Returns | Lives in |
|---|---|---|---|
| **Single figure** | *one* object (Recording, StateSpace, EmbeddingParams, RecurrenceMatrix, Threshold, DataFrame) | one Figure | `recurra.viz` |
| **Comparison figure** | a *mapping* of already-built objects | one Figure with panels | `recurra.viz.compare` |

Three rules follow:

1. **No plotting function generates data.** It does not filter a signal, build a state space or call the generator. The caller builds the objects; the plot only arranges them. That is why comparing synthetic regimes and comparing real subjects is exactly the same code.
2. **No panel is exclusive to comparison mode.** Every view appearing inside a multi-panel figure also exists as a single figure. Comparison mode is layout convenience, never the only door to a view.
3. **Drawing and saving are separate.** Functions return a `Figure` and write nothing; `save_figure(fig, name)` is an explicit step. So the same figure can be embedded in a notebook, composed with others, or saved.

`compare_attractors` and `compare_recurrence` default to `share_limits=True`: matplotlib's independent autoscaling makes different point clouds look alike, which in a comparison figure is simply misleading.

**Self-documenting catalogue.** `recurra.viz.list_figures()` and `recurra figures` list the single and comparison figures, stating what object each one takes. The `SINGLE_FIGURES` and `COMPARISON_FIGURES` registries are checked in tests, so a new unregistered figure fails the suite.

---

<a id="dd-32"></a>
## DD-32 · Sample pairs from the region the recurrence rate is defined over

**Status:** accepted

The recurrence rate is the fraction of recurrent pairs **within the region considered**, which excludes the Theiler band when one is used. Estimating a threshold from pairs drawn uniformly over all (i, j) — including the near-diagonal pairs that will later be discarded — biases the threshold downwards, because near-diagonal pairs are systematically closer than average.

So the sampler honours the Theiler window: pairs with `|i − j| ≤ theiler` are rejected and redrawn. Verified in tests: excluding the band raises the mean sampled distance, as it must.

The draft also concatenated subjects before sampling, mixing within-subject and between-subject distances and inflating the threshold. Cross-record pairs are excluded by construction here: each record is sampled separately and the samples are pooled.

A corollary: the recurrence rate's **denominator** must exclude the same band, or the reported rate is not comparable with the threshold's own estimate. Tested exactly.

Cross recurrence sampling deliberately applies **no** Theiler exclusion: the two trajectories are different systems, so index proximity carries no auto-correlation artefact to remove.

---

<a id="dd-33"></a>
## DD-33 · Squared-norm expansion, not broadcasting

**Status:** accepted

The draft built `Ei[:, None, :] - Ej[None, :, :]`, a temporary of `block² × dim` floats: 96 MB for a 1000-point block in 12 dimensions, allocated and discarded for every tile.

Euclidean distance is instead

$$\lVert a - b \rVert^2 = \lVert a \rVert^2 + \lVert b \rVert^2 - 2\,a \cdot b$$

where the cross term is a single matrix product routed to BLAS. One temporary of `block²` floats — a factor of `dim` less memory and roughly an order of magnitude faster. Squared distances are carried throughout and compared against a squared threshold, so no square root is taken on the hot path.

Verified numerically identical to naive broadcasting to 1e-9, and clamped at zero so floating-point noise cannot produce negative squared distances.

---

<a id="dd-34"></a>
## DD-34 · The library refuses rather than thrashes

**Status:** accepted

Principle P7 says the memory budget is declared, not discovered when the machine starts swapping. `plan()` reports what each storage strategy would cost and picks one that fits; when nothing fits it raises with the specific numbers, so the user knows whether to window, subsample or stream.

Measured, with an 8 GiB budget:

| Points | Duration at 500 Hz | Dense | Sparse at 5% | Chosen |
|---|---|---|---|---|
| 5 000 | 10 s | 0.02 GiB | 0.005 GiB | memory |
| 50 000 | 100 s | 2.3 GiB | 0.47 GiB | memory |
| 200 000 | 400 s | 37 GiB | 7.5 GiB | streaming |
| 1 800 000 | 1 hour | **3017 GiB** | 604 GiB | streaming |

An hour of EEG is three terabytes dense. The planner streams it instead, and the metrics remain exact [DD-36].

---

<a id="dd-35"></a>
## DD-35 · The default is never a fixed epsilon

**Status:** accepted

The project records the problem plainly: recurrence density falls exponentially with dimension at fixed epsilon, so a threshold that gives a sensible plot in 3 dimensions gives an empty one in 12.

The default mode is therefore `target_rr`: fix the recurrence rate, let epsilon follow. What a paper should report is the **achieved** rate, which is what `Threshold.achieved_rr` carries and what `RecurrenceMatrix.describe()` puts next to the target.

`method="bisect"` refines the sampled quantile against fresh, independent samples until the achieved rate is within tolerance. The plain quantile is exact for the sample it saw but slightly biased for the population.

**Measured accuracy.** Across targets from 0.5% to 30%, and both methods, the worst absolute deviation between requested and actually achieved rate was **0.0036**.

Modes available: `fixed`, `target_rr`, `percentile`, `fan`, `std_fraction`, `maxdist_fraction`, `per_block`, `adaptive_dim`.

Two are worth singling out:

- **`fan`** (fixed amount of neighbours) gives every point its own epsilon, the distance to its k-th neighbour. It keeps the neighbour count constant instead of the radius, which is what one wants in inhomogeneous attractors — at the cost of a matrix that is **not symmetric**, recorded as a warning on the threshold.
- **`per_block`** sets one epsilon per coordinate block and combines them with Chebyshev, so **no relative weighting is needed at all**. It is the scale-free alternative to [DD-02] and to the shape problem of [DD-28].

`adaptive_dim` additionally reports epsilon as a fraction of the attractor radius, and warns above 0.5 that the plot is reflecting the size of the cloud rather than its structure.

A guard rail worth noting: `percentile` rejects values outside [0, 1] with a message naming the mistake, because the draft's README passed `percentile=5.0` meaning the fifth percentile, which silently clamped to 1.0 and produced a completely full recurrence plot.

---

<a id="dd-36"></a>
## DD-36 · The matrix is an implementation detail

**Status:** accepted

Principle P2. For an hour of EEG at 500 Hz the recurrence matrix has 3.2×10¹² entries; it cannot be an array, and an API built around one would be unusable exactly where the project needs it most.

So `RecurrenceMatrix` is a **lazy engine** that knows how to produce any tile of itself on demand. Storing the whole thing is an option (`store="memory"`), not the premise. `store="sparse"` keeps only the recurrent pairs; `store="none"` never allocates the matrix at all.

Everything that matters — the recurrence rate and density profile now, the line-length histograms in F6 — is computed by streaming tiles and accumulating.

**Streaming results are exact, not approximate.** Verified: for tile sizes 64, 128, 512 and 4096, the reassembled matrix is identical to the dense one bit for bit, and the recurrence rate matches to relative tolerance 1e-12.

---

<a id="dd-37"></a>
## DD-37 · A run should explain itself

**Status:** accepted

CSV tables carry the numbers and provenance sidecars carry the operations, but neither answers the question a user actually asks three weeks later: *what did that run do, and which of these files is which?*

`RunReport` accumulates sections, notes, timings, warnings and an index of every artefact written, then writes one readable TXT next to the outputs, plus an artefact index as CSV. It records rather than computes: anything in the report was produced by the analysis, and the report never derives a result of its own.

The report also collects the caveats raised along the pipeline, so the things the analysis did without being asked — edge trimming, envelope smoothing, block alignment, a dominant scale block — end up in one place a reader will actually see.

---

<a id="dd-38"></a>
## DD-38 · Pooling for display, watching for saturation

**Status:** accepted · **Corrected after looking at the output**

A recurrence plot is a texture of thin diagonal and vertical lines. Sampling one pixel in eight to fit a display drops most of those lines and keeps a speckle that looks like noise — precisely the structure the analysis is about. So an oversized plot is reduced by **pooling** rather than subsampling.

Maximum pooling preserves lines, but **it saturates**. At reduction factor f and rate r, an output pixel is lit with probability 1 − (1 − r)^(f²) if recurrences were independent: for f = 5 and r = 5% that is 72%, and the picture turns almost solid. The first version of the comparison figure was nearly black, and the structure was as thoroughly destroyed as it would have been by naive subsampling.

Real recurrence plots are clustered rather than independent, so the formula is an upper bound; the measured inflation is smaller but still large — a 5× reduction of a 5% plot reads as 25% density under maximum pooling.

`pooling="auto"` (the default) estimates that saturation and switches to mean pooling, which preserves density instead of lines, whenever maximum pooling would light more than 35% of the pixels. Mean-pooled images are rescaled to their own upper quantile so contrast survives. Whichever method was used is written into the caption.

A second bug surfaced in the same place: the streaming renderer anchored its bin edges to each tile's start rather than to the global grid, so any tile size that was not a multiple of the reduction factor shifted the blocks. Now tested against the dense path for three deliberately awkward tile sizes and both pooling methods.


---

<a id="dd-39"></a>
## DD-39 · One window spec, three ways to state it

**Status:** accepted

Users describe windows in whichever way the problem hands them: *"four-second windows every half second"*, *"ten windows covering the record with 50% overlap"*, or explicit sample indices. All three produce the same `WindowSpec`, so everything downstream sees one representation.

Windows are **resolved against a record length at the moment they are used**, not when they are declared, so the same spec applies to subjects of different length — the normal case with real data [DD-18].

**A correction made while testing.** The first implementation of `n_windows=k` derived a step and walked forward, which gave nine windows for `n_windows=10, overlap=0.5` and left 5.4 s of a 60 s record unanalysed. If someone asks for ten windows they should get ten, covering the record. Now the starts are placed evenly between 0 and `n − size`, which guarantees exactly `k` windows spanning from the first sample to the last; any discrepancy between the requested and realised overlap is reported as a caveat.

For the `size`/`step` form, an uncovered tail is normal and is **reported**, not silently dropped: *"the last 1500 samples (3.000 s) do not fill a whole window and are not analysed; set drop_last=False to include them"*. With `drop_last=False` a final window is pulled back to the record end, and the extra overlap that creates is recorded too.

---

<a id="dd-40"></a>
## DD-40 · Threshold scope decides what the windows can say

**Status:** accepted · **The consequential choice in windowed analysis**

Fixing the recurrence rate **per window** makes every window comparable in structure but constant in density: RR carries no information, by construction, and DET, LAM and the rest describe geometry at matched density.

Fixing one threshold **globally** instead turns RR(t) into an informative time series — density itself becomes the signal — at the cost that a window whose amplitude drifts may come out nearly empty for reasons that have nothing to do with its dynamics.

Both are right for different questions, neither is right for both, and the choice silently changes what a windowed analysis *means*. So `scope` is an explicit argument, the invalid-value message explains the stakes rather than just listing the options, and the scope used is recorded on **every row** of the metrics table.

Measured on one PAC record, ten windows at 50% overlap, 5% target:

| scope | epsilon std | RR mean | RR std | RR range |
|---|---|---|---|---|
| `per_window` | 0.0213 | 5.01% | 0.0004 | 4.96–5.09% |
| `global` | 0.0000 | 5.06% | 0.0043 | 4.14–5.64% |

Exactly the trade: one quantity is pinned and the other moves. `per_window` is the default because comparing structure is the more common intent.

---

<a id="dd-41"></a>
## DD-41 · Windows are lazy

**Status:** accepted

A hundred subjects with ten windows each is a thousand recurrence matrices. Materialising them all is pointless when the goal is a metrics table, so `WindowedRecurrence` holds the trajectory, the window list and the thresholds, and builds a matrix only when one is asked for.

`keep=True` caches them when they will be reused (drawing every window, for instance); the default computes metrics and lets each matrix go, so a corpus costs one window of memory at a time regardless of its size.

This composes with the storage strategy of [DD-36]: a lazy window whose matrix is itself streamed never allocates anything larger than a tile.

`batch_windowed_recurrence` gives each record an independent random stream derived from the caller's seed [DD-05], so results do not depend on the order records are processed in, or on how the work would later be distributed across workers.

**Validated end to end.** Eight subjects at coupling strengths from 0 to 1, six windows each at 50% overlap, joint recurrence between a 2-D phase circle and a 1-D envelope: the mean independence ratio ranks the subjects perfectly against the ground-truth alpha, Spearman **+1.000**. On non-stationary signals the same measure traces the coupling in time — a bell for transient coupling, a monotone rise for a ramp, flat at 1.0 for the uncoupled control.


---

<a id="dd-42"></a>
## DD-42 · Comparability is a contract, not an accident

**Status:** accepted

The project's aim is to relate RQA and nonlinear metrics to classical coupling indices, and eventually to separate groups by them. That only means anything if the structures being compared were built the same way.

**Three defaults broke this silently**, and all three were reachable from the API as it stood. Measured on three subjects of 60, 45 and 72 s:

| Failure | What happens |
|---|---|
| `m="auto"`, `tau="auto"` | dimensions 5, 5 and 6 — subjects end up in different phase spaces |
| `n_windows=k` | window 0 covers 10.6 s, 7.9 s and 12.8 s. "Window 0" stops meaning the same thing |
| `subsample(max_points=N)` | decimation factors 5.85, 4.35 and 7.05 → effective rates 85, 115 and 71 Hz |

None of the three announces itself, and each produces perfectly plausible numbers.

**Decision.** A `ComparabilityContract` states the parameters that must be shared; `check_comparability` verifies that the structures actually produced satisfy it and names the offenders. The contract is data, exported next to the results, so a reader can see what was held fixed.

Axes checked: `dim`, `route`, `scaling`, `blocks`, `m`, `tau`, `fs`, `n_points`, `metric`, `theiler`, `threshold_mode`, `target_rr`, `epsilon`, `window_samples`, `window_step`.

Without a contract, every axis present is checked for **constancy**, which answers "are these comparable at all?" — so the check costs nothing to run and `batch_windowed_recurrence` runs it by default. Passing `WindowSpec(n_windows=...)` to a batch now warns explicitly, because that is the one spec form that cannot give a shared window grid across records of unequal length.

**The fix for cross-subject work** is a window size fixed in seconds or samples. Records then differ in the *number* of windows they yield, which is fine and expected, while window *k* covers the same stretch of every record. Verified: with `WindowSpec(size=6.0, overlap=0.5)`, window 0 spans 0.75–6.75 s with N = 3000 for all three subjects, and they yield 18, 13 and 22 windows respectively.

---

<a id="dd-43"></a>
## DD-43 · What matched decides what may be compared

**Status:** accepted

Matching is not all-or-nothing, and a plain pass/fail verdict would be misleading in both directions.

- Holding the **recurrence rate** fixed makes DET, LAM and the line statistics comparable while making RR itself uninformative by construction.
- Holding **epsilon** fixed does the opposite: RR becomes the signal, and the line statistics are no longer at matched density.
- Comparing any **length-valued** metric (L, Lmax, trapping time, and the entropies computed from line-length distributions) additionally requires the same number of points, because those distributions are bounded by N.
- Comparing anything in **time units** requires the same effective sampling rate.

So `ComparabilityReport.metric_validity()` returns, per metric family, what it requires and whether the observed matching supplies it:

| Family | Requires | Metrics |
|---|---|---|
| `recurrence_rate` | dim, metric, theiler, **epsilon** | RR, independence ratio |
| `line_length` | dim, metric, theiler, **n_points**, fs, target_rr | L, Lmax, TT, Vmax, DIV |
| `line_ratio` | dim, metric, theiler, target_rr | DET, LAM, transitivity |
| `entropy` | dim, metric, theiler, **n_points**, target_rr | ENTR, ENTW, RTE |
| `timescale` | dim, metric, theiler, **fs**, target_rr | T1, T2, TREND |
| `invariant` | dim, metric, **fs** | D2, λ₁, K₂ |

Demonstrated on the failing batch above: the report refuses `line_length` and `entropy` for lack of a common N, while correctly certifying `line_ratio` — DET and LAM survive because they are ratios. That is a more useful verdict than "not comparable", and a safer one than silence.

---

<a id="dd-44"></a>
## DD-44 · Groups are compared like with like

**Status:** accepted

In a windowed analysis, window *k* only means the same thing across subjects when the window grid is shared [DD-42]. So `group_comparison` compares **within each window index by default**: every subject's window 0 against every other subject's window 0, and so on. Pooling all windows is available with `by=None`, but it is not the default, because pooling mixes early and late stretches of a record and hides any time structure the analysis exists to find.

Per window and metric the table reports group sizes, means, standard deviations, the difference, Cohen's d, the rank-based AUC (the probability that a random member of group B exceeds one of group A) and an uncorrected two-sided p-value. The **number of tests is in the table** and a Bonferroni-corrected column sits next to the raw one: windows overlap, so the tests are not independent and Bonferroni is conservative — the correction is offered, not imposed, and the multiplicity is made impossible to overlook.

`feature_matrix()` pivots the tidy table into subject-by-window form, which is what a classifier consumes; records that yield fewer windows leave missing entries rather than being silently truncated.

**Validated end to end.** Twenty subjects, ten per group, record lengths of 40/45/50 s, joint recurrence between a 2-D phase circle and a 1-D envelope, 264 structures, comparability verified:

- 10 of 15 windows separate the groups after Bonferroni correction
- mean AUC 0.972, mean Cohen's d 2.25
- Spearman between the true coupling strength and the mean recurrence-based index: **+0.937**

That last number is the parallel the project is looking for, obtained from recurrence geometry alone, before any RQA metric exists.


---

<a id="dd-45"></a>
## DD-45 · The documents are kept in step by tests

**Status:** accepted

Three documents are maintained alongside the library: design decisions, architecture and examples. They were being updated by hand at the end of each phase, which works exactly until it does not — a document drifts the moment a phase lands, and nobody notices until someone reads a stale claim and acts on it.

**Decision.** The parts that can be checked mechanically are checked in the test suite:

| Test | What it catches |
|---|---|
| index vs sections | a decision listed but not written, or written but not listed |
| numbering | gaps and duplicates in the decision numbers |
| citations resolve | a `[DD-nn]` in a docstring pointing at a decision that does not exist |
| citations exist | a decision about the code with no trace in the code |
| module map | a module added to the package but absent from the architecture map |
| figure catalogue | a figure registered but not documented |
| test count | a stale number, the first sign a document stopped being maintained |
| example scripts | a script the README advertises that does not exist, or one nobody is told about |
| public API | an exported name that appears in no example |

**A gap it did not close, found later.** Two implementations of the RQA layer
both passed their own suites while exporting different metric sets -- one with
`RTE`, one with `RATIO`. The design document reported measured results for
`RTE`, so the version without it contradicted its own documentation and
nothing noticed: the checks tied the *shape* of the documents to the code, not
their *claims*. `test_documented_metrics_all_exist` now verifies that every
metric a document reports a number for is one the code produces. Naming a
metric in prose is a claim that it exists.

**What it found on its first run**, which is the argument for having it:

- three decisions (DD-14, DD-17, DD-29) with no citation anywhere in the code
- one comparison figure missing from the architecture catalogue
- **22 exported names appearing in no example** — `Recording`, `Capability`, `Step`, `get_config`, `spawn_rngs`, `threshold_sampling` and others were reachable from the top-level namespace but undocumented, which is how an API grows a shadow half nobody knows about

The last one produced a public API index in the examples document, which should have existed anyway.

**What this cannot do** is verify that the prose is *true*. A design decision can be described accurately and still be the wrong decision, and no test will say so. It verifies that the documents are not obviously stale, which is where the rot starts, not where it ends.

**A note on the test count.** The architecture document states the number of *declared test functions* rather than the number of cases pytest collects, because parametrisation makes the second ambiguous to count without running the suite. Stating the unambiguous number keeps the check exact.


---

<a id="dd-46"></a>
## DD-46 · Warnings carry their own category

**Status:** accepted

The library warns about things it did without being asked: edge trimming, envelope smoothing, a channel chosen by frequency, a block dominating the geometry, records that are not comparable. Those are valuable during analysis and pure noise during a test run — a first local run of the suite produced pages of them.

The blunt fix, `warnings.simplefilter("ignore")`, was what the example scripts used, and it is wrong: it silences numpy and scipy too, including the ones raised for good reason about overflow or invalid values.

**Decision.** A warning hierarchy of the library's own:

```
RecurraWarning(UserWarning)
├─ CaveatWarning         something was done that the caller did not ask for
├─ InferenceWarning      a choice was made on the caller's behalf
├─ GeometryWarning       the state space may not measure what was intended
└─ ComparabilityWarning  analyses about to be compared were built differently
```

So a user can quieten the library without going deaf to everything else:

```python
warnings.simplefilter("ignore", recurra.RecurraWarning)
warnings.simplefilter("ignore", recurra.InferenceWarning)   # or just one kind
```

The test suite filters `RecurraWarning` through `pyproject.toml`. Tests that assert on a warning use `pytest.warns`, which overrides those filters, so nothing is hidden from the tests themselves.

**Silencing a warning does not lose the record.** Every caveat is also written into the provenance trail and, where a run report is used, collected into its warnings section. The warning is the notification; the provenance is the evidence.


---

<a id="dd-47"></a>
## DD-47 · Decimation changes the sampling rate, and the object must say so

**Status:** accepted · **Found by a user reading the output, not by a test**

`StateSpace.subsample()` kept every k-th point but left `fs` untouched. The
space then reported a duration k times shorter than the stretch of signal it
actually covered:

```
StateSpace(route='observable', 1200 points, dim=3, 2.40 s @ 500 Hz)
```

That trajectory covered **18.5 s**, not 2.40 s. Everything expressed in time
inherited the error: figure axes, window boundaries resolved in seconds,
density profiles, the `t_start_s` column of every windowed metrics table.

The caveat did say *"time-based quantities change units accordingly"*, which
was true and useless: it told the user to do the arithmetic the library should
have done.

**Decision.** Keeping every k-th point means consecutive states are k original
samples apart, so the effective rate is `fs / k`. `subsample` now divides the
rate by the realised factor and records both in the provenance:

```
{'mode': 'max_points:1200', 'n_before': 9250, 'n_after': 1157,
 'factor': 8.0, 'fs_before': 500.0, 'fs_after': 62.5, 'uniform': True}
```

**A second change followed.** `max_points` used `np.linspace`, which hits the
requested count exactly but leaves unequal gaps — and a trajectory with
unequal gaps has no single sampling rate at all. It now uses a uniform stride,
returning slightly fewer points than asked for with a well-defined rate. That
is the better trade: `max_points=1200` on 9250 points gives 1157 points at
62.5 Hz rather than exactly 1200 points at an undefined rate.

Irregular explicit `indices` are still allowed, since a user may have good
reason, but the rate is left alone and a warning states that nothing in
seconds can be trusted for such a space.

**It also fixed a comparability check.** Asking for the same point count from
records of different length decimates them by different factors — failure mode
3 of [DD-42]. Before this change the rates were all reported as the original,
so the check could not see the divergence; it now flags `fs` as violated and
correctly withdraws the `timescale` and `invariant` metric families.

**The lesson.** The library was careful to *record* the simplification and
careless about *propagating* its consequence. Recording is not enough when the
recorded fact contradicts an attribute the object still advertises.


---

<a id="dd-48"></a>
## DD-48 · Phase locking is a bounded deviation, not a mixture of phases

**Status:** accepted · **A generator bug, found by looking at a scatter matrix**

The `ppc` generator built the fast phase as a convex mixture of two *unwrapped*
phases:

```python
phi_high = angle(exp(1j * (alpha * locked + (1 - alpha) * free)))
```

That does not interpolate between locked and free. An unwrapped phase is a
ramp whose slope is the frequency, so a weighted average of two ramps is a
ramp at the **weighted average frequency**. Measured on the generator's own
components, with no filtering involved:

| alpha | PLV (1:1) | mean frequency of phi_high |
|---|---|---|
| 0.00 | 0.0005 | 59.5 Hz |
| 0.50 | 0.0014 | 32.7 Hz |
| 0.90 | 0.0071 | **11.2 Hz** |
| 1.00 | **1.0000** | 5.9 Hz |

At alpha = 0.9 the "fast" component oscillated at 11 Hz. There was no phase
locking at any alpha except exactly 1, where the mixture degenerates to the
locked term and the right answer appears by accident.

**Decision.** Locking is n:m plus a *bounded* phase deviation:

```
phi_high = (n/m) * phi_low + s(alpha) * w(t)
```

with `w` a zero-mean random walk and `s(alpha) = sqrt(-2 ln alpha)`, which
inverts the standard relation `PLV ≈ exp(-s²/2)`. The frequency is then set by
the locking ratio alone, and alpha controls only how tightly the phase adheres
to it.

Measured after the fix, `nm_ratio=(10,1)`, f_low 6 Hz, f_high 60 Hz:

| alpha | 0.00 | 0.25 | 0.50 | 0.75 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|
| PLV | 0.048 | 0.193 | 0.247 | 0.532 | 0.798 | 1.000 |

Spearman(alpha, PLV) = 1.00, and the locked component stays at 58 Hz
throughout, inside the band a gamma filter would keep.

**A second problem in the same place.** `nm_ratio=(1,1)` with f_low = 6 and
f_high = 60 is incoherent: 1:1 locking means the same frequency, so the "fast"
component oscillates at 6 Hz and a band-pass at 50-70 Hz removes it entirely.
The generator now checks `n/m` against `f_high / f_low` and raises a
`ParameterWarning` naming the frequency the component will actually have and
the ratio that would be consistent.

**Why the tests missed it.** `test_ppc_locks_phases` compared alpha = 0 against
alpha = 1 and asserted only that the second was larger. Both endpoints were
right; everything between them was wrong. The replacement checks monotonicity
across the whole range and that the locked component stays inside its band.

---

<a id="dd-49"></a>
## DD-49 · Shared axes only between comparable spaces

**Status:** accepted

[DD-31] gave `compare_attractors` shared axes by default, because matplotlib's
independent autoscaling makes different point clouds look alike. The opposite
failure turns out to be just as real. Extents of three spaces drawn in one
figure:

```
pac      [ 2.09,  2.09,  5.61]
lorenz   [38.5,  45.9,  41.2 ]
takens   [36.3,  36.3,  36.3 ]
```

Sharing limits across those shrinks the PAC panels to a dot. Worse, the axes
are not the same quantities at all: one panel's x is a cosine, another's is a
position in a Lorenz system.

**Decision.** `share_limits="auto"` is the default. Axes are shared only when
the spaces are built from the same coordinate blocks *and* their extents are
within a factor of four. Otherwise each panel scales itself and the figure
says why in its subtitle: *"axes not shared: the spaces have different
coordinate blocks"*. `True` and `False` remain available to force either way.

---

<a id="dd-50"></a>
## DD-50 · Robust axis limits for heavy-tailed coordinates

**Status:** accepted

Instantaneous frequency from a Hilbert transform is spiky: a phase slip sends
it to tens of hertz, and negative values are physically impossible but
mathematically unavoidable when the phase momentarily retreats. Measured on a
theta band:

```
median 5.68 Hz | 1-99 percentile [1.3, 7.9] | min/max [-12.0, 24.4]
```

Autoscaling to the full range compresses 98% of the data into a smear.

**Decision.** `plot_attractor` defaults to limits at the 0.5-99.5% quantiles,
**but only when the extremes actually distort the view**: the full range must
exceed the quantile span by more than a factor of three. Calibrated:

| coordinate | full range | verdict |
|---|---|---|
| Gaussian | [4.05, 7.63] | untouched |
| uniform | [0.00, 1.00] | untouched |
| circle coordinate | [-1.00, 1.00] | untouched |
| instantaneous frequency with slips | [-12.0, 80.0] | clipped to [4.6, 7.4] |
| lognormal | [0.01, 49.5] | clipped to [-0.6, 15.0] |

Well-behaved coordinates are left exactly as they are. When clipping happens
the subtitle says so, and `clip=None` disables it. The data are never
modified; only the view is.

---

<a id="dd-51"></a>
## DD-51 · Circle blocks show the phase, not its cosine

**Status:** accepted

In `mode="pairs"`, the diagonal histogram of a `phase_circle` coordinate is
the distribution of cos φ, which for a uniformly rotating phase is an arcsine
U-shape: piled up at ±1, thin in the middle. It is the correct answer to a
question nobody asked, and it is identical for every signal.

**Decision.** For a `phase_circle` block the diagonal shows the histogram of
the phase itself over [-π, π], which is flat under free rotation and shows the
concentration when the phase is locked or preferred. The panel is labelled
`phase` so it is not mistaken for a coordinate histogram.

---

<a id="dd-52"></a>
## DD-52 · A clean check reports what it checked

**Status:** accepted

`plot_quality` drew an empty frame containing the words "no issues detected"
when nothing was wrong. It was working, and it read as a panel that had failed
to render — which was the actual report from a first-time user.

**Decision.** The panel lists the checks and their verdicts either way:
sampling rate, equal channel lengths, non-empty channels, finite samples,
non-constant channels, converter clipping. Green for passed, amber for a
warning, red for an error, with the count in the title. A clean recording now
produces a panel that says what was verified, which is information; the
previous version conveyed only that the author had thought about it.


---

<a id="dd-53"></a>
## DD-53 · Route comparison: what may be compared, and which route to pick

**Status:** accepted

`compare_routes` builds the same recording several ways and reports geometric
descriptors, which is the empirical answer to "which state space should I
use?" that objective O2.1 demands. Three things the figure got wrong.

### Scale-dependent descriptors are not comparable across routes

The mean nearest-neighbour distance is expressed in the units of the weighted
space. Measured on one recording:

| route | dim | scaling | nn_mean | nn_cv | corr_slope |
|---|---|---|---|---|---|
| `observable_pac` | 3 | rms_balanced | 0.032 | 1.000 | 1.56 |
| `hybrid_pac_m3` | 5 | rms_balanced | 0.172 | 0.525 | 2.18 |
| `observable_envelope` | 2 | rms_balanced | 0.020 | 1.035 | 1.65 |
| `delay_takens` | 5 | **none** | 0.607 | 0.781 | 2.24 |

The Takens route sits twenty times higher on `nn_mean` than the PAC space, for
no reason but its scaling policy. Reading that panel as geometry is the same
mistake DD-49 fixed for shared axes, left unfixed here. Scale-dependent
descriptors are now excluded from the default set, and when requested across
mixed scalings their panel is greyed and labelled *"not comparable: the routes
mix scaling policies"*.

### A route that failed must not vanish

`compare_routes` reports failures as rows carrying a status message rather
than raising, so a route that does not apply to a recording is recorded. On
the recording above, one of five routes failed:

```
observable_phase_only  failed: ParameterError: spec[0] phase_circle: several
    channels have role 'phase' (['phase_theta', 'phase_gamma']); name one
```

The figure plotted only the successes, so a reader could not tell that a fifth
route had been attempted at all. Failures now appear in the caption with their
reason.

### The most interpretable descriptor was missing

`compare_routes` computes the fraction of nearest neighbours each route shares
with a reference, which answers the question a user actually has: *do these
constructions see the same geometry?* It was in the table and not in the
figure. It is now in the default metric set.

The answer, on the same recording, is that they do not:

| route | neighbours shared with `observable_pac` |
|---|---|
| `observable_pac` | 1.000 |
| `hybrid_pac_m3` | 0.198 |
| `observable_envelope` | 0.058 |
| `delay_takens` | 0.030 |

The hybrid space shares a fifth of its neighbourhoods with the observable one
despite being built from the same data; Takens shares three per cent. **The
choice of route is not an implementation detail.** Two routes over one
recording produce spaces that barely agree on who is near whom, and will
therefore produce different recurrence plots and different RQA.

### Which route to build recurrence plots from

The descriptors above characterise a geometry; they do not rank one. A
criterion needs a question, and the question here is whether the geometry
*responds to coupling*. Measured as Cohen's d between alpha = 0 and alpha =
0.95, three seeds each:

| route | slope at alpha 0 | at alpha 0.95 | d (slope) | d (nn_cv) |
|---|---|---|---|---|
| `observable_pac` | 1.96 | 1.40 | **-45.2** | 1.0 |
| `hybrid_pac_m3` | 3.16 | 2.07 | **-75.0** | 3.5 |
| `observable_envelope` | 1.84 | 1.50 | -12.1 | 2.2 |
| `delay_takens` | 2.61 | 2.17 | -11.8 | 35.4 |

Read with care -- these effect sizes are enormous because the synthetic
signals differ only in alpha, with the same seed and spectrum. On real data
they will be far smaller. The *ordering* is what matters.

**The hybrid route responds most strongly**, and it is also the most
homogeneous space (nn_cv 0.52 against 1.00), which matters practically: a
single threshold suits a homogeneous cloud everywhere and an inhomogeneous one
only in places.

**The observable PAC route is the one that encodes the hypothesis.** Its
coordinates are physical, its geometry is the torus of Hypothesis 1, and a
result stated in it can be explained. The hybrid space buys sensitivity with
three delay coordinates of the envelope that have no individual meaning.

**Takens is the control, not a candidate.** It assumes nothing about coupling,
so it is the right null: if RQA on a Takens reconstruction separates groups as
well as RQA on the PAC space, the cross-frequency framing is adding nothing
and should be abandoned. Its large `nn_cv` effect is a warning rather than a
result -- coupling changes the raw signal's own geometry, so a difference
there does not license any statement about phase and amplitude.

**Recommendation.** Build on `observable_pac` for anything to be interpreted,
and report `hybrid_pac_m3` alongside as a sensitivity check. Run `delay_takens`
as the null.

### The decisive comparison, now that the RQA layer exists

Cohen's d between alpha = 0 and alpha = 0.95, three seeds, per route:

| route | DET | L | Lmax | ENTR | LAM |
|---|---|---|---|---|---|
| `observable_pac` | **63.0** | 7.8 | 2.8 | 11.3 | -48.9 |
| `hybrid_pac_m3` | 10.9 | 7.9 | 1.3 | 8.4 | -17.2 |
| `envelopes` | 23.5 | **11.5** | 1.5 | 11.2 | 18.1 |
| `delay_takens` | 2.3 | -4.0 | -0.8 | -5.1 | 5.4 |

**This reverses the ordering the geometric criterion gave.** On the
correlation-sum slope the hybrid route responded most strongly; on DET the
observable PAC space responds six times more strongly than the hybrid. The
recommendation stands, but now for a better reason: the space that encodes the
hypothesis is also the one whose determinism responds most to the coupling.

Two details worth keeping. `delay_takens` responds barely at all (d = 2.3 on
DET), which is the null behaving as a null should: RQA on a reconstruction
that knows nothing about phase and amplitude does not detect the coupling, so
the cross-frequency framing is doing real work. And LAM moves **downwards**
under coupling in the PAC space (d = -48.9): the laminar states become less
laminar, which is worth understanding before LAM is used as a coupling index.

The recommendation is no longer provisional on this point. It remains
provisional on real data.


---

<a id="dd-54"></a>
## DD-54 · The joint independence ratio detects coupling in time

**Status:** provisional — a measured result, not yet a validated method

### What the quantity is

Build two subsystems separately, each in its own phase space with its own
threshold fixed to its own recurrence rate [DD-20]: the theta phase circle
(2-D) and the gamma envelope (1-D). The joint recurrence plot is their
elementwise AND.

If the two recurred independently, the chance of coinciding would be the
product of their rates. The **independence ratio** is the observed joint rate
divided by that product, so 1.0 means independence and above 1.0 means the two
subsystems tend to return to previous states at the same moments.

### Why it is not visible to either subsystem alone

Measured on a transiently coupled signal, windows of 4 s at 75% overlap, each
subsystem held at a 10% rate:

| | RR phase | RR amplitude | product | joint RR | ratio |
|---|---|---|---|---|---|
| outside the transient | 0.1001 | 0.0999 | 0.01000 | 0.00982 | 0.98 |
| inside | 0.1000 | 0.1001 | 0.01000 | 0.01678 | 1.68 |

The subsystem rates are **identical** inside and out. Nothing an analysis of
either signal on its own could see has changed; only the coincidence has. That
is the argument for the joint construction, and it is why the rate must be
matched per subsystem rather than shared.

### Detection of a transient

Ground truth from the generator: alpha = 0.95 between 19.8 s and 39.6 s, zero
elsewhere. Detection criterion, baseline plus three standard deviations:

| | true | detected | latency |
|---|---|---|---|
| onset | 19.8 s | 19.8 s | **-0.1 s** |
| offset | 39.6 s | 39.8 s | **+0.1 s** |

The window step is 1 s, so the latency is below the resolution of the grid:
the method detects the transition as fast as the windowing allows.

The transitions occupy 2 to 3 seconds on the curve. That is not gradual onset
in the signal, which is a step, but the window straddling it: a window sitting
half inside the transient reports half the effect. **The width of the
transition is the window length, not a property of the signal.**

### Three things this does not yet establish

**The baseline is not exactly 1.** Measured at 0.983 with a standard deviation
of 0.053, slightly below the theoretical value. The bias comes from estimating
each rate by a sampled quantile inside a 400-point window, and from the Theiler
window removing pairs from the denominator. Using 1.0 as the null would be
wrong; a serious test needs the baseline calibrated against surrogates.

**Consecutive points are not independent.** At 75% overlap, adjacent windows
share three quarters of their data. Any statistic computed across the series
must account for that, and the figure draws each window's extent as a
horizontal segment so the overlap is impossible to forget.

**Only one modality, one SNR, one record length.** Everything above is PAC at
20 dB over 60 seconds of synthetic signal. Nothing here shows the ratio
survives realistic noise, shorter records, or applies to PPC and AAC.

### Why it matters to the project

This is a purely geometric coupling detector, obtained before any RQA metric
exists, with time resolution equal to the window step and latency below it. It
is a working prototype of objective O3.1.

When the RQA layer lands, DET and LAM will appear in the same per-window table
and can be compared against this ratio directly: which of them traces the
transient more sharply, which detects weaker coupling, which is more robust to
noise. That comparison is the point of the project, and this decision records
the geometric baseline it will be measured against.


---

<a id="dd-55"></a>
## DD-55 · Four statuses cannot live in colour alone

**Status:** accepted

The comparability figure encoded four verdicts — `ok`, `violated`, `absent`,
`unchecked` — as four colours with no legend. A reader had to ask what a grey
bar meant, which is the report that the figure had failed.

The distinction it was failing to convey is not cosmetic. **`absent` and
`violated` mean opposite things.** Violated is a quantity that exists for every
unit and disagrees between them. Absent is a quantity that does not exist for
these units at all.

The case that arises constantly: with `scope="per_window"` every window carries
its own epsilon, so a record has no single epsilon to report and the axis comes
out `absent`. Any metric family requiring it is then withheld — correctly,
because a recurrence rate pinned to a target by construction carries no
information to compare across subjects [DD-40]. Reading that grey bar as a
failure would send a user hunting for a bug that is not there.

The figure now carries a legend naming each status, annotates absent axes with
*"not reported by these units"*, and labels each withheld metric family with
the requirement it lacks — turning a red bar from a verdict into an
instruction.


---

<a id="dd-56"></a>
## DD-56 · Line histograms carry runs across tiles

**Status:** accepted

Every RQA metric is a function of three distributions: the lengths of the
diagonal lines, the vertical lines and the white vertical lines. Those can be
accumulated tile by tile, so the recurrence matrix never has to exist [DD-36].
The price is that a run crossing a tile boundary must be *carried*: its length
so far held open, completed only when a zero closes it or the matrix ends.

Tile order is not free. Tiles are visited row band by row band, left to right
within a band, because that is the order in which a diagonal and a column each
meet their own cells contiguously. One open length is carried per column
(`n_cols` values) and one per diagonal (`n_rows + n_cols` values).

**Verified exactly.** Against a deliberately naive dense reference, for point
clouds of 37, 64, 100 and 211 points, Theiler windows of 0, 1 and 3, and tile
sizes of 7, 16, 31 and 1024 — 48 combinations, all three histograms identical
in every one. Not approximately: the same integers.

**What the check caught.** The first implementation agreed on diagonals and
verticals in all 48 cases and disagreed on white lines in 24 of them. The
cause was a discarded run extending past the tile where it met the Theiler
band: the fragment inside that tile was dropped and the remainder counted as a
fresh run. The fix is a per-column *dropping* state that persists across
tiles. Neither reading the code nor reasoning about it found this; comparing
against the reference did, in one run.

---

<a id="dd-57"></a>
## DD-57 · White lines and the Theiler band

**Status:** accepted

A white vertical line is a run of non-recurrent cells and its length is a
recurrence time. But cells excluded by a Theiler window are stored as zero, so
they read as white and would inflate every recurrence time by the width of the
band.

Excluded cells therefore break the column into segments, and a white run
touching the band is **discarded** rather than counted: it was truncated by a
choice of analysis rather than by a recurrence.

Runs reaching the start or end of the record are **kept**. They are truncated
too, but by the data rather than by a choice, and dropping them would bias the
distribution against long recurrence times — exactly the ones a recurrence
time entropy is sensitive to. The asymmetry is deliberate and worth knowing
when reading RTE.

---

<a id="dd-58"></a>
## DD-58 · The DET denominator is a choice, and it is stated

**Status:** accepted

DET is the fraction of recurrent points lying on diagonal lines. Under a
Theiler window the numerator naturally excludes the band, because lines are
not counted there. The denominator is a decision, and the old draft made it
silently: it counted all recurrent points including the excluded band, biasing
DET and LAM downwards by an unreported amount.

`denominator="theiler_corrected"` (default) counts only points outside the
band, so numerator and denominator describe the same region.
`denominator="all"` reproduces the older convention for comparison with
published values that used it. Measured on Lorenz: DET 0.99988 against
0.99328, LAM 0.99760 against 0.99102. Small, systematic, and previously
invisible.

The choice is recorded on every result row.

**Clarified.** `"all"` restores the line of identity, whose cells are recurrent
by definition. Cells inside a wider Theiler band were never evaluated and
cannot be restored, so with `theiler > 1` the convention means "plus the LOI",
not "plus the band". The docstring now says so.

---

<a id="dd-59"></a>
## DD-59 · Entropy needs its normalisation stated

**Status:** accepted

ENTR is the Shannon entropy of the diagonal line-length distribution. Its
maximum is the logarithm of the number of distinct lengths present, which
grows with record length, so raw ENTR is **not comparable between recordings
of different size**.

Both are reported: `ENTR` raw, for comparison with the literature, and
`ENTR_norm` divided by that maximum.

**The normalisation does not remove the dependence — it reverses it.** Measured
across window lengths of 1, 2, 4 and 8 seconds at a fixed 500 Hz, on coupled
signals at α = 0.7 [DD-67]:

| N | 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|
| `ENTR` | 3.483 | 3.663 | 3.719 | 3.754 |
| `ENTR_norm` | 0.876 | 0.823 | 0.776 | 0.743 |

Raw entropy drifts up by 8%, the normalised version down by 15%. Dividing by
`log(number of distinct lengths)` overcorrects, because the count of distinct
lengths grows faster than the entropy does.

So `ENTR_norm` is **not** the N-independent version of `ENTR`, and the earlier
wording implying it was is withdrawn. Both require a matched point count, and
[DD-43] is right to demand one for the whole entropy family rather than only
for the raw member. The normalised form is still useful — it is bounded in
[0, 1] and easier to read — but it is not a fix for the comparability
problem.

---

<a id="dd-60"></a>
## DD-60 · RQA does not beat the geometric index yet

**Status:** provisional — a first measurement, on synthetic data

The project's aim is to relate RQA metrics to classical coupling indices and,
in the end, to propose a coupling index built on them. The first fair
comparison is now possible: same sixteen subjects, same window grid, same
target rate, RQA on the PAC space against the joint independence ratio
[DD-54].

| metric | mean separation | windows surviving Bonferroni |
|---|---|---|
| **independence_ratio** | **0.966** | **4 of 10** |
| RTE | 0.882 | 0 |
| L | 0.868 | 0 |
| ENTR | 0.866 | 0 |
| ENTR_norm | 0.853 | 1 |
| DET | 0.809 | 0 |
| Lmax | 0.764 | 0 |
| TT | 0.702 | 0 |
| W | 0.688 | 0 |
| LAM | 0.686 | 0 |

**The geometric index wins**, and by enough to matter: it is the only metric
with more than one window surviving correction for ten tests.

Three things this does not mean.

It does not mean RQA is the wrong tool. All ten metrics separate the groups
well above chance, and this is one corpus of sixteen synthetic subjects with
one modality at one noise level. Sixteen subjects with ten overlapping windows
is a small, correlated sample and the Bonferroni column is conservative.

It does not mean the two are alternatives. The independence ratio is computed
from a *joint* recurrence structure across two subsystems; the RQA metrics
here come from a single recurrence plot of the joint space. They see different
objects. RQA of the joint recurrence plot has not been measured, and is the
obvious next comparison.

It does not settle which RQA metric to prefer. RTE and L lead here, DET and
LAM trail, which is worth noting because DET is the metric the field reaches
for first. Whether that ordering survives other modalities, noise levels and
real data is exactly what the project has to establish.

**What it does establish** is that the comparison is now mechanised: adding a
metric to the table and re-running `rank_metrics` is one line, and the ordering
falls out with effect sizes and multiplicity accounted for.


---

<a id="dd-61"></a>
## DD-61 · The scaling region is located, not guessed

**Status:** accepted

D2 and the Lyapunov exponent are both slopes of a curve that is straight over
only part of its range. Below the straight part the curve is dominated by
noise and by the finite number of pairs; above it, by the finite size of the
attractor. The old draft took a fixed percentile band -- the 5th to 50th
percentile of the distance distribution -- which is a guess that happens to be
reasonable for some attractors and silently wrong for others.

`find_scaling_region` takes the window whose local slopes vary least, with
width as a tie-breaker, and returns it with the estimate: where it was, how
many points, R², how much the slope varied inside it. A reader can see what
was fitted instead of trusting that something sensible was.

**The width bonus must be relative.** The first version scored a window as
`variation − 0.05 × width`, with width in the units of the x axis. That is
log-radius for a correlation sum, spanning about 2, and plain sample counts
for a divergence curve, spanning 200. On the second the bonus dwarfed the
flatness term and swallowed the whole curve, saturation included. Normalising
the width by the span of the curve fixes it.

Measured against reference values, on 12 000 points:

| system | D2 measured | reference | error |
|---|---|---|---|
| Lorenz | 2.012 | 2.05 | −1.8% |
| Rossler | 1.766 | 1.81 | −2.4% |

**A note on which dimension.** These are references for the **correlation**
dimension (Grassberger and Procaccia 1983), not the Kaplan-Yorke dimension
frequently quoted beside it: 2.06 for Lorenz and 2.01 for Rossler. The two are
different quantities and this estimator will not reproduce the second. An
early draft of these tests used 2.01 for Rossler and the 12% "error" was
entirely the wrong reference.

---

<a id="dd-62"></a>
## DD-62 · Rates carry their units

**Status:** accepted

The Lyapunov exponent and K2 are rates: nats per unit time. Whether that unit
is a sample or a second depends on the sampling rate. The old draft returned a
bare number *and* decimated the data inside the estimator, so what came back
was nats per decimated sample and two windows of different length were not
comparable.

Every rate here is an `Invariant` carrying `units`, the sampling rate it was
computed at, the method, and the fitted region. `units="per_second"` is the
default and multiplies by the rate the state space itself reports — which is
correct after decimation because `subsample` now updates it [DD-47].

---

<a id="dd-63"></a>
## DD-63 · Noiseless data breaks the Lyapunov estimator

**Status:** accepted · **found by a test that failed for the right reason**

A test asserted that a pure sine has no positive Lyapunov exponent. It failed:
the estimator returned 8.09 per second. The cause is worth recording.

On a sampled circle with no noise, the nearest neighbour of a point outside
the Theiler window sits at a separation of **1.3 × 10⁻¹⁵** — machine
precision. The divergence curve then tracks the growth of floating-point error
through a logarithm, rising from −34 to −33 in log units, and a slope fitted to
that is meaningless. The separations involved are 10⁻¹⁵ against an attractor
two units across.

Adding realistic noise does not rescue it either: at 40 dB the exponent comes
out −0.37, correctly negative for a limit cycle, but at 30 and 20 dB it rises
to 12 and 15 as the noise itself drives the separation.

**So the estimator cannot distinguish a limit cycle from chaos on these
signals, and pretending otherwise with a lenient tolerance would have hidden
that.** What it can do is recover the exponent of a genuine chaotic attractor:
Lorenz 0.89 against 0.906, unflagged.

The response is a guard, not a fix. The initial separation is compared against
the extent of the attractor, and a ratio below 10⁻⁸ produces a warning naming
both numbers. The tests now check that the warning fires on noiseless data and
does not fire on Lorenz, which is an honest statement of what the estimator
knows about itself.

A second guard covers the curve being too short to contain a linear region.
On Lorenz the transient runs to about step 70 with a local slope near 1.7, and
the linear region follows at 0.95; `max_steps=80` fits the transient and
returns 1.8. That case is now flagged rather than returned quietly.

**Accuracy, stated plainly.** Across settings that do not trigger a warning,
Lorenz λ₁ lands between 0.86 and 0.99 against a reference of 0.906 — a scatter
of about 15%. That is normal for a Lyapunov estimator on twelve thousand
points and it is what a user should expect. D2 is considerably better behaved,
within 3%.


---

<a id="dd-64"></a>
## DD-64 · The harmonic artefact did not exist

**Status:** accepted

`generate_cfc(harmonic_contamination=...)` claimed to produce a non-sinusoidal
slow rhythm whose harmonics fake phase-amplitude coupling. It is the artefact
Aru et al. (2015) name as the principal source of spurious PAC, and the
generator advertised it from the first phase.

**It produced nothing.** Measured across contamination from 0 to 8, the
modulation index moved from 0.00023 to 0.00036, against 0.00241 for a genuine
α = 0.3. Every adverse-scenario test built on it was testing nothing.

The cause is instructive. The line was

```python
x += contamination * sign(slow) * abs(slow) ** 0.5
```

which distorts the *waveform* of the slow rhythm. But `slow` here is not a
sinusoid: it is band-limited coloured noise with a fluctuating amplitude and
phase. Distorting that produces harmonics smeared across a broad band rather
than placed at multiples of the slow frequency, and their envelope is not
locked to the slow phase — so no phase-amplitude index sees anything. Applied
to a pure sinusoid the same line works, which is presumably how it came to be
believed.

The correction applies the distortion to the **phase**, which is what a
non-sinusoidal oscillation physically is: a sharp pulse at the slow phase,
scaled by the slow envelope.

```python
pulse = exp(harmonic_sharpness * cos(phi_low))
x += contamination * env_slow * normalised(pulse)
```

Measured with `harmonic_sharpness=30`:

| contamination | MVL | MI | comparable to |
|---|---|---|---|
| 0.0 | 0.0248 | 0.00023 | no coupling |
| 0.5 | 0.0367 | 0.00057 | genuine α ≈ 0.2 |
| 1.0 | 0.0746 | 0.00269 | genuine α ≈ 0.35 |
| 2.0 | 0.1787 | 0.01496 | genuine α ≈ 0.65 |

The generator now produces an artefact indistinguishable from real coupling by
the classical indices, which is what makes [DD-65] answerable.

---

<a id="dd-65"></a>
## DD-65 · Recurrence measures do carry information about the harmonic artefact

**Status:** accepted · **and the analysis nearly went wrong four times**

### The question

The hope behind a nonlinear approach to cross-frequency coupling is that it
sees something the classical indices miss. The sharpest test available is the
harmonic artefact: a signal with **no coupling** whose non-sinusoidal slow
rhythm produces harmonics in the high band, which every phase-amplitude index
reports as coupling [DD-64].

If a recurrence measure distinguishes that from genuine coupling, it is a
concrete advantage of the method. If not, the framework inherits the artefact.

### The design

Matching pairs by hand failed: the modulation index still separated the groups
at AUC 1.00, so any difference could have been coupling strength rather than
mechanism. Instead the classical index enters as a **covariate**. For each
metric, fit

```
metric ~ 1 + log(MI) + is_genuine
```

over the range of MI both groups cover, and test the group term. That asks
whether there is information beyond what MI already carries.

### The result

120 signals, **one unique seed each**, five coupling levels against five
contamination levels, 101 in the overlapping MI range:

| metric | effect | 95% CI | t | p (Bonferroni, 8 tests) | direction |
|---|---|---|---|---|---|
| **DET** | +0.0060 | [0.0024, 0.0095] | 3.31 | 0.011 | higher in genuine |
| **TT** | −0.0141 | [−0.0233, −0.0048] | −3.01 | 0.026 | higher in artefact |
| **ENTR** | +0.0176 | [0.0055, 0.0297] | 2.89 | 0.038 | higher in genuine |
| **L** | +0.0385 | [0.0117, 0.0654] | 2.84 | 0.043 | higher in genuine |
| Lmax | −0.615 | [−1.95, 0.72] | −0.91 | 1.00 | — |
| LAM | −0.0005 | [−0.0047, 0.0038] | −0.22 | 1.00 | — |
| Vmax | −0.072 | [−0.53, 0.38] | −0.31 | 1.00 | — |
| RTE | +0.0027 | [−0.0008, 0.0063] | 1.53 | 1.00 | — |

**Genuine coupling gives higher determinism, longer diagonal lines and higher
line entropy than a harmonic artefact of the same classical strength; the
artefact gives longer trapping times.** The diagonal family carries the
signal, the vertical family carries the opposite sign, and the effects are
modest -- DET differs by 0.006 on a scale where DET itself is about 0.93.

### How the analysis nearly went wrong, four times

This is recorded because the sequence is instructive and because the wrong
answer was nearly written into this document.

**First**, hand-matched pairs. The classical indices still separated the groups
at AUC 1.00, so the comparison was about coupling strength. Replaced by the
covariate design.

**Second**, an exploratory run of 49 signals put trapping time at t = −3.40,
p = 0.0014, surviving Bonferroni, with a plausible mechanism. It looked like a
finding.

**Third**, a "confirmation" run of 82 signals reported **nothing** -- DET at
t = 0.12 -- and TT reversed sign. On that basis this decision was written up as
a negative result, prominently, with three paragraphs on what it meant for the
applied papers.

**Fourth**, a test asserting that absence failed, flagging L. Chasing that
revealed the error: the confirmation run used **seven seeds reused across seven
coupling levels**. Forty-nine rows, seven independent draws.
Pseudo-replication, treated by the regression as independent observations. With
the identical protocol and twelve seeds instead of seven, DET went from
t = 0.12 to t = 4.42.

The definitive run gives every signal its own seed, which is what should have
been done from the start.

**The lesson is not about recurrence analysis.** A null result from an
underpowered, pseudo-replicated design is not evidence of absence, and it is
seductive precisely because a negative finding feels rigorous and publishable.
Three of the four wrong turns produced a *cleaner-looking* answer than the
right one.

### What the effect probably is, and what it is not

The artefact signal is not "the same signal plus fake coupling": its slow
rhythm has a genuinely different waveform and its spectrum has extra harmonic
content. So the discrimination is most likely **detection of
non-sinusoidality**, not detection of a coupling mechanism.

That is still the useful outcome -- non-sinusoidality is precisely what has to
be detected -- but it should be described accurately. The claim supported here
is: *at equal modulation index, RQA of the joint phase-amplitude space carries
information about whether the high-band energy comes from an independent
oscillation or from harmonics of the low-band rhythm.* The claim **not**
supported is that recurrence measures are immune to the artefact; the effects
are small and four of eight metrics show nothing.

### What has not been tested

One sharpness, one noise level, one band pair, one state-space route, 25-second
records. Not tested: the hybrid and delay routes, cross and joint recurrence
against a separately recorded channel, the dynamical invariants, and whether
the effect survives at the noise levels of real recordings. Each is a concrete
next experiment.


---

<a id="dd-66"></a>
## DD-66 · The classical indices win on detection, and DET is the worst performer

**Status:** accepted · **the result that should shape what the papers claim**

### The question

How weak can the coupling be, and how noisy the record, before each measure
stops telling a coupled signal from a matched control? Without that curve, a
null result on real data cannot be read: it might mean there is no coupling, or
that the method does not reach that far.

### The design

Five coupling strengths crossed with four noise levels. At each cell, the
coupled signals are compared against a control generated at the **same SNR**
with zero coupling, and the separation is the folded AUC — the probability that
a random coupled value exceeds a random control one, with 0.5 as chance. Nine
seeds per cell, all 216 seeds distinct [DD-65].

### The detection floor

The lowest coupling strength reaching AUC ≥ 0.90:

| metric | +20 dB | +10 dB | 0 dB | −5 dB |
|---|---|---|---|---|
| **MVL** | **0.30** | **0.30** | **0.30** | **0.30** |
| **MI** | **0.30** | **0.30** | **0.30** | **0.30** |
| independence_ratio | 0.50 | 0.30 | 0.50 | 0.30 |
| LAM | 0.50 | 0.50 | 0.50 | 0.70 |
| TT | 0.50 | 0.50 | 0.50 | 0.70 |
| RTE | 0.70 | 0.70 | 0.70 | 0.70 |
| ENTR | 0.70 | 0.70 | 0.70 | 0.90 |
| L | 0.70 | 0.70 | 0.90 | 0.90 |
| **DET** | **0.90** | **0.70** | **never** | **never** |
| Lmax | never | never | never | never |

### Three findings, in order of how much they matter

**The classical indices are the most sensitive, and remarkably robust.** MVL
and the modulation index detect coupling at α = 0.30 at every noise level
tested, including −5 dB. No recurrence measure matches that consistently.

**DET is the worst performer of the ten.** It needs α = 0.90 at 20 dB, and
below 10 dB it never reaches AUC 0.90 at any coupling strength tested. This is
the metric the field reaches for first, and the one most often reported as
evidence of coupling. On this benchmark it is close to useless for detection.

**The vertical-line family outperforms the diagonal family.** LAM and TT reach
0.50, against 0.70–0.90 for ENTR and L and 0.90 for DET. That inverts the
usual emphasis, and it also inverts the direction of [DD-65], where the
*diagonal* family carried the artefact information. The two families are doing
different work.

The joint independence ratio is the best of the recurrence measures and is
competitive with the classical indices, consistent with [DD-54] and [DD-60].

### What this means for the project

**The framework should not be sold on sensitivity.** It is less sensitive than
a modulation index that takes a hundredth of the computation. Any claim that
recurrence analysis detects weaker coupling is not supported by this benchmark
and should not be made.

**What it is for is characterisation.** Put beside [DD-65], a coherent story
emerges: use the classical indices to *detect* coupling, and the recurrence
measures to *characterise* it — whether it is genuine or a waveform artefact,
how it is distributed in time [DD-54], how the joint geometry is constrained
[DD-02]. Those are questions a modulation index cannot answer at all.

**For the applied papers this sets the reporting floor.** With records of 25 s
at 500 Hz and nine subjects per group, α below roughly 0.3 is not detectable by
anything here, and α below 0.5 is not detectable by most of the recurrence
measures. A null result in that range says nothing about the biology.

**For the methodological paper this is the central table.** A detection surface
for ten measures against coupling strength and noise, with matched controls and
distinct seeds, is exactly what the literature lacks — and the fact that DET
comes last is the kind of result that is useful precisely because nobody
expects it.

### What has not been tested

One record length (25 s), one band pair, one state-space route, one modality,
nine seeds per cell. The AUC granularity at nine against nine is 0.012 and the
surface is visibly noisy — a cell at α = 0.15 fluctuating between 0.51 and 0.79
across noise levels is sampling error, not structure. Record length is
experiment 3; the other modalities are experiment 4.


---

<a id="dd-67"></a>
## DD-67 · Only three metrics are stable against window length

**Status:** accepted

### The question

The comparability layer demands a matched point count for several metric
families [DD-43], but nothing said how short a window may be before a metric
stops meaning anything, or which metrics depend on N at all. That is the
parameter every applied analysis has to justify.

### The design

Six coupled subjects at α = 0.7 and six controls, 60 s each at 500 Hz, **no
decimation**, so window length in seconds and point count are the same
variable. Non-overlapping windows of 1, 2, 4 and 8 s, giving N = 500, 1000,
2000, 4000. Five windows per subject per length. Three quantities measured:
whether the value drifts with N, how much it varies between windows of one
subject, and whether it still separates coupled from control.

### Bias: which metrics depend on N

Mean value on coupled subjects:

| metric | N=500 | N=1000 | N=2000 | N=4000 | drift |
|---|---|---|---|---|---|
| RR | 0.0498 | 0.0501 | 0.0500 | 0.0501 | **none** |
| DET | 0.9992 | 0.9993 | 0.9994 | 0.9993 | **none** |
| LAM | 0.9959 | 0.9980 | 0.9986 | 0.9988 | **none** |
| ENTR | 3.483 | 3.663 | 3.719 | 3.754 | +8% |
| RTE | 0.879 | 0.844 | 0.817 | 0.789 | −10% |
| L | 20.95 | 19.74 | 19.05 | 18.78 | −10% |
| ENTR_norm | 0.876 | 0.823 | 0.776 | 0.743 | −15% |
| TT | 6.14 | 7.53 | 8.35 | 8.80 | +44% |
| **Lmax** | **480** | **926** | **1845** | **3065** | **proportional to N** |

**Only RR, DET and LAM are usable across different window lengths.** Lmax is
bounded by N and scales with it, so comparing it between records of different
length is meaningless rather than merely noisy. TT drifts by 44%. And
`ENTR_norm` drifts *more* than the raw entropy and in the opposite direction,
which is a correction to [DD-59].

### Precision: variation between windows of one subject

Coefficient of variation, averaged over subjects:

| metric | 1 s | 2 s | 4 s | 8 s |
|---|---|---|---|---|
| DET, LAM | ≤0.003 | ≤0.001 | 0.000 | 0.000 |
| RR | 0.008 | 0.010 | 0.010 | 0.011 |
| ENTR, ENTR_norm, RTE | 0.02–0.05 | 0.02–0.04 | 0.03 | 0.02 |
| TT | 0.072 | 0.067 | 0.055 | 0.030 |
| L | 0.147 | 0.092 | 0.080 | 0.051 |
| **Lmax** | **0.088** | **0.129** | **0.159** | **0.261** |

Everything improves with longer windows except Lmax, which gets *worse*: it is
an extreme-value statistic, so a longer window gives it more chances to catch
an outlier.

### Detection: AUC coupled against control

| metric | 1 s | 2 s | 4 s | 8 s |
|---|---|---|---|---|
| **TT** | **1.000** | 1.000 | 1.000 | 1.000 |
| LAM | 0.972 | 1.000 | 1.000 | 1.000 |
| ENTR_norm | 0.944 | 1.000 | 0.944 | 1.000 |
| ENTR | 0.917 | 1.000 | 0.972 | 1.000 |
| RTE | 0.833 | 0.972 | 1.000 | 1.000 |
| L | 0.556 | 0.917 | 0.972 | 1.000 |
| **DET** | 0.667 | 0.750 | 0.750 | **0.556** |
| Lmax | 0.569 | 0.667 | 0.583 | 0.722 |
| RR | 0.528 | 0.556 | 0.639 | 0.528 |

**One-second windows are enough** for trapping time and laminarity at α = 0.7:
500 points, five windows per subject, six subjects per group, AUC 1.000 and
0.972. That is a far shorter window than the field typically uses, and it sets
the temporal resolution of the method at roughly one cycle of the modulating
rhythm rather than tens.

RR does not separate anything, as it must not: it is pinned to the target rate
by construction [DD-40]. Its appearance here is a check that the threshold
machinery is doing what it claims.

**DET fails again**, and gets *worse* with longer windows. Together with
[DD-66], where it had the highest detection floor of ten measures, the case
against using DET as a coupling index on this kind of data is now made twice
over from independent directions.

### Practical recommendation

| purpose | metric | minimum window |
|---|---|---|
| detection, time-resolved | TT, LAM | 1 s (500 points) |
| detection, robust | ENTR, RTE | 2 s |
| comparison across records of unequal length | RR, DET, LAM only | any |
| never compare across unequal N | Lmax, TT, ENTR, ENTR_norm, L, RTE | — |

The last row is the one that matters for the applied papers: six of nine
metrics cannot be compared between records of different length, and the
comparability layer withholds them automatically when N differs [DD-43].

### What has not been tested

One coupling strength, one noise level, one band pair, five windows per
subject. Windows shorter than 1 s were not tried, and the interesting question
of whether 0.5 s still works is open — at 6 Hz that is three cycles of the
modulating rhythm, which is probably the floor on physical grounds rather than
statistical ones.

### Re-verified

This section was found duplicated under a shared number and its central claim
was re-measured before being kept: six coupled subjects, α = 0.7, no
decimation, five windows each at N = 500, 1000, 2000, 4000.

| metric | drift over the range |
|---|---|
| RR | 0.3% |
| DET | 0.0% |
| LAM | 0.2% |
| ENTR | +7.1% |
| RTE | −11.1% |
| L | −12.6% |
| ENTR_norm | −15.4% |
| TT | +41.4% |
| **Lmax** | **+530%, and Lmax/N is constant** |

The reproduction is close: TT was recorded at +44% and measures +41%, and Lmax
is confirmed proportional to N. The claim stands.

---

<a id="dd-68"></a>
## DD-68 · Window length with a fixed sampling rate

**Status:** accepted

### The question

A windowed analysis has to choose a window length, and that choice is usually
made by convention. How short can a window be before each measure stops
separating a coupled signal from a control? The answer sets the temporal
resolution the framework can honestly claim.

### The design

One 45-second record per seed, windows of increasing length taken **from the
same record**, so length is the only thing that varies. The state space is
decimated to a fixed 100 Hz first, so the number of points is proportional to
the duration and the effective sampling rate is identical at every length
[DD-47]. Coupling fixed at α = 0.7 and 20 dB, comfortably above the detection
floor of [DD-66], so what is being measured is length and not sensitivity.
Nine records per group.

### Shortest window reaching AUC ≥ 0.90

| measure | window | points | theta cycles |
|---|---|---|---|
| **MVL** | **1 s** | 100 | 6 |
| **MI** | **1 s** | 100 | 6 |
| **LAM** | **2 s** | 200 | 12 |
| **independence_ratio** | **2 s** | 200 | 12 |
| TT | 4 s | 400 | 24 |
| ENTR | 8 s | 800 | 48 |
| DET | 16 s | 1600 | 96 |
| L | 16 s | 1600 | 96 |
| RTE | 16 s | 1600 | 96 |
| Lmax | never | — | — |

**A factor of eight separates the recurrence measures from each other.** LAM
and the joint independence ratio work at two seconds; DET, L and RTE need
sixteen. Anyone windowing at four seconds — a common choice — is using DET well
below the length at which it functions, and LAM comfortably above it.

The classical indices work at one second, twelve theta cycles fewer than DET
needs. That is consistent with [DD-66]: they are simply more efficient
detectors.

### Value drift with length

The same table read differently: how much does each measure change with window
length, holding the signal fixed? Ratio of the 32-second value to the
1-second value, coupled group:

| measure | ratio | comparable across lengths? |
|---|---|---|
| DET | 1.00 | yes |
| LAM | 0.99 | yes |
| L | 0.98 | yes |
| MVL | 0.97 | yes |
| TT | 0.91 | roughly |
| independence_ratio | 1.08 | roughly |
| ENTR | 1.12 | no |
| MI | 0.83 | no |
| RTE | 0.73 | no |
| **Lmax** | **2.81** | **no** |

DET, LAM and L are essentially length-invariant, which is what makes them
usable across records of different length once N is matched [DD-43]. `Lmax`
nearly triples: it is the longest line found, so it can only grow with the
record, and comparing it between windows of different length is meaningless.
`RTE` falls by a quarter and `ENTR` rises by a tenth, both because their
underlying distributions gain distinct values as the record lengthens [DD-59].

That vindicates the comparability layer's requirement of a matched point count
for the length and entropy families, and shows the ratio family really is
exempt.

### Practical recommendation

**Four seconds is a poor default and eight is a defensible one**, at these
parameters. At four seconds LAM, TT and the independence ratio work and DET, L,
RTE and ENTR do not; at eight, only DET, L and RTE are still short.

If the analysis needs DET specifically, the window must be sixteen seconds,
which caps the temporal resolution at roughly that — and a transient shorter
than the window is smeared over it, as [DD-54] showed.

If the analysis needs fine temporal resolution, use LAM or the independence
ratio and say so, rather than reporting DET from windows too short to support
it.

### What has not been tested

One coupling strength, one noise level, one band pair, one effective sampling
rate, nine records per group. The floor will move with all of them, and in
particular a weaker coupling will need longer windows than this. The
interaction between window length and coupling strength is a two-dimensional
surface and only one slice of it has been measured.

### Re-verified, and partly corrected

This section was found duplicated under a shared number, so its claims were
re-measured with the same protocol: state space decimated to 100 Hz, α = 0.7 at
20 dB, nine records per group, one window per record.

| measure | recorded here | re-measured |
|---|---|---|
| MVL | 1 s | **1 s** |
| MI | 1 s | **1 s** |
| LAM | 2 s | **4 s** |
| TT | 4 s | **4 s** |
| ENTR | 8 s | **4 s** |
| DET | 16 s | **8 s** |

**The ordering reproduces exactly** -- classical indices first, then the
vertical-line family, then ENTR, with DET last -- and that ordering is what
the section is for. The individual thresholds move by one grid step in either
direction, which is what nine records per group and an AUC granularity of 1/81
will do. The table above should be read as an ordering with a resolution of
about one doubling, not as six precise numbers.

---

<a id="dd-69"></a>
## DD-69 · Many short windows beat few long ones

**Status:** accepted

### The question

How short can a window be before a measure stops seeing the coupling? The
answer sets the time resolution of every windowed analysis, and it is a
parameter that has to be justified in any application.

### Two questions, deliberately separated

A first run averaged each metric over all the windows inside a 60-second
record and compared subjects. It gave a startling answer -- five measures
detecting at 0.5 s, three theta cycles -- and it was confounded: a 0.5 s window
grid yields 117 windows to average and an 8 s grid only 7. That measures the
benefit of averaging, not what a window sees.

So the two were separated.

### One window, α = 0.7 at 20 dB, 12 subjects per group

AUC against a matched control:

| metric | 0.5 s | 1 s | 2 s | 4 s | 8 s |
|---|---|---|---|---|---|
| | 250 pts | 500 | 1000 | 2000 | 4000 |
| LAM | 0.764 | 0.896 | 0.875 | **0.917** | **1.000** |
| TT | 0.715 | 0.854 | 0.722 | **0.979** | **1.000** |
| RTE | 0.806 | 0.854 | 0.715 | **0.979** | **0.917** |
| independence_ratio | 0.514 | 0.889 | 0.861 | **0.931** | **1.000** |
| ENTR | 0.569 | 0.729 | 0.792 | **0.944** | **0.917** |
| L | 0.708 | 0.576 | 0.701 | 0.882 | 0.868 |
| Lmax | 0.542 | 0.576 | 0.556 | 0.514 | 0.778 |
| DET | 0.535 | 0.632 | 0.556 | 0.653 | 0.604 |

**A single window needs about 4 seconds — 2000 points, 24 cycles of the slow
rhythm — before anything reaches AUC 0.90.** Below that the metrics carry
information (0.7 to 0.9) but not enough to classify a subject.

### The whole record, cut finely and averaged

With the same total data and metrics averaged across windows:

| metric | 0.5 s (117 windows) | 1 s (58) | 2 s (29) | 4 s (14) | 8 s (7) |
|---|---|---|---|---|---|
| ENTR, LAM, TT, RTE | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| independence_ratio | **0.953** | 1.000 | 1.000 | 1.000 | 1.000 |
| L | 0.625 | 0.812 | **1.000** | 1.000 | 1.000 |
| DET | 0.609 | 0.609 | 0.719 | 0.641 | 0.594 |
| Lmax | 0.578 | 0.734 | 0.516 | 0.609 | 0.719 |

**Cutting a fixed record into many short windows and averaging beats using few
long ones**, down to 0.5 s and three theta cycles. Splitting finer costs
per-window reliability and buys more independent estimates, and for a
subject-level comparison the second wins.

### The practical rule

- **Subject-level classification**, where the whole record is available:
  short windows, averaged. Half a second works; there is no gain past 2 s.
- **Time-resolved analysis**, where each window must stand alone: at least
  4 seconds, roughly 24 cycles of the slow rhythm. That is the real cost of
  time resolution, and it is much larger than the window grids used in [DD-54],
  where detection came from a *series* of windows rather than from any one.

Note that the two are not in conflict. The transient detection of [DD-54] used
4 s windows at 75% overlap and worked, because it read the shape of the series
rather than classifying a single window.

### Consistency with the other findings

DD-68 measures the same floor at a decimated 100 Hz and reaches the same
place by a different route: 4 s for the vertical-line family, DET last.


DET never reaches 0.90 at any window length under either analysis, matching
[DD-66] where it was the worst of the ten measures. The vertical-line family
leads again, as it did there. Two independent experiments now point the same
way about which metrics are worth reporting.

### What has not been tested

One coupling strength, one noise level, one modality, one band pair. The
interaction that matters most for real data is untested: whether the 4-second
floor for a single window rises as the SNR falls. It probably does, and
experiment 2's surface suggests by how much, but the two grids were not
crossed.


---

<a id="dd-70"></a>
## DD-70 · The other three modalities, and where recurrence wins

**Status:** accepted

### The question

The library offers four coupling modalities and, until now, had evidence for
one. Phase-phase coupling, amplitude-amplitude coupling and the trivariate
phase-phase-amplitude case each have a generator, a state space and a natural
recurrence structure, and none had been tested end to end.

### The design

Each modality against a matched uncoupled control at 20 dB, three coupling
strengths, ten records per cell, 30 s each. Each is given its own classical
index as the baseline and its own structure:

| modality | classical baseline | recurrence structure |
|---|---|---|
| PPC | PLV at 10:1 | cross recurrence between two phase circles, plus RQA of the joint 4-D space |
| AAC | rank correlation of the envelopes | RQA of the 2-D envelope space, plus a joint plot of the two envelopes |
| PPA | modulation index, slow phase against fast envelope | RQA of the 5-D space, plus a joint plot of its three blocks |

### Phase-phase coupling: recurrence clearly wins

AUC against control:

| measure | α=0.3 | α=0.6 | α=0.9 |
|---|---|---|---|
| **L, joint space** | **1.000** | **1.000** | **1.000** |
| **L, cross plot** | **1.000** | 0.990 | **1.000** |
| **ENTR, cross plot** | **1.000** | 0.990 | **1.000** |
| ENTR, joint space | 0.950 | 0.990 | 0.990 |
| PLV (classical) | 0.770 | 0.800 | 0.980 |
| DET, cross plot | 0.570 | 0.760 | 0.810 |
| independence ratio | 0.620 | 0.560 | 0.510 |

**This is the first case where a recurrence measure beats the classical index,
and it does so decisively.** Mean diagonal line length separates perfectly at
every coupling strength, including α = 0.3 where PLV reaches only 0.77.

The reason is visible in the raw numbers: after band-pass filtering at 50-70 Hz
the measured PLV falls from about 0.80 to 0.12, because the filtered gamma
phase mixes the locked component with the free-running one. PLV asks whether a
single phase difference is constant, and the filter destroys that. The diagonal
line length asks whether the two trajectories stay close *for a while*, which
survives.

Note also that the joint 4-D space does as well as the cross plot, so the
result is about the geometry rather than about the cross construction.

### Amplitude-amplitude coupling: the classical index is fine

| measure | α=0.3 | α=0.6 | α=0.9 |
|---|---|---|---|
| LAM | 0.860 | 0.760 | **1.000** |
| independence ratio | 0.740 | **1.000** | **1.000** |
| rank correlation (classical) | 0.530 | **1.000** | **1.000** |
| DET | 0.780 | 0.710 | 0.990 |

Everything works above α = 0.6 and nothing works at 0.3. The joint independence
ratio matches the classical index and LAM is marginally ahead at the weakest
level, but the differences are within what ten records per group can resolve.
There is no case here for preferring the recurrence route.

### Phase-phase-amplitude: recurrence is the only thing that works

| measure | α=0.3 | α=0.6 | α=0.9 |
|---|---|---|---|
| LAM | 0.520 | **0.990** | **1.000** |
| L | 0.600 | **0.950** | **1.000** |
| ENTR | 0.610 | **0.950** | **1.000** |
| TT | 0.550 | 0.890 | **1.000** |
| DET | 0.530 | 0.820 | **1.000** |
| modulation index (classical) | 0.630 | 0.540 | 0.610 |
| independence ratio | 0.620 | 0.570 | 0.520 |

**The classical index fails completely** -- it never exceeds 0.63 -- while five
RQA metrics separate perfectly at α = 0.9 and four reach 0.89 or better at 0.6.

That is expected and is the point of the modality: in PPA the fast envelope is
modulated by the *sum* of two slow phases, so a bivariate index between one of
them and the envelope sees almost nothing. The 5-D joint space contains all
three components at once, and RQA of it sees the constraint.

The joint independence ratio also fails here, which is informative: splitting
the space into three blocks and asking whether they recur together is not the
same as asking whether the joint trajectory is constrained.

### What this changes

Three of the four modalities now have end-to-end evidence, and the picture from
[DD-66] needs qualifying. It is true that for **PAC** the classical indices are
more sensitive and the recurrence measures are for characterisation. It is not
true in general:

- **PAC**: classical indices win on detection [DD-66].
- **PPC**: recurrence wins clearly, because filtering destroys what PLV measures.
- **AAC**: a tie; the classical index is simpler and should be preferred.
- **PPA**: only recurrence works, because no bivariate index sees a trivariate
  constraint.

**For the methodological paper this is the argument for the framework.** Its
case does not rest on being more sensitive than a modulation index at the task
that index was designed for. It rests on covering couplings that have no
adequate classical index at all — and on the higher-order case that motivates
F11 tensors.

Note which metrics carry it: **L and ENTR for PPC, LAM and L for PPA**. DET
again trails in both. Three experiments now agree that DET, the metric most
often reported, is among the weakest here.

### What has not been tested

Ten records per cell gives an AUC granularity of 0.01 and no more than a
coarse ordering. One noise level, one record length, one n:m ratio for PPC, one
frequency triplet for PPA. The PPC result in particular deserves a proper
sweep, because a recurrence measure beating a classical index by that margin is
the kind of claim that will be checked.


---

<a id="dd-71"></a>
## DD-71 · Each block contributes its natural scalar

**Status:** accepted

Every complexity measure -- Higuchi, Katz, Petrosian, the entropies, DFA,
Hjorth, Lempel-Ziv -- takes a one-dimensional series, while a state space has
several coordinates. Applying them to a coordinate chosen by position would be
arbitrary and would change meaning silently between routes.

Each **block** therefore contributes its natural scalar, and the columns are
named `measure_block`, so `higuchi_fd_amp_gamma` says what was measured and on
what.

**The phase circle needed a decision.** Three candidates, all measured on the
same recording:

| series | DFA exponent | what it is |
|---|---|---|
| wrapped angle | **0.13** | jumps at ±π that the measures read as structure |
| unwrapped angle | **2.01** | a ramp; the value of a pure trend |
| **increment of the unwrapped angle** | **0.73** | the instantaneous frequency |

Only the third is stationary and carries the phase dynamics rather than an
artefact of representation. It is the instantaneous frequency in radians per
sample, and it is what a phase-circle block contributes.

An amplitude block contributes its amplitude, a delay block its first
coordinate, anything else its first column.

---

<a id="dd-72"></a>
## DD-72 · Cheap by default, invariants on request

**Status:** accepted

Higuchi, Katz, Petrosian, the entropies, DFA, Hjorth and Lempel-Ziv are all
O(N log N) or better and cost milliseconds on a window. D2 and the Lyapunov
exponent build neighbour structures and cost seconds.

`metrics(dynamics=True)` therefore computes only the cheap set, and D2, λ₁ and
K2 are opt-in through `invariants=True`. Asking for dynamical measures over a
thousand windows should not silently start an overnight job.

Measured: 18 windows of a 40-second record, RQA **and** the default dynamical
set, in **1.2 seconds**. Adding K2 takes it to 1.6.

The default scalar set is deliberately compact -- one fractal dimension, one
entropy, one scaling exponent, one complexity measure and two Hjorth
parameters -- because the full set across four blocks would be forty columns
per window. `measures="all"` gives everything.

---

<a id="dd-73"></a>
## DD-73 · The Higuchi fit is anchored at k = 1

**Status:** accepted

Higuchi's construction is a power law only for small k; past that the log-log
curve bends and then falls. Letting the general scaling-region finder [DD-61]
pick the flattest window lands it on the bend.

Measured on a smoothed gamma envelope, local slopes across k:

```
1.04  1.06  1.08  1.17  1.28  1.42  1.59  1.79  2.03  2.29
2.58  2.87  3.12  3.27  3.20  2.85  2.24  1.53  1.13  0.79
```

The finder returned **3.16** -- outside the [1, 2] on which the dimension is
defined, and therefore not a dimension at all.

The fit is now anchored at k = 1 and extended while the local slope stays
within 25% of its initial value. A result still outside the range is returned
**with a warning rather than clipped**, because a clipped number looks like a
measurement and is not one.

After the change: white noise 2.001, pink noise 1.834, a sine 1.050, Lorenz
1.043 — all where they should be.

**The general lesson.** [DD-61] argued that locating the scaling region beats
guessing it, and that stands. But a finder with no knowledge of the quantity
it is fitting will happily return a value the quantity cannot take. Where a
measure has a defined range or a defined fitting convention, that convention
belongs in the estimator, not in a general-purpose search.


---

<a id="dd-74"></a>
## DD-74 · Dynamical measures on the harmonic artefact

**Status:** accepted

[DD-65] tested eight RQA metrics against the harmonic artefact. Now that the
dynamical measures reach the table [DD-71], the same experiment can be run with
thirty-one measures: the same design, the same covariate, one unique seed per
signal, 107 in the overlapping range of the modulation index.

### The ranking

| measure | effect | t | p (Bonferroni, 31 tests) |
|---|---|---|---|
| **hjorth_activity** (gamma envelope) | −0.0338 | **−6.29** | **0.00000** |
| **higuchi_fd** (gamma envelope) | −0.0038 | **−4.52** | **0.00051** |
| hurst / DFA_alpha (envelope) | −0.0608 | −3.09 | 0.079 |
| sample_entropy (envelope) | −0.0062 | −3.03 | 0.095 |
| DET | +0.0041 | 2.52 | 0.414 |
| petrosian_fd (envelope) | −0.0001 | −2.47 | 0.470 |
| permutation_entropy (envelope) | −0.0026 | −2.44 | 0.509 |

**The dynamical measures beat every RQA metric at this task**, and by a
margin: the best of them reaches t = −6.3 where the best RQA metric reaches
2.5.

### The strongest effect is a scale confound, and must be discounted

Hjorth activity is the **variance** of the series it is given, and here that
series is the gamma envelope. The artefact adds harmonic energy inside the
gamma band by construction, so its envelope has more power:

| | mean envelope variance |
|---|---|
| genuine | 0.2718 |
| artefact | 0.2947 (+8.4%) |

That is not a dynamical difference. It is the artefact's extra energy,
detected by a measure of energy. Reporting it as a nonlinear discovery would be
wrong, and it would not survive on real data where the two conditions are not
matched in power by construction.

**Higuchi is the real finding.** It is scale-invariant -- multiplying the
series by a constant shifts log L by a constant and leaves the slope unchanged
-- so the difference it reports is shape. The artefact's gamma envelope is
measurably *rougher* than genuine coupling's at the same modulation index
(1.1118 against 1.1082), which is what one would expect: harmonics of a sharp
slow waveform produce a spikier envelope than a smooth stochastic modulation.

The effect is small in absolute terms and it takes 107 signals to see it.

### A methodological point worth keeping

DD-65 reported four RQA metrics surviving Bonferroni over **eight** tests.
Adding twenty-three dynamical measures leaves them unchanged in effect size and
none of them surviving Bonferroni over **thirty-one**. Nothing about the
measurements changed; only the correction did.

Both statements are true and neither is more honest than the other. What is
dishonest is choosing the family after seeing the results. The family here is
"every measure the library computes", declared before the run, and that is the
number the correction should use — which means DD-65's four metrics should be
read as suggestive rather than established.

### What this suggests for the paper

The measure that most cleanly separates a harmonic artefact from genuine
coupling, once scale confounds are discounted, is a **fractal dimension of the
high-frequency envelope**, not a recurrence metric. That is worth stating
plainly, and it is the kind of cross-family comparison the methodological paper
exists to make.


---

<a id="dd-75"></a>
## DD-75 · Detection with both families, and why combining hurts

**Status:** accepted

[DD-66] measured a detection surface for ten measures. Repeated with all
thirty-six, on the same 216 signals, three things come out.

### The detection floor, both families

Lowest coupling strength reaching AUC ≥ 0.90 against a matched control:

| measure | +20 dB | +10 dB | 0 dB | −5 dB |
|---|---|---|---|---|
| MI, MVL | **0.30** | **0.30** | **0.30** | **0.30** |
| independence_ratio | 0.30 | 0.30 | 0.50 | 0.50 |
| **hjorth (activity, mobility, complexity)** | 0.50 | 0.50 | 0.50 | **0.50** |
| **katz_fd, svd_entropy** | 0.50 | 0.50 | 0.50 | **0.50** |
| LAM | 0.50 | 0.50 | 0.50 | 0.70 |
| TT | 0.50 | 0.50 | 0.70 | 0.70 |
| ENTR, L | 0.70 | 0.50 | 0.90 | never |
| DET | 0.70 | 0.70 | 0.90 | never |
| every phase-block measure | never | never | never | never |

All on the gamma envelope unless stated.

### Three findings

**The dynamical measures are more noise-robust than the RQA metrics.** Five of
them hold a floor of 0.50 all the way to −5 dB, where LAM slips to 0.70, TT and
RTE to 0.70, and DET, ENTR and L fail entirely. At good SNR the two families
are comparable; the difference appears as the noise rises, which is the regime
real recordings live in.

**Every phase-block measure fails at every level.** That is the expected
answer and a good check on the block-wise design of [DD-71]: PAC modulates the
amplitude of the fast rhythm, not the dynamics of the slow phase, so the
instantaneous frequency of theta carries no coupling information. A measure
that had detected something there would have been a warning, not a result.

**The classical indices still win**, as in [DD-66]. Nothing here changes that
for PAC.

### Combining the families does not help. It hurts.

Leave-one-out cross-validated logistic regression, coupled signals at one
coupling strength pooled across all four noise levels against the controls at
the same levels, 36 against 36 with SNR balanced:

| α | classical (2) | RQA (6) | dynamics (13) | RQA+dyn (19) | all three (21) |
|---|---|---|---|---|---|
| 0.15 | **0.824** | 0.424 | 0.384 | 0.481 | 0.776 |
| 0.30 | **0.982** | 0.759 | 0.657 | 0.840 | 0.968 |
| 0.50 | **1.000** | 0.967 | 0.992 | 0.995 | 0.998 |
| 0.70 | **1.000** | 0.996 | 1.000 | 1.000 | 1.000 |

**Two classical features beat every combination at every coupling strength**,
and adding the other nineteen to them makes the classifier *worse*: 0.776
against 0.824 at α = 0.15.

This is not a statement about information content, it is one about estimation.
Twenty-one features on seventy-two samples is a regime where a fitted
combination spends its degrees of freedom on noise. The cross-validation is
what makes the effect visible; an in-sample fit would have shown the opposite
and been meaningless.

**The practical rule.** For detecting PAC, use the modulation index. Do not
build a composite index out of these families hoping to gain sensitivity: on
this benchmark it costs sensitivity. A composite might pay in a regime with
many more subjects, or where the classical index fails outright as it does for
the trivariate case [DD-70], but that has to be demonstrated rather than
assumed.

### Consistency, and what it adds up to

Four experiments now agree on the shape of the answer:

| task | best family |
|---|---|
| detecting PAC | classical |
| detecting PPC | recurrence, decisively [DD-70] |
| detecting AAC | classical, marginally [DD-70] |
| detecting PPA | recurrence, the only thing that works [DD-70] |
| separating a harmonic artefact at equal MI | dynamical, on the envelope [DD-74] |
| robustness to noise at fixed coupling | dynamical [this decision] |

The framework's case is not sensitivity on the task the classical indices were
designed for. It is coverage of couplings they cannot express, discrimination
of artefacts they cannot see, and robustness where they are not the constraint.


---

<a id="dd-76"></a>
## DD-76 · SVD entropy is the short-window measure

**Status:** accepted

[DD-69] found that a single window needs about four seconds before any RQA
metric classifies a subject. Repeated with all thirty-three measures on the same
design -- one window per subject, α = 0.7 at 20 dB, twelve per group -- the
answer changes.

### Shortest single window reaching AUC ≥ 0.90

| measure | 0.5 s | 1 s | 2 s | 4 s | 8 s |
|---|---|---|---|---|---|
| | 250 pts | 500 | 1000 | 2000 | 4000 |
| **svd_entropy** (envelope) | 0.799 | **0.924** | **1.000** | **1.000** | **1.000** |
| LAM | 0.569 | 0.729 | **0.903** | **1.000** | 0.965 |
| hjorth_activity (envelope) | 0.681 | 0.812 | **1.000** | **1.000** | **1.000** |
| TT | 0.514 | 0.847 | 0.840 | **0.972** | **1.000** |
| independence_ratio | 0.562 | 0.618 | 0.854 | **1.000** | **1.000** |
| katz_fd, hjorth_mobility | 0.562 | 0.792 | 0.646 | **0.972** | **1.000** |
| ENTR | 0.556 | 0.792 | 0.861 | 0.833 | **0.931** |
| DET | 0.503 | 0.576 | 0.521 | 0.660 | 0.819 |

**SVD entropy of the gamma envelope halves the window a single-window analysis
needs**: one second against two for the best RQA metric, and it is already at
0.80 with half a second.

Confirmed on independent seeds, fourteen per group:

| measure | 0.5 s | 1 s | 2 s | 4 s |
|---|---|---|---|---|
| **SVD entropy** | 0.857 | **0.918** | **1.000** | **1.000** |
| LAM | 0.520 | 0.684 | **0.949** | 0.949 |
| DET | 0.630 | 0.559 | 0.592 | 0.730 |

### Why it should be the one

SVD entropy is the entropy of the singular spectrum of a delay embedding of the
envelope. It asks how many directions the envelope's trajectory really
occupies, which is a global property of a short segment rather than a count of
structures inside it. DET, L and LAM all need enough points for *lines* to
form, and a 500-point plot at a 5% rate does not have many long ones. A
singular spectrum needs only enough points to estimate a covariance.

That also explains why it survives noise so well [DD-75]: noise raises every
singular value roughly together and moves the entropy less than it moves the
count of diagonal lines.

### Every measure was computable at every length

No measure returned a missing value at 250 points, including DFA, sample
entropy and SVD entropy, which need enough samples to define their scales.
That was worth checking before reading the table: a measure that quietly fails
at short windows would look like a measure that detects nothing.

### What this changes

**For time-resolved analysis the floor moves from four seconds to one**, if the
measure is chosen for the job. That is a factor of four in temporal resolution
and it comes from a measure outside the recurrence family entirely.

The recommendation of [DD-69] should be read as: for a single window, use
**SVD entropy of the modulated envelope**; RQA needs two to four seconds; DET
needs more than eight and should not be used for time-resolved work at all.

### What has not been tested

One coupling strength, one noise level, one modality. The interaction that
matters is whether SVD entropy keeps its advantage as the SNR falls -- [DD-75]
suggests it should, since it held a 0.50 floor to −5 dB, but the two grids were
not crossed. Nor was it tested on the averaged analysis of [DD-69], where many
short windows are pooled.


---

<a id="dd-77"></a>
## DD-77 · A univariate measure cannot be a coupling measure

**Status:** accepted · **the check that stopped a wrong claim**

### What the experiment appeared to show

[DD-70] tested the other three modalities with RQA. Repeated with both
families, phase-phase coupling gave a spectacular-looking result: **nine
dynamical measures separate coupled from control at AUC 1.000 at every coupling
strength**, including α = 0.3 where PLV reaches only 0.71.

| measure | α=0.3 | α=0.6 | α=0.9 |
|---|---|---|---|
| Higuchi, Katz, Petrosian, SVD, permutation, sample, spectral entropy, Lempel-Ziv, Hjorth complexity — **all on the gamma phase block** | **1.000** | **1.000** | **1.000** |
| L, joint space | 1.000 | 0.960 | 1.000 |
| PLV (classical) | 0.710 | 0.920 | 0.990 |

Read at face value: nine cheap univariate measures crush the classical index at
detecting phase-phase coupling.

### Why that reading is wrong

Every one of those nine is computed on the **gamma block alone**. None of them
looks at theta. A quantity that never sees one of the two signals cannot be
measuring the relation between them.

What they are measuring is that the gamma phase is *regular*. Under this
generator a locked phase is a regular one, so regularity and coupling coincide
— but only because the control is a freely running phase. A signal could have a
perfectly regular gamma phase without being coupled to the theta beside it.

### The test that settles it

Pair the gamma phase of a coupled recording with the theta phase of an
**independent** one. The gamma phase is still regular; it is no longer coupled
to the theta it sits next to. A coupling measure should now say no.

| measure | coupled vs control | mismatched vs control |
| | *should separate* | *should not* |
|---|---|---|
| PLV | 0.901 | **0.605** |
| L (joint space) | 1.000 | **0.704** |
| SVD entropy, gamma block | 1.000 | **1.000** |
| permutation entropy, gamma block | 1.000 | **1.000** |

**The univariate measures separate the mismatched pairing exactly as strongly
as the genuine one.** They are regularity detectors. PLV and the joint-space
line length both fall towards chance, which is what a coupling measure must do.

### The rule this establishes

**A measure computed on a single block is not a coupling measure, whatever its
AUC.** It may be a useful feature — a regular gamma phase is worth knowing
about — but it cannot support a claim about coupling, and reporting it beside
PLV in a table of coupling detectors would be misleading.

The measures that survive this test are the ones defined on the **joint**
object: RQA of the joint space or of a cross plot, the joint independence
ratio, and the classical indices. `dynamics_measures` is block-wise by
construction [DD-71], so **every** scalar measure it produces is univariate and
falls under this rule.

### Consequences for the earlier findings

This does not undermine [DD-74] or [DD-75], but it sharpens how they must be
read.

- [DD-74], the harmonic artefact: the discriminating measures were on the
  gamma envelope, univariate. The claim there was about telling two
  *mechanisms* apart at matched coupling strength, not about detecting
  coupling, and that claim stands. But it is a statement about waveform, not
  about a relation.
- [DD-75], noise robustness: same. The dynamical measures hold their detection
  floor better, and what they are detecting is a change in the envelope's own
  character, which under PAC is caused by coupling. In a recording where the
  envelope's character varies for other reasons, they would report coupling
  that is not there.
- [DD-76], SVD entropy at one second: same caveat, and it matters most here,
  because a short window is exactly where a spurious envelope change is most
  likely.

**The joint measures are the ones that can carry a coupling claim.** The
univariate ones are sensitive, cheap and easy to fool, and the library should
present them as what they are.

### What has not been tested

The mismatched-pairing control was run for PPC only, at one coupling strength,
nine per group. The same test should be run for PAC and PPA before any
univariate measure is reported as evidence of coupling in those modalities —
and on this evidence it will give the same answer.


---

<a id="dd-78"></a>
## DD-78 · Most tiles never touch the Theiler band

**Status:** accepted

A full-recording analysis of the stated use case -- eight minutes at 500 Hz,
240 000 points -- was measured at roughly two hours per subject, and profiling
gave the opposite answer to the one expected: the line-length accumulator was
29% of the time and building the tiles was 63%.

Two things were being repeated needlessly.

**The Theiler mask.** Excluding the band costs an `h × w` index comparison per
tile, and almost every tile of a long record lies far from the diagonal where
the band cannot reach. At 240 000 points with 8192-sample tiles, 30 of 900
tiles intersect the band and 870 do not. The mask is now built only where the
index ranges come within the window of each other.

**The squared norms.** Every tile of a row band needs the same `‖a‖²`, and it
was recomputed for each. Now cached.

Together: 30.1 s to 23.3 s on a 15 000-point trajectory, with every existing
test unchanged.

The lesson is narrow and familiar: the bottleneck was not where the design
notes had been arguing about it. Profiling took one command and redirected the
whole exercise.

---

<a id="dd-79"></a>
## DD-79 · Single precision is a choice, not a default

**Status:** accepted

The distance block exists only to be compared against a threshold. Single
precision resolves about seven significant digits, far more than any recurrence
decision needs, and it makes the kernel **3.2 times faster** -- 435 ms against
1373 ms for an 8192-square tile, measured directly. The memory halves too, so a
tile of a given budget can be twice as wide.

Measured end to end on a 15 000-point trajectory:

| precision | RQA time | DET | L | LAM |
|---|---|---|---|---|
| double | 25.2 s | 0.981132 | 23.0752 | 0.986884 |
| single | **18.8 s** | 0.981132 | 23.0754 | 0.986884 |

DET and LAM identical to six figures; L differs in the fifth.

**It is not the default.** A recurrence matrix is the substrate of every metric
downstream, and a user who has not asked for reduced precision should not
receive it. `precision="single"` opts in and the choice is recorded in the
provenance.

The one case where it can change a result is a pair sitting within one part in
10⁷ of the threshold. At a 5% rate on 100 000 points that is a handful of cells
out of 10¹⁰, and their effect is far below the variation between seeds.

**Revised.** The last paragraph was true only for coordinates near the origin.
The error of the expanded norm is a few parts in 10⁷ of the *norm*, and on data
with an offset that is not a handful of cells -- see [DD-110], which centres
the coordinates before the kernel and restores the claim in general.

---

<a id="dd-80"></a>
## DD-80 · Decimation changes the metrics, not just the cost

**Status:** accepted

Decimating the state space is the obvious way to make a full-recording analysis
affordable, and it is legitimate: after the Hilbert transform the space holds a
phase and an envelope, both slow, and sampling them at 500 Hz is redundant. But
it is not free, and [DD-67] said only that some metrics depend on N. Here are
the numbers.

Four signals, full rate against decimation by two:

| metric | ratio | verdict |
|---|---|---|
| RR | 0.998 | unchanged |
| DET | 0.997 | unchanged |
| LAM | 0.992 | unchanged |
| ENTR_norm | 0.969 | changes a little |
| RTE | 0.959 | changes a little |
| L, TT, Vmax | ≈0.51 | **halve**, as a count of samples should |
| ENTR | 0.808 | changes |
| **Lmax** | **0.017** | **collapses** |

And across a wider range, DET falls from 0.9996 at full rate to 0.9512 at
factor 5 and 0.7753 at factor 10. **Factor 2 is safe, factor 5 is arguable,
factor 10 is not.**

**Lmax is the casualty.** At full rate the longest diagonal spanned almost the
whole record, 9001 samples; at factor 2 it is 157. A long line survives
decimation only if the trajectory stays within ε at the decimated spacing, and
mostly it does not. Lmax should not be reported from decimated data — which is
consistent with it being the weakest measure in [DD-66] and [DD-76] anyway.

**The practical rule.** Decimate by two, use single precision, and report RR,
DET and LAM directly; report L, TT and Vmax knowing they are in units of the
decimated sample and are comparable only across records decimated identically,
which the comparability contract already enforces [DD-42]. Do not report Lmax.

For the eight-minute case that brings a single full-recording plot from roughly
two hours to about twenty minutes per subject.

**Why this is a trap worth naming.** Someone will decimate to go faster and
report a DET that is not comparable with another subject decimated differently.
That is failure mode 3 of [DD-42], now with numbers attached.


---

<a id="dd-81"></a>
## DD-81 · Parallelism must not change the answer

**Status:** accepted

[DD-05] promised that results do not depend on `n_jobs`. That promise is easy
to make and easy to break, and the usual way to break it is to draw random
numbers from a shared generator, so what a worker receives depends on the order
the scheduler happened to run things in.

Every unit of work is therefore given its own stream, derived from the caller's
seed by `SeedSequence.spawn` **before any work starts** and indexed by the
item's position in the input. Results are reassembled in input order.

The promise is tested rather than argued: `test_batch_results_do_not_depend_on_n_jobs`
runs the real pipeline at one and three workers and asserts equality of
epsilon, the recurrence rate, DET, L, LAM and ENTR across every subject and
window.

**A mistake the first implementation made.** The batch dispatched a lambda,
which the thread backend accepts and the process backend cannot pickle. The
worker is now a module-level function and a test pickles it. A parallel path
that works only on one backend is worse than none, because it will be
discovered on the machine where the run matters.

---

<a id="dd-82"></a>
## DD-82 · Parallelise across units, not inside one

**Status:** accepted

A single recurrence plot cannot usefully be split. The line-length accumulator
carries open runs from tile to tile and that carry is sequential by
construction [DD-56]; producing tiles in parallel and consuming them serially
is possible, but the consumer then becomes the bottleneck and the complexity
buys little.

Across **units** -- subjects, windows, parameter cells -- the work is entirely
independent, and that is where a corpus spends its time. `n_jobs` is available
on `batch_windowed_recurrence` and through `parallel_map` for anything else.

**Threads by default, not processes.** The distance kernels spend their time in
BLAS, which releases the interpreter lock, so threads give real parallelism
without pickling a state space per task. `backend="process"` is available where
the work is genuinely interpreter-bound.

**What was and was not measured.** The container this was developed on has a
single core, so the correctness of the parallel path is verified and its
*speedup is not*. A test asserting a speedup would fail on a laptop with one
free core and pass on a cluster, which is not a useful test. The expected gain
follows from the structure -- the units are independent, so wall clock should
scale close to the core count until memory bandwidth binds -- but it is a
prediction, not a measurement, and should be checked on the machine that
matters.

For the eight-minute use case: a single full-recording plot is about twenty
minutes per subject with single precision and decimation by two [DD-79,
DD-80], and forty subjects are independent, so the corpus is roughly two hours
of wall clock on eight cores rather than fourteen.


---

<a id="dd-83"></a>
## DD-83 · Windows of one subject are not independent samples

**Status:** accepted · **the decision the whole classification layer rests on**

A windowed metrics table has one row per subject and window. Treating those
rows as independent observations inflates every estimate: fifteen windows of
one subject share that subject's anatomy, electrode placement, alertness and
noise floor, and a fold that puts some of them in training and the rest in
testing is asking the model to recognise a **subject**, not a group. The AUC
that comes back is high, reproducible, and worthless.

Two levels are offered and both are safe.

**`level="subject"`** reduces the windows of a subject to one row — their mean,
and optionally their spread and temporal trend — and folds are drawn over
subjects. The default, because it cannot leak.

**`level="window"`** keeps the rows, and every fold splits **by subject** using
a grouped splitter, so all windows of a subject fall on the same side. More
data for the model, still no leakage, but the effective sample size is still
the number of subjects and the confidence intervals must be read that way.

The whole-recording case is the same code with a single window, so a user who
wants one recurrence plot per subject and a user who wants fifteen use the same
call.

**The guard is a test, not a comment.** `test_shuffled_labels_score_at_chance`
permutes the labels and asserts the AUC falls to chance. A pipeline that leaks
would score above chance on random labels, and that single test is what makes
any other number in this module worth reading.

---

<a id="dd-84"></a>
## DD-84 · Everything that learns goes inside the fold

**Status:** accepted

Scaling, imputation and any feature selection are fitted on the training part
of each fold and applied to the held-out part, through a pipeline. Fitting them
on all the data first and then cross-validating is the commonest way to produce
an AUC that does not survive new subjects, and it leaves no trace in the output.

This is not hypothetical here: [DD-75] found that a fitted combination of
twenty-one features scored *worse* than two under cross-validation and would
have scored better in sample. The effect was visible only because the fitting
happened inside the fold.

---

<a id="dd-85"></a>
## DD-85 · Imbalanced-aware scores, with a baseline

**Status:** accepted

Plain accuracy on thirty controls against ten patients is 0.75 for a model that
always answers "control". Real corpora are rarely balanced, so the report
carries:

- **AUC**, which ignores the threshold and the prevalence
- **balanced accuracy**, the mean of sensitivity and specificity
- **sensitivity, specificity, precision, F1 and MCC**
- plain accuracy, for comparison with the literature
- the **majority-class baseline** for each, so a number can be compared against
  doing nothing

Classifiers are fitted with balanced class weights by default.

Measured on a deliberately imbalanced corpus of ten controls and six study
subjects: accuracy 0.887 against a baseline of 0.625, balanced accuracy 0.850,
sensitivity 0.700 and specificity 1.000. Reporting accuracy alone would have
overstated the result and hidden that the minority class is the one being
missed.

---

<a id="dd-86"></a>
## DD-86 · One split is an anecdote

**Status:** accepted

With forty subjects a single five-fold split has eight per test fold, and the
AUC swings by more than any effect this project is looking for. The whole
cross-validation is repeated with different partitions and the **mean and
spread across repeats** are reported, per repeat and per fold.

A result whose spread crosses the baseline is not a result, and the summary
prints the standard deviation next to every mean so that is visible without
asking.

Three situations are flagged in the report rather than left to be noticed:
fewer than twenty subjects, more features than subjects, and a minority group
too small for the requested number of folds — in which case the scheme is
reduced and the reduction is stated.

---

<a id="dd-87"></a>
## DD-87 · Significance from permutation, at the subject level

**Status:** accepted

The features are many, correlated, derived from overlapping windows and chosen
from families that overlap. No closed-form test applies, and Bonferroni over
the number of columns would be both wrong and crushing.

The labels are shuffled and the **entire repeated cross-validation is rebuilt**
on each shuffle. The p-value is the fraction of shuffles reaching the observed
score, with the conventional +1 correction in numerator and denominator so it
can never be reported as zero.

**The shuffle is at the subject level.** Permuting rows would leave a subject
holding two different labels, which is not a null hypothesis anyone means, and
would make the null far too easy to beat.

Measured on the imbalanced corpus with sixty shuffles: observed AUC 0.950, null
0.468 ± 0.178, p = 0.016. The null sitting at 0.47 rather than 0.5 is sampling
noise at sixty shuffles and sixteen subjects, and a null that did *not* sit near
chance would be the signal that something in the pipeline leaks.

**Cost.** Each shuffle rebuilds a full cross-validation, so a permutation test
is roughly `permutations` times the cost of the analysis. Two hundred shuffles
of a corpus that takes a minute is three hours; the runs are independent and
`n_jobs` applies [DD-82].


---

<a id="dd-88"></a>
## DD-88 · A recording carries more than its samples

**Status:** accepted

The EDF reader had been ornamental from the first phase: declared, exported,
and never given a file. Installing `pyedflib` and writing one showed it worked
— and that it discarded almost everything a real recording carries.

**What is now kept.** The physical unit of each channel, which is a fact about
the recording and takes effect unless the caller overrides it; the patient
code, which becomes the subject; the start time, the equipment and the
technician; the file duration; and the annotations, with onset, duration and
text. All of it is unrecoverable later.

**Mixed sampling rates are the normal case, not an error.** An EEG montage at
500 Hz commonly sits beside an ECG at 250 and an annotation channel. The
reader used to refuse with a bare list of numbers. It now names the offending
channels with their rates and states the three ways out:

```
/tmp/mixed.edf mixes sampling rates (250 Hz: ['ECG']; 500 Hz: ['Cz', 'Fz', 'Pz']).
Recurra needs one rate per Recording. Select a homogeneous set with
channels=[...], drop the others with exclude=[...], or read the groups
separately and resample.
```

`channels=` and `exclude=` were added for exactly that.

**Verified against real files, not mocks.** The tests write genuine EDF with
`pyedflib` and read them back, and they skip rather than mock where the
optional dependency is absent. A round trip is exact to the quantisation step
of the 16-bit storage, which the test asserts rather than assumes.

---

<a id="dd-89"></a>
## DD-89 · A corpus is files plus a table, and the join is checked

**Status:** accepted

Everything downstream — the batch, the comparability contract, the classifier —
needs a mapping from subject to group. In practice that mapping lives in a
spreadsheet beside the recordings, and the commonest way a study goes wrong is
a silent mismatch: a subject in the table with no file, a file with no row, an
identifier differing by a prefix or by case.

`read_corpus` reports the join rather than performing it quietly. Files without
a row and rows without a file are both listed, `complete` says whether either
happened, and `labelled()` returns only the subjects that have both.

Measured on a twelve-subject study with one deliberate orphan:

```
CorpusIndex: 12 recording(s) under /tmp/study
  labels     : control=7, patient=5   (column 'group')
  1 row(s) with no file: sub-099
```

**The index holds paths, not data.** A corpus of eight-minute recordings cannot
be loaded at once, so the caller loads one at a time. Two files mapping to one
subject is refused rather than silently resolved.

**BIDS is recognised, not validated.** `sub-XXX` directories and a
`participants.tsv` keyed by `participant_id` are the convention, so they are
the default, but this reads files and joins a table — it does not check that a
dataset conforms to the standard, and does not claim to.

**The chain end to end.** Twelve EDF files on disk, joined to a participants
table, ingested one at a time, decimated to 250 Hz [DD-80], cut into ten
overlapping windows, measured with both metric families, and classified with a
permutation test: AUC 1.000, permutation p = 0.038, with both the small-sample
and the more-features-than-subjects warnings firing as they should.


---

<a id="dd-90"></a>
## DD-90 · Resizing a recurrence plot is not resampling a photo

**Status:** accepted

A classifier that takes pictures needs the plot as a fixed-size array. The
obvious way is to hand the matrix to an image library and ask for 256 squared,
which is what the earlier draft did — and at the sizes this project works with
it destroys the thing being classified.

A plot of 34 000 points holds 1.16 billion cells. Nearest-neighbour resampling
to 256 squared keeps **one cell in seventeen thousand**. Two recordings with
quite different structure can then produce near-identical pictures, decided by
which cells happened to land on the grid.

`recurrence_image` **pools**: each output pixel is the mean of the cells it
covers, so every cell contributes and the value is the local recurrence
density. The mean of the whole image is the recurrence rate, which a test
asserts. The picture is a density map in [0, 1], not a binary plot, and that
is the better input anyway: a convolution over densities sees gradients that a
thresholded image has already thrown away.

Maximum pooling remains available and preserves lines rather than density, but
saturates fast [DD-38].

**A fixed size is guaranteed.** Pooling lands near the requested size without
hitting it — 20 000 points reduce to 254, 34 000 to 256 — and a network needs
every image the same shape. The last step is an interpolation over a factor
below 1.05, applied to the already-pooled array. It is recorded in the
metadata as `resized_to_fit`, and it is nothing like the seventeen-thousandfold
discard this module exists to avoid.

---

<a id="dd-91"></a>
## DD-91 · The brightness scale belongs to the batch

**Status:** accepted

At a 5% recurrence rate the pooled densities sit around 0.05, so mapping them
straight onto 0–255 gives a nearly black image. The obvious fix is to divide
each image by its own maximum — and that is the one thing that must not be
done.

Every recording is pinned to the same target rate by construction [DD-35], so
what distinguishes two subjects is **how the density is distributed**, and
per-image normalisation rescales exactly that away. Two subjects with quite
different structure come out looking equally bright.

`vmax` is therefore a property of the batch: one high quantile computed across
every plot exported together, recorded in the metadata, applied to all of them.
A fixed number can be passed when several batches must be comparable.
`vmax="per_image"` exists because someone will ask for it, and its metadata
column says *not comparable*.

**The floor matters as much as the ceiling.** At a reduction factor of 133 each
pixel averages seventeen thousand cells, so by the law of large numbers every
density lands near the global rate: a measured batch spanned 0.050 to 0.105
around a mean of 0.0496. Mapping that from zero puts the whole image between
grey 121 and 255 and the structure is invisible. `vmin` comes from a low batch
quantile for the same reason.

---

<a id="dd-92"></a>
## DD-92 · Past some reduction, decimate rather than pool

**Status:** accepted

Pooling a 34 000-point plot down to 256 pixels costs a full pass over 1.16
billion cells — **98 seconds measured** — and the result is nearly uniform,
because each pixel averages seventeen thousand cells and the law of large
numbers flattens them.

Building the plot from a **decimated trajectory** of about a thousand points
costs under a second and every pixel is a real recurrence decision.

Neither is wrong and they answer different questions: the pooled image is a
density map of the full-rate structure, the decimated one is a plot of a
coarser trajectory. `decimate="match"` chooses the second, the metadata records
`trajectory_decimation`, and **images made the two ways are not comparable with
each other** — which is why the column exists.

For 930 images the difference is a day against twenty minutes.

---

<a id="dd-93"></a>
## DD-93 · Invariants per block as well as per trajectory

**Status:** accepted

The earlier draft computed `lyapunov` and `corr_dim` twice: once univariate,
from a delay embedding of each component, and once multivariate from the full
embedding. Only the multivariate version had been carried over.

Both are useful and they are different quantities: the multivariate one
describes the joint trajectory, the univariate ones describe each signal on its
own. Measured on the same recording, the correlation dimension of the joint PAC
space and of the gamma envelope alone differ substantially.

`block_invariants=True` adds `lambda_1_<block>` and `D2_<block>`, embedding each
block's natural scalar [DD-71]. They are features, **not coupling measures**
[DD-77], and the module says so.

**A failure is warned about, not returned as a silent nan.** A block too short
to embed raised, was caught, and produced `nan` — which reads as a measurement
that came out empty rather than one that never happened. It now warns with the
reason, and a test checks that it does.

**And a rejected estimate must not look like a measurement.** The first version
discarded the `Invariant`'s own warning, so a block Lyapunov exponent of
**55.0 per second** — nonsense from a divergence curve of a hundred steps —
sat in the table unmarked. Every block invariant now carries a companion
`_ok` column, zero when the estimator's diagnostics reject it, because a
warning scrolls past and a column does not [DD-63].

Measured on a 17 000-point PAC space: the default hundred-step curve rejects
every block estimate, and so does a thousand-step one. That is the honest
answer for these signals — **a per-block Lyapunov exponent is not reliably
estimable here** — and it is visible rather than buried.

### How the draft estimated the Lyapunov exponent, and why it was not copied

`nolds.lyap_r(x_sub, trajectory_len=20, fit="poly")` on a decimated signal.
Two shortcuts, both of which bias the result:

**A twenty-step divergence curve.** [DD-63] measured that on Lorenz the initial
transient runs to about step 70 and the linear region follows; a curve of 20
steps fits the transient and roughly doubles the exponent. This is exactly what
the `lambda_1` flag detects.

**Decimating the signal.** `nolds` then returns nats per *decimated* sample and
the draft never multiplied back, so the values were not interpretable as a rate
and were comparable only between records of identical length [DD-62].

The version here is equally fast — 0.4 s at 34 000 points — by a different
route: it subsamples the **reference points**, not the trajectory, so the
divergence curve keeps its full time resolution. Same cost, neither bias.


---

<a id="dd-94"></a>
## DD-94 · The Lyapunov fit window is the estimate

**Status:** accepted · **and it replaces a guard that was wrong**

### The measurement that forced this

Lorenz, whose largest Lyapunov exponent is 0.906 per second. The same
estimator, the same data, one parameter changed:

| max_steps | λ₁ | slope variation inside the fit |
|---|---|---|
| 150 | 1.06 | 0.01 |
| 300 | 0.73 | 0.01 |
| 600 | 1.09 | 0.00 |
| 1200 | **0.30** | 0.01 |

**A factor of three, decided by the length of the curve.** And the flatness of
the fitted window — the criterion [DD-61] relies on — is 0.01 in every case,
because the *saturated plateau is flat too*. Fitting it gives a beautifully
straight line through a region where the separation has stopped growing.

### The guard that was wrong

[DD-63] flagged a fit starting in the first quarter of the curve, on the
reasoning that the transient lives there. The linear region of Lorenz begins
near step 70 **whatever the curve length**, so on a 600-step curve a correct
fit starts at 17% and the guard fired on it. It was a statement about
`max_steps`, not about the data, and it has been removed.

### What replaces it

No automatic criterion reliably picks the linear region; this is a known hard
problem and Rosenstein's own method assumes the curve is inspected. So instead
of a better guess, the estimator **measures how much the answer depends on the
choice**: the exponent is refitted on truncations of the same curve — free,
the curve is already computed — and flagged when it moves by more than
`stability_tolerance`.

Plus a saturation check: a window whose mean sits above 85% of the curve's
range is in the plateau, and says so.

On the table above, 150, 300 and 1200 are now flagged and 600 is not, with the
unflagged estimate the one near the reference.

**An unflagged value is one the data determines. A flagged one is a number
that depends on a parameter** — usable as a descriptor if computed identically
everywhere, but not reportable as an exponent.

### Where λ₁ is estimable, measured

The exponent is defined for any trajectory, univariate or not; what matters is
whether the trajectory has an exponential divergence regime at all.

| state space | dim | λ₁ | verdict |
|---|---|---|---|
| Lorenz | 3 | 1.087 | **usable** |
| delay embedding of a band-limited signal | 6 | 7.10 | **usable** |
| two band signals as coordinates | 2 | 5.63 | slopes vary by 89% |
| the same plus delays | 2 | 5.63 | slopes vary by 89% |
| PAC space: phase circle plus envelope | 3 | 2.03 | fit is in the saturated tail |

Two things follow, and the first corrects an over-broad claim made earlier.

**λ₁ is perfectly estimable from multivariate data.** A delay embedding of one
band, or several channels as coordinates, are exactly the objects the method
was built for.

**A phase-circle space is a poor object for it.** Two neighbours on a circle
rotate together and stay at constant separation — a neutral direction, not an
expanding one — and the envelope separates diffusively rather than
exponentially. The curve saturates almost immediately, which is what the flag
reports. That is a property of the construction, not a defect of the
estimator, and the answer is to compute λ₁ on a delay embedding of the
underlying signals and keep the recurrence analysis on the coupling space.


---

<a id="dd-95"></a>
## DD-95 · Two estimators, two option sets

**Status:** accepted

`dynamics_measures` forwarded one `**invariant_kwargs` to both
`correlation_dimension` and `lyapunov_max`. They do not take the same options:
`max_steps` belongs to a divergence curve and `n_radii` to a correlation sum,
and passing one to the other reaches `find_scaling_region` as an unexpected
keyword.

The result, on a real run: **every block correlation dimension failed**, and
because the failure was caught and turned into a `nan`, the columns
`D2_phase_theta` and `D2_amp_gamma` came out entirely empty across 32
recordings. The warning said so, once per subject, in a stream of a hundred
other warnings.

Options are now routed per estimator. The general point is that catching an
exception and returning `nan` makes a *programming* error look like a
measurement that came out empty; the warning is what saved this one, and it
nearly was not enough.

---

<a id="dd-96"></a>
## DD-96 · A delay embedding needs a large Theiler window

**Status:** accepted

Two points of a delay embedding whose indices differ by less than the embedding
window `(m-1)·τ` **share coordinates** — they are the same numbers in a
different order. Counting them as neighbours makes the cloud look
one-dimensional at small radii, which is exactly where the correlation
dimension is read.

Measured on a Lorenz delay embedding, τ = 16, m = 3, reference D2 = 2.05:

| Theiler | 1 | 16 | 32 | 64 | 200 |
|---|---|---|---|---|---|
| D2 | 1.598 | 1.605 | 1.601 | 1.600 | **1.972** |

The embedding span is 32, and 32 is not enough: what is needed is the orbital
scale. `theiler="auto"` takes the larger of the embedding window and a
hundredth of the record, capped at a twentieth.

**And the radius range decides whether there is a scaling region at all.** With
the lower end at the 0.1st percentile of the distance distribution, the
straight part of the curve spans 0.39 decades and is rejected as too thin. At
the 0.02nd it spans 0.81 and D2 comes out 2.009. The local slopes at small
radii were 2.05, 1.98, 2.00, 1.98 all along — the estimate was right and the
window was too short to accept it.

Together: **D2 = 1.996 and λ₁ = 1.009 on Lorenz, both unflagged**, against
references of 2.05 and 0.906.

---

<a id="dd-97"></a>
## DD-97 · Embed the oscillation, not its description

**Status:** accepted

[DD-94] found that a phase-amplitude space has no exponential divergence
regime. The natural next thought is to embed what the blocks contribute
instead — an instantaneous frequency and a smoothed envelope — and that fails
too, for the same reason: both are stochastic descriptions of a signal, not
the signal.

Measured, every combination rejected by its own diagnostics:

| series | τ | λ₁ | D2 |
|---|---|---|---|
| instantaneous frequency of theta | 1 or 3 | flagged | flagged |
| gamma envelope | 1 or 13 | flagged | flagged |

What the method was built for is the **oscillation**: a band-limited signal
`A(t)·cos(φ(t))`. `invariants_from_series` builds a delay embedding of that,
takes D2 and λ₁ from it, and discards it. The recurrence analysis stays on the
coupling space; only these two move.

The delay and dimension are estimated from the series and **returned with the
result**, because an invariant nobody can audit is not a result.

---

<a id="dd-98"></a>
## DD-98 · DET and LAM need different floors

**Status:** accepted

DET and LAM are bounded above by 1, and a smooth trajectory pushes them
against it. On a real corpus DET came out **0.9965 with a standard deviation
of 0.0005** across 32 recordings — a metric with no room to separate anything,
however real the difference.

Measured across coupling strengths from 0.1 to 0.9:

| floor | α=0.1 | α=0.5 | α=0.9 | range |
|---|---|---|---|---|
| DET, l_min=2 | 0.804 | 0.928 | 0.952 | 0.148 |
| **DET, l_min=8** | 0.177 | 0.447 | 0.608 | **0.431** |
| DET, l_min=20 | 0.006 | 0.084 | 0.247 | 0.241 |

Confirmed on the corpus: with l_min=8, DET moved to 0.766 ± 0.047.

**But the same floor destroys LAM.** Vertical lines are short by nature — a
laminar state lasts a few samples, not eight:

| floor | α=0.1 | α=0.5 | α=0.9 | range |
|---|---|---|---|---|
| **LAM, v_min=2** | 0.754 | 0.176 | 0.048 | **0.706** |
| LAM, v_min=3 | 0.112 | 0.000 | 0.000 | 0.112 |
| LAM, v_min≥4 | 0.000 | 0.000 | 0.000 | 0.000 |

So `l_min=8, v_min=2`. Treating them as one number is wrong in both
directions.

**The defaults are now `l_min=8, v_min=8`** — see the revision at the end of
this entry, which moved `v_min` from 2 to 8 on the strength of a real corpus.

**The original decision was `l_min=8, v_min=2`.** The first draft of this decision
kept both at 2 on the grounds that the literature does, and that reasoning was
wrong for the wrong reason: a value that is comparable with the literature but
constant across every subject is not more useful than one that is
incomparable and informative. A DET of 0.9965 ± 0.0005 cannot answer any
question.

The cost is real and is stated on every result: `l_min` and `v_min` are
recorded in the frame, and **a DET computed here is not the DET of a paper
that used 2**. That is a difference of definition, not of implementation.

**The sweep itself does not warn.** Its job is to evaluate floors that do not
work, so a warning at each one is noise about the very thing being measured --
and it arrives once per floor, per subject, per channel, which on a real run
buried everything else. The `headroom` column says the same thing once, as a
number.

Two things guard it. Saturation *and* collapse both raise a `GeometryWarning`
naming the floor and the remedy — a metric crushed to zero separates nothing
either, and the first version of the check only looked at the ceiling. And
`line_length_sweep` answers the question from the data: every metric at every
floor is a sum over counts the histogram already holds, so a sweep costs
milliseconds against the minutes the plot took.

**The analytic ground truths move with the convention, not with the default.**
The identity DET = 1 − (1 − RR)² for an independent matrix holds at `l_min=2`
and nowhere else. Those tests now pass `l_min=2` explicitly, so a future change
of default cannot silently invalidate the strongest check the module has.


**Follow-up.** `WindowedRecurrence.metrics()` kept `l_min=2, v_min=2` after
`rqa()` had moved to 8, so the two entry points returned different DET for the
same matrix and the whole-record recipe came back saturated. Both default to 8
now, and a test pins them to each other.

**Revision: `v_min` from 2 to 8 (0.22).** This is the revision the paragraph
above refers to; until 0.25 it was announced here but recorded only in the
changelog. The original `v_min=2` rested on one synthetic phase-amplitude
space, where laminar states last a few samples. On the real corpus, 49
delta-gamma recordings gave **LAM = 0.9983 at `v_min=2`**, saturated exactly
as DET had been, so the default moved to 8 with `l_min`. The synthetic table
above still holds for that synthetic space, which is why the validation tests
that re-derive it pin `l_min = v_min = 2` explicitly: the right floor depends
on the space, and `line_length_sweep()` is how to choose it.

---

<a id="dd-99"></a>
## DD-99 · A fine-scale measure needs its own scale

**Status:** accepted

Higuchi, Katz, Petrosian, permutation entropy, sample entropy and Lempel-Ziv
all read point-to-point structure. Given a series sampled far above its own
content they report, correctly, that it is a smooth line.

Across 98 real recordings those columns were constants:

| column | mean | coefficient of variation |
|---|---|---|
| `katz_fd_phase_theta` | 1.0030 | 0.00034 |
| `petrosian_fd_phase_theta` | 1.0007 | **0.00004** |
| `katz_fd_amp_gamma` | 1.0002 | **0.00008** |
| `petrosian_fd_amp_gamma` | 1.0018 | **0.00003** |
| `higuchi_fd_amp_gamma` | 1.0488 | 0.0018 |

Six of thirty-nine columns unable to separate anything, and nothing in the
output said why.

### The series says how oversampled it is

Sign changes in the first difference — what Petrosian's dimension already
counts — give the samples spent between changes of direction. **White noise
sits at 1.5**: one sample per event, which is the scale these measures are
defined at. A gamma envelope low-passed at 8 Hz and sampled at 250 Hz measures
**24**.

Decimating to about two samples per turning point:

| | Higuchi | Katz | Petrosian | permutation |
|---|---|---|---|---|
| as sampled | 1.043 | 1.0003 | 1.0015 | 0.494 |
| **at its own scale** | **1.921** | **1.039** | **1.022** | **0.952** |

The measures were never wrong. They were being asked about a scale where
nothing happens.

### What was done

`fine_scale="auto"` decimates each block to about two samples per turning
point **for the fine-grained measures only**, and records the factor and the
ratio it came from as `fine_scale_<block>` and `samples_per_turn_<block>`. A
decimation nobody can see is one nobody can check.

Everything else sees the whole series: DFA and Hurst need every scale they can
get, Hjorth is a frequency estimate, and the invariants have their own
embedding. A test asserts those are unchanged to nine figures.

A block already at a sensible scale gets factor 1 and is untouched — the phase
circle in the case above.

**Revision (0.25): the factor is capped.** On a 250-point window of a slow
envelope the automatic factor left nine samples, and every fractal and entropy
measure failed as too short. The factor is now limited so that at least
`FINE_SCALE_MIN_SAMPLES = 64` remain, and `dynamics_measures` raises a
`ParameterWarning` when the cap acts. The factor actually used is still the
`fine_scale_<block>` column, next to `samples_per_turn_<block>`, so a capped
row can be recognised in the table.

---

<a id="dd-100"></a>
## DD-100 · A dimension with no scaling region measures the construction

**Status:** accepted

The correlation dimension of the PAC space came out **1.9812 with a
coefficient of variation of 0.0065 across 98 recordings**, and R² = 1.0000 on
every one.

A perfect fit is not reassurance here, it is the symptom. `find_scaling_region`
looks for the part of the correlation sum that is straight; if the curve is
straight over its whole range there is no *region* to find, because nothing
bends.

And 2 is what the geometry gives whatever the signal does: a phase circle
crossed with an amplitude is a torus, dimension 1 plus dimension 1. **The
number is a property of how the space was built.**

The estimate is now flagged when the fitted window covers more than 90% of the
curve with R² above 0.9999, and `region_share` is reported either way.

Measured, the guard discriminates:

| | D2 | region covered | verdict |
|---|---|---|---|
| PAC space | 1.98 | 95% | flagged |
| Lorenz, 40 radii | 2.056 | 57% | **usable**, reference 2.05 |
| Lorenz, 80 radii | 2.058 | 54% | **usable** |

A real attractor leaves most of its correlation sum outside the scaling
region, because a real correlation sum bends.


---

<a id="dd-101"></a>
## DD-101 · The exponent is averaged over samplings, not drawn once

**Status:** accepted

### The measurement

A divergence curve is built from a few thousand reference points drawn at
random out of the record. Which ones are drawn should be a detail. It was not.

Lorenz, 20 000 points, reference exponent 0.906 per second, twelve runs
differing **only in the random stream**:

```
0.501!  0.837  0.703  0.881  0.911  0.875  1.366  0.912  0.954  0.610  0.873  0.679
```

Range 0.501 to 1.366, coefficient of variation 0.25, and **eleven of the twelve
came back unflagged** — including the 1.366, which is 51% from the reference.

For comparison, the two other quantities that also sample:

| | coefficient of variation across streams |
|---|---|
| λ₁ from a delay embedding | **0.25** |
| D2 | 0.0083 |
| recurrence rate | 0.0069 |

The problem is specific to the exponent, and the reason is visible in the
numbers: the divergence curve uses two thousand reference points out of twenty
thousand, while D2 samples four thousand pairs out of 10⁸.

### Why the existing guard missed it

[DD-94] checks whether the estimate survives **truncating the curve**. This is
a different axis: the same curve length, a different draw of points. Two ways
for the answer to depend on something other than the data, and only one was
being watched.

### What was done

`n_repeats` (default 3) draws several independent samplings, fits each,
returns their **mean**, and reports how far they disagreed as `sampling_cv`.
A spread above `sampling_tolerance` is flagged.

**Three, because that is where the benefit is.** Six runs of the same record:

| n_repeats | spread across runs | instability detected | seconds |
|---|---|---|---|
| 1 | 0.424 | 3 of 6 | 5.2 |
| 2 | 0.228 | **6 of 6** | 8.1 |
| **3** | **0.211** | **6 of 6** | 11.0 |
| 5 | 0.105 | 6 of 6 | 16.5 |

Two repeats already catch every unstable case; three halve the spread. Five
halves it again for another 50% of the time, which is worth having on a single
recording and not on nine hundred.

Measured over eight runs of the same Lorenz record:

| | range | mean | flagged |
|---|---|---|---|
| one sampling | 0.380 – 0.903 | 0.780 | 1 of 8 |
| five samplings | 0.631 – 0.989 | 0.833 | **5 of 8** |

The spread narrows and, more usefully, **the instability becomes visible**: the
flag now fires on the runs where the samplings disagree, where before they
passed silently.

### Which axis dominates depends on the space

On a multivariate Lorenz trajectory the sampling barely matters and the curve
length decides everything:

| | mean λ₁ |
|---|---|
| max_steps = 150 | 1.13 |
| max_steps = 400 | **0.894** |
| varying the stream at fixed max_steps | ±4% |

On a delay embedding of one coordinate it is the other way round. Both guards
are needed, and neither replaces the other. A test that had been passing at
`max_steps=150` was fitting the transient and was corrected.

### The honest position

An exponent estimated from finite data is not a number the data determines to
better than about 10-25%, and no amount of guarding changes that. What the
guards do is stop the library reporting a single draw as though it were. **A
value is now the mean of several samplings and carries the spread**; a flagged
one should be used as a descriptor computed identically everywhere, not as an
exponent.

The cost is linear: five repeats of a curve that takes 0.4 s.

### A collision worth recording

The first implementation reused the name `slopes` for the per-sampling fits and
for [DD-94]'s per-truncation fits. The second assignment silently replaced the
first, so `sampling_cv` was computed from the wrong list and every estimate
came back `nan`. Two different questions with one variable name answered
neither.

**Revision (0.25): what averaging buys, measured.** Over twelve streams on
the same Lorenz series the standard deviation went from 0.138 (one sampling)
to 0.119 (three averaged), means 0.932 and 0.928 against 0.906. That is a 14%
narrowing, not the 42% three independent draws would give: the samplings
share a divergence curve and are correlated. A test asserting that the spread
shrinks was therefore a coin toss, and it failed or passed with the platform.
The reliable gain is the flag, so the slow tests now check a contract across
three trajectories: every unflagged estimate lies within 25% of the reference.

Measured on that contract (three Lorenz trajectories, `max_steps` 300 and
1200, three samplings): the two unflagged estimates were 0.911 and 0.791, and
both estimates more than 25% off (1.126, 1.273) were flagged, so no bad value
passed. The guards are conservative, though: two of the four estimates within
25% were flagged as well (0.858, 0.794). A flag on `lambda_1` therefore says
"not established by these data", not "wrong".


---

<a id="dd-102"></a>
## DD-102 · The tail was a warning and not a column

**Status:** accepted

`rms_balanced` equalises the *variance* of the coordinate blocks and not their
*shape*, so a block with a heavy tail contributes points that sit far from
everything else. The library has warned about this since [DD-28] — and that was
the whole of it. **Nothing recorded how heavy the tail was**, so nobody could
check whether it differed between the groups being compared.

It matters more than a warning suggests. With coupling held constant across
sixteen recordings, the tail ratio of the amplitude block drove twenty-two of
thirty-nine metrics:

| | Spearman with the tail ratio |
|---|---|
| spectral entropy, envelope | +0.93 |
| Katz dimension, envelope | −0.91 |
| SVD entropy, envelope | +0.89 |
| RTE | −0.87 |
| DET | +0.83 |
| every phase-block measure | below 0.3 |

And it moved the fraction of isolated points — those recurring with almost
nothing — from 0.3% to 12.9%.

A study whose groups differ in envelope burstiness will therefore show
differences in most of the table for that reason alone. On a real corpus the
tail ratio ranged from 3.3 to 12.6 between subjects.

`shape_descriptors` reports `tail_ratio`, `skew` and `max_ratio` per block, and
`geometry_descriptors` now includes them, because the quantity most likely to
be confounding a comparison should not need asking for. It costs nothing: no
recurrence matrix is built.

**What it is for.** Adjusting for it as a covariate, exactly as the modulation
index was adjusted for in [DD-65]. If a metric still separates the groups with
the tail partialled out, that is a finding; if it does not, it was the shape of
the envelope.

**A correction to the warning's own advice.** It suggested `scale='rank' or
'robust'`. Measured, `robust` changes nothing — tail ratio 9.4 against 9.2,
metrics identical to four figures — because it is a linear rescaling and a
linear map cannot change the shape of a distribution. Only `rank` does, and
the warning now says so.

---

<a id="dd-103"></a>
## DD-103 · An attractor image counts rather than marks

**Status:** accepted

The obvious way to picture an attractor is to draw its points. At 34 000 points
on a 256-pixel grid most pixels are hit many times and the picture saturates
into a silhouette: the shape survives and **how often the trajectory visits
each part does not**.

That is the same failure as thresholding a recurrence plot [DD-90], and the
same answer works. Each pixel holds the number of trajectory points falling in
it, normalised by the largest count, so the value is local occupancy.
`density=False` gives the binary silhouette a scatter plot would show.

Three things carried over from the recurrence images because they were learned
there:

**The brightness scale belongs to the batch** [DD-91], or per-image
normalisation rescales away the differences between subjects.

**The limits are robust** [DD-50]: one outlying burst would otherwise squeeze
the whole trajectory into a handful of pixels, and a test injects one to check.

**The projection is recorded.** A picture of a three-dimensional space is two
of its coordinates, and which two is not a detail: `coords=(0, 1)` is the phase
circle itself, `(0, 2)` crosses a phase coordinate with the amplitude.


---

<a id="dd-104"></a>
## DD-104 · Figures in separate files still have to share their axes

**Status:** accepted

Two attractors drawn on their own axes are two pictures of two shapes. The
reader cannot tell whether one trajectory is genuinely wider than the other or
whether matplotlib chose different limits, and the comparison the figures exist
to support is the one they cannot support.

Within a single comparison figure this was already handled [DD-49]: panels
share limits when the spaces are comparable. **And that was where it stopped.**
A caller writing one file per subject — which is what anyone examining thirty
subjects across thirty-one channels will do — had no way to get the same
treatment, because the logic lived inside `compare_attractors` and nothing
exposed it.

`shared_limits(spaces)` returns the limits, `plot_attractor(..., axis_limits=)`
accepts them. Limits come from the pooled coordinates at a robust quantile, so
one outlying excursion does not squeeze every figure [DD-50].

**Whether sharing is legitimate remains a separate question.** Axes should only
be shared between spaces that are comparable, and `compare_attractors` still
answers that with `share_limits="auto"`. `shared_limits` computes them when the
caller has already decided.

### An asymmetry found on the way

None of the eight comparison figures — `compare_attractors`,
`compare_recurrence`, `compare_routes`, `compare_thresholds`,
`compare_series`, `compare_window_series`, `compare_scale_policies`,
`compare_phase_amplitude` — was reachable as `rc.something`. They existed only
as `rc.viz.compare.something`, while every single-object figure was exported at
the top level. Nothing in the design intended that; it was an import that had
never been written. All eight are now exported.


---

<a id="dd-105"></a>
## DD-105 · The absolute scale of a weighted space is not information

**Status:** accepted · **found by a user looking at the pictures**

### What was reported

Three control subjects from the same channel, drawn on the shared axes
[DD-104] had just introduced. One was a tiny blob, one a flat ring, one filled
the whole box. The question was whether the state space was being built wrong.

### It was not, and the evidence says so

`tail_ratio_phase_theta` measured **1.407 to 1.420 across all 49 subjects**, a
coefficient of variation of 0.002. A phase circle satisfies cos² + sin² = 1, so
that ratio is fixed for a true circle and free otherwise. The geometry was
correct for every subject.

What varied was epsilon: **0.024, 0.101, 0.202** for those three — a factor of
eight.

### The mechanism

`rms_balanced` sets `w_g = (1/rms_g²) / Σ(1/rms_j²) · n`, which makes
`w_g · rms_g²` equal across blocks. The *ratio* is right. But the common value
works out at `n / Σ(1/rms_j²)`, and that follows the raw amplitude of the
blocks: a recording measured in a smaller unit produces a smaller space.

Measured on one signal with only the microvolt gain changed:

| gain | phase-circle radius | epsilon | RR | DET |
|---|---|---|---|---|
| 0.1 | 0.054 | 0.016 | 0.0500 | 0.9072 |
| 1 | 0.509 | 0.144 | 0.0498 | 0.9060 |
| 10 | 1.370 | 0.393 | 0.0493 | 0.8993 |
| 100 | 1.414 | 0.417 | 0.0500 | 0.9033 |

**Nothing computed from the space is affected.** Epsilon follows the scale, the
recurrence rate is pinned to its target [DD-35], and DET moves by 0.8% over a
thousandfold change in gain — less than the seed-to-seed variation. The
recurrence plots are fine and so is every metric in every table produced so
far.

**A picture is affected, completely.** Sharing axes in units that depend on the
recording's gain shows how many microvolts the electrode measured, not what
the trajectory does.

### What was done

`scale_factor(space)` is the root-mean-square extent of the weighted
coordinates. `shared_limits` divides by it before computing limits, and
`plot_attractor` normalises whenever explicit limits are given. After the
change, the three gains above give a circle radius of 1.2247 each.

**The axis labels say so.** A figure that has been rescaled must not be read as
physical units, and a reader who did not choose the normalisation has no other
way to know.

### The general lesson

[DD-104] fixed the right problem in the wrong units. Sharing axes was correct;
sharing them in an arbitrary scale made the figures *look* comparable while
comparing something else, which is worse than not sharing at all — an obviously
incomparable pair invites a second look, a plausibly comparable one does not.

Every metric in this library is invariant to the overall scale. That is exactly
why the scale had never been noticed, and exactly why it had to be removed from
the pictures.


---

<a id="dd-106"></a>
## DD-106 · The reference index belongs in the same table

**Status:** accepted · **the omission that took longest to notice**

A user looked at forty-nine attractors of a phase-amplitude space and said he
could see no coupling in any of them. Checking that took two steps, and the
second was the important one.

### A scatter cannot show phase-amplitude coupling

Not badly, not on this data -- **at all**, and the reason is quantitative. On a
realistic heavy-tailed envelope the spread of amplitude *within* one phase bin
dwarfs the movement of the mean *across* bins:

| coupling | spread within a bin | movement of the mean | ratio |
|---|---|---|---|
| α = 0.3 (MVL 0.16, clearly detectable) | 1.317 | 0.298 | **4.4 : 1** |
| α = 0.9 (MVL 0.46, strong) | 1.324 | 0.873 | 1.5 : 1 |

Individual points drown the effect even when the coupling is strong. So an
attractor showing nothing is not evidence of no coupling, and the observation
that prompted this was correct about the picture and would have been wrong
about the data.

The conditional mean is where the signal is. `modulogram` bins the amplitude
by phase and averages: the contrast between highest and lowest bin measured
0.118 with no coupling, 0.888 at α = 0.3 and 2.546 at 0.9, and the peak lands
on the true preferred phase to within a bin. `comodulogram` does the same
across every band pair, and on a signal coupled theta-to-gamma it puts that
pair an order of magnitude above the other three.

### And the reference index was in no table at all

Chasing the first question exposed the second. Forty-nine subjects, three
channels, a hundred and five columns each -- and **not one classical coupling
index anywhere**. The library had `mvl`, `modulation_index` and `plv` from the
first phase and nothing put them beside the recurrence metrics.

That is the difference between a null result and a well-measured absence. If
MVL is near zero across a corpus, then the recurrence metrics are
characterising a space with no coupling structure, the attractors are correctly
featureless, and an absence of group separation is the expected outcome rather
than a failure of the method. Without the column there is no way to tell, and
the analysis of three channels had already been read as though there were.

`coupling_indices` now also returns the modulogram's contrast and preferred
phase, and `metrics(classical=True)` puts all of them in the windowed table
beside DET and LAM. They cost milliseconds and no recurrence matrix.

**The general form of the mistake.** Every metric in the table was
sophisticated and none was the obvious one. A reference measure is worth having
precisely when it is uninteresting -- it says whether the interesting ones have
anything to work with.


---

<a id="dd-107"></a>
## DD-107 · A circular-shift surrogate saturates its own z-score

**Status:** accepted

The baseline for "no coupling" cannot be a number from a synthetic signal.
Real EEG is 1/f with non-sinusoidal waveforms and both raise the modulation
index without any coupling: measured on one realistic recording the surrogate
null came out **0.0075 against 0.00002** for a smooth synthetic one, 375 times
higher. Comparing a real index against a synthetic reference overstates it by
that factor.

So the surrogate is necessary. But it has a property worth stating, because a
reader will otherwise misread the output:

| coupling | MI | surrogate null | z | p |
|---|---|---|---|---|
| α = 0.0 | 0.00016 | 0.00017 | −0.07 | 0.488 |
| α = 0.2 | 0.00369 | 0.00304 | 2.04 | 0.015 |
| α = 0.4 | 0.01439 | 0.01181 | 2.34 | 0.005 |
| α = 0.6 | 0.03304 | 0.02700 | 2.46 | 0.005 |
| α = 0.9 | 0.08076 | 0.06464 | **2.61** | 0.005 |

**The modulation index rises twentyfold and the z-score moves from 2.04 to
2.61.** A circular shift preserves the envelope entirely, so an envelope
genuinely modulated at the phase frequency is still modulated after shifting —
only its preferred phase moves. The null rises with the signal and the ratio
stops growing.

The p-value is unaffected and reaches its floor of 1/(n+1) from α = 0.4. **Report
the p-value.** A z of 2.6 is not weak evidence here, it is the ceiling of this
surrogate, and reading it as an effect size would understate a strong coupling
as much as the synthetic baseline overstated a weak one.

What the shift *does* test is whether the coupling sits at the observed
preferred phase, which is the question worth asking of a phase-amplitude index.
A surrogate that also destroyed the envelope's own rhythm would test something
else and is not what the literature uses.


### Revision · `v_min` moved from 2 to 8

The original decision set `v_min=2` and argued the point at some length:
vertical lines are short by nature, a laminar state lasts a few samples, and
raising the floor destroys LAM rather than freeing it. The measurement behind
that was a synthetic phase-amplitude space where LAM's range went **0.71 at 2,
0.11 at 3, and exactly zero at 4**.

A real corpus said the opposite. On 49 recordings of a delta-gamma space, LAM
came out **0.9983 at `v_min=2`** — saturated against its ceiling exactly as DET
had been, and for the same reason.

The synthetic envelope was too rough to stand in for a recorded one. It had
short vertical lines because its amplitude jumped between samples; a real
low-passed gamma envelope moves smoothly, so laminar states last far longer and
a floor of 2 catches all of them. The argument was sound and the data behind it
was not representative.

**Both floors are now 8, and the test pins the departure from the classical 2
rather than the number itself.** Which failure mode a space falls into —
saturated or crushed — is a property of the space: on the synthetic one the new
default crushes LAM to zero, and the collapse guard says so. Neither number is
right everywhere, `line_length_sweep` finds the right one from the histogram
already accumulated, and that is the part of this decision worth keeping.


---

<a id="dd-108"></a>
## DD-108 · The line histogram is kept, so exploring floors is free

**Status:** accepted

[DD-98] establishes that the line-length floor has to be chosen from the data,
and gives `line_length_sweep` for doing it from a single histogram. That works
for one recurrence plot. For a windowed batch there was no such route, and a
caller wanting the same answer had to call `metrics()` once per floor.

Which is very expensive, because changing `l_min` invalidates the metrics cache
and re-walks every matrix. Measured on a 19-window batch of two subjects:

| | seconds |
|---|---|
| building the windowed batch | 13.0 |
| first `metrics(rqa=True)` | 50.6 |
| the identical call again | 0.0 (cached) |
| one call at a different floor | **50.1** |
| a seven-floor sweep, this way | **350** |

**Five times the cost of the analysis it was meant to inform**, and the study
script doing it was the reason a single channel took much longer than the
measured cost of its own recurrence plots.

Every RQA metric at every floor is a sum over counts the histogram already
holds. `WindowedRecurrence` now keeps them, keyed by subject and window, and
exposes `line_length_sweep`. The same seven floors: **0.06 seconds**.

It also fixes a second problem the script had reintroduced. Calling `metrics()`
per floor emits the saturation warning at each one — the noise [DD-98] had
already removed from `line_length_sweep`, arriving again through a different
door. Routing through the sweep silences it, because the sweep's whole job is
to evaluate floors that do not work.

**Why windows need this more than whole records do.** The floor that suits a
136-second recording does not suit a 13.6-second window: measured on the same
data, LAM read 0.913 at a floor of 2, 0.582 at 3, 0.149 at 4 and zero from 6,
while the whole record wanted 8. A parameter that has to be re-chosen at every
window length has to be cheap to re-choose.


---

<a id="dd-109"></a>
## DD-109 · The metrics are the work, so they parallelise too

**Status:** accepted

`batch_windowed_recurrence` takes `n_jobs` and spreads the *build* over every
core: resolving windows, sampling distances, estimating thresholds. It does not
keep the matrices [DD-41], so the first call to `metrics()` rebuilds and walks
every one of them.

That is where the time is. Measured on a 34 000-point record in 19 windows:

| | seconds, one subject |
|---|---|
| build and resolve | 4.8 |
| `metrics(rqa=True)` | 20.0 |
| `metrics(rqa, classical)` | 19.6 |
| `metrics(rqa, classical, dynamics)` | 27.7 |

**The cheap quarter was parallel and the expensive three quarters ran on one
core.** On 49 subjects that is sixteen minutes per channel with fifteen cores
idle, which is what a user reported as "still not finished on one channel"
after the sweep had already been made free [DD-108].

`metrics()` now takes `n_jobs`. Results do not depend on it: each record's
stream is fixed at build time [DD-81], and a test asserts the tables match.

### The cache does not survive a process boundary

The first version lost the histograms. `metrics()` stores them on each
`WindowedRecurrence` so that [DD-108]'s sweep can reuse them — but a process
worker returns a *copy*, so everything it cached stayed in the worker and the
original came back empty. The sweep then failed with "no histograms yet",
which was at least a clear failure rather than a silent recomputation.

They are returned explicitly alongside the frame and merged back. Anything
cached inside a worker has to be handed over deliberately; nothing crosses that
boundary by itself.

---

<a id="dd-110"></a>
## DD-110 · Centre the coordinates before expanding the norm

**Status:** accepted

The kernel computes `||a||² + ||b||² − 2a·b` [DD-33]. The algebra is exact and
the arithmetic is not: when the coordinates sit far from the origin the two
norms are large and nearly equal, and the distance is what remains after they
cancel. The error is therefore a few parts in 10⁷ of the *norm*, not of the
distance, and it grows with the offset.

Measured on a phase-amplitude space at a 5% rate, single against double
precision on the same 6000 points:

| offset added to every coordinate | cells that differ | RR double | RR single |
|---|---|---|---|
| 0 | 0 of 3.2e7 | 0.0507 | 0.0507 |
| 100 | 21 468 | 0.0507 | 0.0503 |
| 1000 | 1 780 704 | 0.0507 | **0.0822** |

A phase-amplitude space is nearly centred, so the study was never affected;
a raw channel with a DC offset, or an unscaled delay embedding, would have
been. Distances are invariant to translation, so the column mean of the first
trajectory is subtracted from both trajectories once before the kernel sees
them. The same vector is used for both, which keeps a cross recurrence plot
exact. It is applied in double as well, so the two precisions follow one code
path. A test asserts that offsets of 100, 1000 and 10 000 leave a single-
precision plot bit for bit unchanged.

**Revision (0.25): only for translation-invariant metrics.** Cosine distance
is the angle seen from the origin, and centring moves the origin. It was
centred anyway, while its threshold was estimated on the raw coordinates, so
matrix and threshold described different geometries: on an offset point cloud
at a target rate of 0.05 the matrix came out at 0.0013, with 113 630 cells
wrong. Centring now applies to Euclidean, Chebyshev and Manhattan only. A test
checks all four metrics cell by cell against `scipy.spatial.distance.cdist`.

---

<a id="dd-111"></a>
## DD-111 · Normalised entropies count admissible lengths, not lengths present

**Status:** accepted

`ENTR_norm` and `RTE` divided the entropy by log of the number of *distinct
lengths present* in the histogram. That number is itself random: it rises by
one whenever a single line of a new length appears, so one rare long line --
which says nothing about the shape of the distribution -- moved the normalised
value. It also depends on the recurrence rate and the record length in a way
that has nothing to do with entropy.

The denominator is now log of the number of *admissible* lengths, `Lmax −
l_min + 1` (and the same for the vertical and white families with their own
floors). Two equiprobable lengths out of two admissible give 1.0; the same two
out of 243 give 0.13 where the old convention gave 0.63 after a single line of
length 250 appeared. The raw `ENTR` is untouched, and the comparability layer
still requires a matched point count for it.

Numbers computed before this version are not comparable with numbers computed
after it for these two columns.

---

<a id="dd-112"></a>
## DD-112 · The envelope low-pass has to pass the phase band

**Status:** accepted

`pac_space(smooth=...)` low-passes the envelope so that estimation noise faster
than the modulation does not fill the space [DD-29]. But the modulation the
space exists to show runs at the phase band's own frequency. A cut-off below
the band's upper edge removes it by construction, and every metric downstream
then describes an envelope from which the coupling has been filtered out
before it was measured.

The study's loader used 8 Hz for every pair. That is right for theta (4–8 Hz)
and blind to alpha (8–12) and beta (12–30): a beta–gamma space built that way
cannot show beta–gamma coupling whatever the data contain.

When the band edges are known -- recorded by `bandpass()`, or passed to
`ingest()` as `meta["bands"]` for a decomposition done elsewhere --
`pac_space` warns when `smooth` lies below the upper edge of the phase band.
It warns rather than raises because a cut-off inside the band is occasionally
wanted; but it is never wanted silently.

---

<a id="dd-113"></a>
## DD-113 · The amplitude band has to hold the sidebands

**Status:** accepted

An envelope modulated at *f* Hz around a carrier at *f_c* is, in the spectrum,
the carrier plus two sidebands at *f_c ± f*. A band-pass narrower than 2*f*
around the carrier cannot pass both sidebands, and the modulation is removed
before the envelope is taken. Phase-amplitude coupling with a slow band whose
upper edge is *f_max* therefore needs an amplitude band at least 2*f_max* wide,
however well the band isolates the carrier otherwise (Aru et al. 2015).

Measured on a synthetic 40 Hz response modulated by a 6 Hz phase at depth 0.5:

| amplitude band | width | MVL | MI |
|---|---|---|---|
| 38–42 Hz | 4 | 0.000 | 0.0000 |
| 35–45 Hz | 10 | 0.014 | 0.0001 |
| 32–48 Hz | 16 | 0.156 | 0.0085 |
| 30–50 Hz | 20 | 0.203 | 0.0145 |
| 30–80 Hz | 50 | 0.195 | 0.0134 |

The study had just added a 35–45 Hz band around its steady-state response,
chosen for isolating the driven response from the 1/f tail and the muscle. As
built it could not have shown theta coupling whatever the data contained.
`pac_space` now warns when the amplitude band, if recorded, is narrower than
twice the phase band's upper edge; the study band moved to 30–50 Hz.

---

<a id="dd-114"></a>
## DD-114 · The unit of inference is the subject; the peak of a curve is a max-statistic

**Status:** accepted

Aligned windows invite two comparisons beyond the aggregated one, and each has
a standard mistake attached.

*Window-level testing.* Treating every window as an observation multiplies the
sample size by the window count with rows that are strongly correlated within
a subject. Measured on a null table (no group effect, 49 subjects x 24
windows, within-subject correlation 0.64): a Mann-Whitney over windows pooled
as independent rows rejected at p < 0.05 in **62%** of 400 simulations; over
per-subject medians, in 4.8%. Aggregation to one number per subject is not a convenience, it is
what keeps the test honest -- and `classify_groups` already splits folds by
subject for the same reason [DD-83].

*Reading the best window.* A 24-point AUC curve offers 24 correlated looks;
its extreme read against an uncorrected 0.05 is fished. `group_timecourse`
permutes **subject labels** and recomputes the whole curve, so ``p_peak`` is
the probability of an excursion this large *anywhere* on the curve, and
``p_trend`` asks separately whether the separation drifts along the recording.
Rows excluded by a quality mask stay excluded under every permutation, so
uneven window loss between groups cannot fake an effect by itself. Validated
on constructed tables: an effect present only in the second half returns
p_peak 0.02 and p_trend 0.004; a time-constant effect returns p_peak 0.0005
with p_trend 0.85 -- the two questions separate; a null metric returns 0.64
and 0.84. `classify_timecourse` applies the same discipline to a fitted
combination of features, with the same subject-level permutation of the whole
curve when asked.


---

<a id="dd-115"></a>
## DD-115 · The Theiler window follows the standard convention

**Status:** accepted (0.25.0)

**Context.** DD-19 stated that `theiler=0` excludes nothing and `theiler=1`
excludes the line of identity. The code excluded every pair with
`|i - j| <= theiler`, so `theiler=1` removed the line of identity *and* both
first off-diagonals, and in general a window `w` here was a window `w + 1` in
the CRP Toolbox, pyunicorn and PyRQA, which exclude `|i - j| < w`. Code, tests,
pair sampler and line histogram agreed with each other, so nothing failed; the
discrepancy was with the decision record and with every other tool. It was
found in the 0.24.0 audit by reading DD-19 against `RecurrenceMatrix._block`.

A second, smaller inconsistency came with it: at `theiler=0` the pair sampler
rejected self-pairs while the matrix kept the line of identity, so the
estimated and the realised rate differed by about `1/N`.

**Decision.** `theiler=w` excludes every pair with `|i - j| < w` everywhere a
recurrence region is defined: the matrix tiles, the recurrence-rate
denominator, the line histograms (streamed and dense), the white-line band
and the threshold sampler. `0` excludes nothing, self-pairs included; `1`
the line of identity only; `w` the `2w - 1` central diagonals. One helper,
`recurrence.core.theiler_band_cells`, counts the band for every denominator.

Neighbour searches follow the same window with one addition: a point is never
its own neighbour, so they require `|i - j| >= max(theiler, 1)`. This covers
the FAN threshold, `geometry_descriptors`, the correlation sum behind D2 and
K2, and both Lyapunov divergence methods.

**Consequences.** Any result computed before 0.25 with `theiler=w` is
reproduced with `theiler=w + 1`. At the library's usual `theiler=1` the
first off-diagonals are now kept: on smooth trajectories they are nearly
always recurrent and lie on the diagonal lines next to the line of identity,
so DET and L rise slightly. The `denominator="all"` option of `rqa()` now
restores exactly the excluded cells at `theiler=1`, which is what it always
claimed [DD-58].

**Alternatives rejected.** Keeping the old arithmetic and correcting only the
text: the library is general, and a parameter that means something different
from every other implementation of the same quantity is a standing source of
irreproducible comparisons. A new parameter name for the new meaning: two
names for one concept is worse than one migration note.

---

<a id="dd-116"></a>
## DD-116 · Canonical coupling indices need an envelope

**Status:** accepted (0.25.0)

**Context.** `WindowedRecurrence.metrics(classical=True)` computed MVL, MI and
the modulogram summaries from the amplitude *coordinate* of the state space.
When that block had been rescaled (`scale_amplitude="zscore"`, `"robust"`,
`"rank"`), the coordinate was no longer an envelope: a z-scored one gave MVL
13.3 and MI 0.88, or NaN, on a signal whose values were 0.31 and 0.035, and
nothing said so. The same audit found that the default envelope smoothing
(a Butterworth low-pass, DD-29) rings below zero on strongly modulated
envelopes: 4% of samples negative, down to -54% of the mean, at alpha 0.9.

**Decision.** `build()` records `scale` and `smooth_hz` in each block's
`CoordGroup.params`, so the information travels with every window and
subsample. The windowed table writes NaN for the amplitude-based indices of a
rescaled block, with one `CaveatWarning` naming the policy. `mvl`,
`modulation_index` and `modulogram` warn with a `ParameterWarning` whenever
the amplitude has negative values, and `modulation_index` returns NaN when a
phase bin's mean is negative, because the divergence is then taken from
something that is not a distribution. The modulogram now wraps the phase as
the modulation index always did; a phase in [0, 2π) used to land every value
above π in the last bin.

**Decided separately.** Whether the envelope smoothing itself should
preserve the sign changes the geometry of every smoothed space, so it has its
own entry: [DD-117].

---

<a id="dd-117"></a>
## DD-117 · Envelope smoothing preserves the sign

**Status:** accepted (0.25.0). Revises the method of [DD-29], not its purpose.

**Context.** The envelope low-pass of DD-29 was an order-4 Butterworth applied
forward and backward. Its impulse response has negative lobes, so on a
strongly modulated envelope -- which dips close to zero once per slow cycle --
it rang below zero: on generator PAC at alpha 0.9, 4.7% of samples were
negative, down to -54% of the mean. A negative amplitude is not a point of
any envelope; it distorted the geometry of every smoothed PAC space near the
troughs, which is where the modulation is, and it made the canonical indices
computed from the coordinate ill-defined [DD-116].

**Decision.** `smooth_method="gaussian"` is the default: convolution with a
Gaussian kernel, which is non-negative, so the smoothed envelope is a weighted
mean of non-negative samples and cannot go below zero. Its width is set so
the gain at the cut-off is exactly 1/2, the gain the Butterworth had there,
so `smooth=f` means the same thing at `f`. `smooth_method="butterworth"`
keeps the old filter for reproducing earlier results; the slow validation
tests pin it, because their findings were measured with it. Both are
recorded in `CoordGroup.params`.

**The cost, measured.** No non-negative kernel is both flat below its cut-off
and sharp above it, and the Gaussian is the best compromise between the two,
not an exception. With the cut-off at 8 Hz:

| frequency | 4 Hz | 6 Hz | 8 Hz | 12 Hz | 16 Hz |
|---|---|---|---|---|---|
| Butterworth gain | 0.996 | 0.909 | 0.500 | 0.038 | 0.004 |
| Gaussian gain | 0.841 | 0.677 | 0.500 | 0.210 | 0.062 |

On generator PAC (theta 4-8 Hz, five seeds) the coupling left in the smoothed
envelope, as MVL, was 0.482 unsmoothed, 0.420 Butterworth (clipped at zero)
and 0.331 Gaussian at alpha 0.9; 0.149, 0.132 and 0.103 at alpha 0.5. So with
the Gaussian the cut-off should sit at about 1.5 times the upper edge of the
phase band: at 12 Hz for theta it keeps 74% of an 8 Hz modulation and 84% of
a 6 Hz one. The DD-112 guard is unchanged -- for both methods, a cut-off below
the band's edge means a gain below 1/2 at the edge.

**Alternatives rejected.** Clipping the Butterworth output at zero: keeps the
sharp response, but projects 4.7% of samples onto a kink at zero and leaves
the positive ringing beside them. Smoothing the logarithm of the envelope:
positive by construction and sharp, but it filters a geometric mean, which is
biased downward and amplifies the troughs, exactly where precision matters.

---

<a id="dd-118"></a>
## DD-118 · No counter-intuitive calls: one name per idea, defaults that agree

**Status:** accepted (0.25.0)

**Context.** An audit of the 145 public functions found calls that did what
the code said and not what a reader would expect:

* `threshold()` defaulted to `theiler=0` and `recurrence_plot()` to `1`, so
  the natural `rc.recurrence_plot(ss, rc.threshold(ss))` estimated the rate
  over one region and drew another; and a threshold estimated at `theiler=1`
  used at `theiler=50` gave a rate of 0.024 for a target of 0.05, silently,
  although the `Threshold` object knew its own window.
* `store="none"` meant "stream", and read as "no storage".
* Randomness was `rng` in 28 functions and `seed` in the generators;
  `normalise` in seven functions and `normalize` in one.
* The windowed functions hid `target_rr`, `theiler`, `metric` and `store` in
  `**kwargs`, so `help()` did not show them, and `precision` was not passed
  through at all; the cross plot had no `precision`.
* A target rate below the share of exactly repeated pairs put epsilon at
  rounding level (4.7e-14 on a sine), and floating point then decided the plot.

**Decision.** Both defaults are `theiler=1`; `recurrence_plot` and
`joint_recurrence_plot` take `theiler=None`, which inherits the window of a
`Threshold` object, and warn when an explicit window or metric differs from
the threshold's. `store="stream"`, with `"none"` kept as an alias. The
generators accept `rng` as well as `seed` (one of the two); `mvl` accepts
`normalise` (and the old `normalize`). The windowed functions list their
common options explicitly; `precision` reaches every structure. `threshold()`
warns when epsilon is at rounding level. Old spellings keep working, so no
script breaks.

**A regression caught before release.** In the first draft the explicit
windowed options were merged *after* the per-window thresholds had been
estimated, so every windowed threshold used `target_rr=0.05`, `theiler=1`
and the Euclidean metric whatever was asked for. The fast tests all passed,
because they asked for exactly those defaults; a slow validation test caught
it as a drift of RR between window lengths. A test now sets all three to
non-default values (0.20, 7, Chebyshev) and checks them on every window.

---

<a id="dd-119"></a>
## DD-119 · Meta-recurrence of the window measurements

**Status:** accepted (0.25.0)

**Context.** The project this library serves asks for coupling to be treated
"not as a static index but as a dynamic object whose temporal evolution can be
analysed with RQA": a series of measurements taken window by window, and its
recurrence. The first study did this by hand -- a 24-point recurrence matrix
of the window trajectory, with a Theiler window chosen to exclude
overlapping windows -- because the library had no such object.

**Decision.** `meta_space(table)` turns a per-window table (or a
`WindowedRecurrence`) into a `StateSpace` with one point per window, one
coordinate per measurement, `fs` in windows per second and `t0` at the first
window centre. Every existing tool then applies unchanged. `meta_recurrence`
is the shortcut, for `kind="rp"`, `"crp"` (two analyses with the same
columns, e.g. two channels) and `"jrp"` (several, each with its own
threshold); `WindowedRecurrence.meta_recurrence()` is the method form.

Defaults, each for a reason. `theiler="auto"` = `ceil(size / step)`, which
excludes every pair of windows sharing a sample (2 at 50% overlap). Columns are
measurements only: bookkeeping columns are never used, the recurrence rate is
dropped when a target-rate threshold fixed it (a constant by design), and so
are constant columns. Each column is rescaled on its own (`zscore` by
default), because they have different units. Windows with missing values are
dropped with a warning, or refused with `missing="error"`.

**Verified.** A table alternating between two regimes every three windows
gives a meta-RP that is exactly 1 on the diagonals at multiples of six and
exactly 0 halfway between.

---

<a id="dd-120"></a>
## DD-120 · A recurrence plot is drawn with its signals

**Status:** accepted (0.25.0)

**Context.** `plot_recurrence` drew the matrix alone. A bare matrix shows
where recurrences are, not what the trajectory was doing there, and a joint
plot shows the conjunction without the two plots it is the conjunction of.
The classical presentation (Eckmann, Kamphorst and Ruelle 1987; the CRP
Toolbox) puts the signals along the axes, and the library's logo is that
layout.

**Decision.** Three composite figures, registered like every other:
`plot_recurrence_panel` (any structure -- RP, CRP, JRP, meta-RP -- with the
column trajectory along the top in blue, the row trajectory along the left in
orange, each coordinate z-scored, optionally the RQA measures beside it, and
`signals=` to show the raw signal instead), `plot_joint_recurrence` (each
subsystem's plot, the conjunction, and an overlay coloured by which subsystem
recurs, titled with the JRR ratio), and `plot_rqa_summary` (the plot, the
diagonal and vertical line-length histograms with the floors marked, and the
measures). They use the logo's colours. In them the line of identity runs
from bottom left to top right, the classical orientation; `plot_recurrence`
keeps the matrix orientation it always had. `tools/make_gallery.py`
regenerates the gallery shown in the README from fixed seeds.

---

<a id="dd-121"></a>
## DD-121 · D2 is a small-radius limit: the search stays below the saturating scale

**Status:** accepted (0.25.0). Refines [DD-61].

**Context.** The first GitHub CI run failed on all five macOS runners in one
test: D2 of Lorenz from a delay embedding came out right (about 2.0) but
flagged. The macOS runners are ARM machines, whose rounding differs (fused
multiply-add), and a chaotic trajectory integrated for 200 s turns that into a
different trajectory. Reproducing it by perturbing the initial condition by
1e-9 found something worse than the CI had: one trajectory gave **D2 = 1.61,
unflagged**. Its local-slope curve had two plateaus, the true one at about 2.0
for small radii (C(r) below 1%, radii under 5% of the attractor) and a second,
flatter one at about 1.6 for large radii (C(r) from 7% to 50%), where the
correlation sum bends toward saturation. DD-61 chose the flattest window
wherever it was, and both passed its width and flatness checks.

**Decision.** `correlation_dimension` searches automatically only where
`C(r) <= max_C`, default 0.05: a ball holding more than 5% of the attractor
is not small, and the correlation dimension is a small-radius limit
(Grassberger and Procaccia 1983; Eckmann and Ruelle 1992). If fewer than
`min_points` radii qualify, the whole curve is searched and the region's own
checks judge the result. `region=(lo, hi)` still fits any window by hand.

**Verified.** Over twelve Lorenz trajectories 1e-9 apart, D2 ranged from 1.94
to 2.06 (reference 2.05); ten were unflagged and the two flagged ones were
right too. The test is now a contract over four such trajectories, one of
them the case that gave 1.61, instead of a fact about one machine.
