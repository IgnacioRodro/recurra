# Contributing to recurra

These are the rules the project already follows. They exist because each of
them was learned by breaking it once; the reasons are given so that the rules
can be applied with judgement rather than by rote.

## 1. Agree on a change before writing it

Describe what you want to change and why, and agree on it before writing
code. A change that alters results, a default value or the meaning of a
parameter needs that agreement in writing, as a design decision (section 2).

## 2. Every decision with consequences has a DD entry

Decisions live in [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md), one
entry per decision, in the format the file already uses:

- an anchor and a heading, `<a id="dd-NN"></a>` and `## DD-NN · Title`;
- a status: `accepted`, `provisional`, `revised` or `rejected`;
- the context, the decision, the alternatives rejected and why;
- what was measured, with the numbers.

Number the new entry after the last one, with no gaps, and add it to the index
table at the top of the file. Cite it in the docstrings of the code it governs
as `[DD-NN]`. A decision that changes is not rewritten: it gets a dated
revision note, or a new entry that replaces it.

`tests/test_docs.py` checks that the index matches the sections, that the
numbering has no gaps, that every DD cited in the source exists and that the
major ones are cited somewhere [DD-45].

## 3. English throughout

Code, comments, docstrings, documents, commit messages and test names are in
English [DD-30].

## 4. Tests

Run the fast suite after every change:

```bash
python -m pytest -m "not slow" -n auto
```

Run the slow suite as well whenever a default value or a measurement decision
changes. It re-derives the measured findings, and between 0.22 and 0.24 nine
of its tests stayed broken for months because nobody re-ran it after the line
floors moved:

```bash
python -m pytest -m slow
```

When you write tests:

- **Check against an external reference whenever one exists**: `scipy`, a
  published value, or a deliberately naive implementation whose correctness is
  obvious by inspection. Reasoning about code did not find the white-line bug
  of DD-56; comparing against a reference found it in one run.
- **Optional backends.** A test that needs one carries
  `@pytest.mark.requires("module")` and is skipped without it. Every test
  without the mark must pass with the core alone (`numpy`, `scipy`, `pandas`);
  the `core-only` CI job enforces it (principle P6).
- **Chaotic systems: test contracts over several trajectories, never facts
  about one.** Rounding differs between machines (ARM runners use fused
  multiply-add) and a chaotic trajectory amplifies the difference into a
  different trajectory. A test on one trajectory tests one machine.
- **Not only round sizes.** Use sizes that are not multiples of the tile size
  or of the display reduction factor (1157 points, for instance): 400 and 900
  hid a figure failure that the cookbook found.
- **Not only defaults.** A test that uses only default values cannot see an
  option that gets lost on its way; set non-default values and check that they
  arrive where they matter.
- **Parallel paths give the same answer as serial ones** [DD-81]: test them at
  more than one value of `n_jobs`.
- Slow tests carry `@pytest.mark.slow`.

## 5. Keep the documents in step

Several documents are checked by `tests/test_docs.py`, so a change that
forgets them fails the suite rather than leaving them stale [DD-45]:

- after adding or removing a test function, run
  `python tools/generate_test_index.py` (it rewrites `docs/TESTS.md`) and
  update the count `**N test functions, all green**` in
  `docs/ARCHITECTURE.md` and in `README.md`;
- every module appears in the map of `docs/ARCHITECTURE.md`, every registered
  figure in its catalogue;
- every public name appears in `docs/EXAMPLES.md` and is exercised by
  `examples/example_06_cookbook.py`;
- the version in `src/recurra/_version.py`, the newest heading of
  `CHANGELOG.md` and, for releases, `CITATION.cff` agree.

Do not write test counts anywhere else: a number that no test checks goes
stale.

Record every change in `CHANGELOG.md` under the newest heading, with its DD
reference, changes that alter results first.

## 6. Code conventions

- The principles P1 to P7 in [`README.md`](README.md), and DD-118: one name
  per idea, defaults that agree between functions, common options visible in
  the signature, old names kept as aliases.
- **Never simplify silently** (P3): every choice made on the user's behalf is
  recorded in the provenance and announced with a warning.
- Warnings carry a category (`CaveatWarning`, `ParameterWarning`,
  `GeometryWarning`, `ComparabilityWarning`, `InferenceWarning`) and a fixed
  text, so that a loop over windows warns once [DD-46].
- Errors are actionable: `ParameterError` says what to pass instead,
  `BackendUnavailable` names the extra to install, `MemoryBudgetError` gives
  the numbers.
- Random numbers come from an explicit `rng` argument, never from global state
  (P5).
- Expensive loops over independent units (subjects, windows, parameter cells)
  are parallelised, without changing the answer [DD-81, DD-82].
- Anything a script prints that may be needed later is also written to a file.
- Lint with the ruff version pinned in the `dev` extra:
  `ruff check src tests examples tools`.

## 7. Releases

Follow the checklist in section 5 of [`docs/ROADMAP.md`](docs/ROADMAP.md).
