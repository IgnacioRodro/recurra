<picture><source media="(prefers-color-scheme: dark)" srcset="_static/recurra-mark-dark.svg"><img alt="recurra" src="_static/recurra-mark.svg" width="56" align="right"></picture>

# Installation and first run

Aimed at running `recurra` from Spyder on a local machine. Everything here
also works from a plain terminal or a notebook.

---

## 1. Requirements

- Python **3.10 or newer**
- The core needs only `numpy>=1.24`, `scipy>=1.10` and `pandas>=1.5` — all of
  which an Anaconda distribution from 2023 onwards already satisfies, so
  installing does not upgrade anything (see section 9)
- Figures need `matplotlib` (install it: every plotting call raises a clear
  error naming the missing extra otherwise)

Check what you have:

```python
import sys; print(sys.version)
```

---

## 2. Install

Unzip the package somewhere stable — **not** inside a synced folder that
rewrites files, and not inside the folder you will run scripts from, to avoid
shadowing. Then, from a terminal (Anaconda Prompt on Windows):

```bash
cd path/to/recurra
pip install -e ".[dev,viz]"
```

`-e` (editable) means the installed package points at the source, so any edit
takes effect on the next kernel restart without reinstalling.

### If you use conda environments

```bash
conda create -n recurra python=3.11 numpy scipy pandas matplotlib spyder-kernels
conda activate recurra
cd path/to/recurra
pip install -e ".[dev,viz]"
```

Then point Spyder at that environment: **Tools → Preferences → Python
interpreter → Use the following interpreter**, and select the `python`
executable inside the environment. Spyder needs `spyder-kernels` installed in
that environment, which the command above does.

### Without installing

If you would rather not install anything, add the source directory to the path
at the top of each script:

```python
import sys
sys.path.insert(0, r"C:\path\to\recurra\src")   # adjust
import recurra as rc
```

This works but skips the dependency check, so make sure `numpy`, `scipy`,
`pandas` and `matplotlib` are present.

---

## 2b. Installing into the interpreter Spyder actually uses

The commonest installation failure is `ModuleNotFoundError: No module named
'recurra'` right after a `pip install` that appeared to succeed: the package
went into one interpreter and Spyder is running another.

First find out which interpreter Spyder is using, from its console:

```python
import sys
print(sys.executable)
print(sys.version)
```

Then install into **that** interpreter, driving pip from it directly so there
is no ambiguity about which pip runs:

```python
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-e",
                r"C:\path\to\recurra"])
```

Restart the kernel afterwards (Ctrl+.). Nothing installed while a kernel was
running is visible to it until it restarts.

If `sys.version` reports **3.9 or older**, the install will have failed with a
`requires-python` message that is easy to miss in the scroll. Create a newer
environment as in section 2.


---

## 9. Verified environments

The dependency floors in `pyproject.toml` are a claim about what works, so
they are checked rather than assumed. The full suite has been run on both ends
of the declared range:

| | Python | numpy | scipy | pandas | Platform | Result |
|---|---|---|---|---|---|---|
| floor | 3.10.9 | 1.26.4 | 1.10.0 | 1.5.3 | Windows (Anaconda) | 281 passed |
| current | 3.12.3 | 2.4.4 | 1.17.1 | 3.0.2 | Linux | 281 passed |

The versions between are untested but the library uses no API that changed
across them; a `minimum-versions` CI job pins the floors and runs the suite, so
if that stops being true it will show up rather than being discovered by a
user.

**Practical consequence:** an Anaconda base environment from 2023 (Python
3.10, pandas 1.5) needs **no package upgrades**. Installing there will not
touch numpy, scipy or pandas.

---

## 3. Verify

Restart the kernel first (**Consoles → Restart kernel**, or Ctrl+.), then:

```python
import recurra as rc
print(rc.__version__)
print(rc.doctor())
```

Expected:

```
recurra 0.25.0.dev0
  OK  numpy
  OK  scipy
  OK  pandas
  --  scikit-learn
  OK  matplotlib
  ...
```

Lines marked `--` are optional backends that are not installed. The core runs
without any of them; each extra (`io`, `ml`, `viz`, `parallel`) enables the
functions that need it, and a missing one raises `BackendUnavailable` naming
the extra to install.

Run the test suite from a terminal (not from Spyder):

```bash
cd path/to/recurra
pytest -q
```

603 test cases (509 test functions) should pass. `pytest -m "not slow"` runs
the 573 fast ones; on a laptop that takes about ten minutes, on a slower
machine up to forty. The fifteen EDF reader tests need the `io` extra
(`pyedflib`) and are skipped without it. The 30 slow ones regenerate corpora to re-derive measured
findings.

---

## 4. Figures in Spyder

The library calls `matplotlib.use("Agg", force=False)`. The `force=False` is
deliberate: if Spyder has already set its own backend, that backend is kept and
figures appear inline as usual. You do not need to do anything.

If figures do not appear:

- **Tools → Preferences → IPython console → Graphics → Backend → Inline**
  (or `Automatic` for separate windows)
- Restart the kernel

Every plotting function **returns** a `Figure` and writes nothing. Saving is a
separate, explicit step:

```python
fig = rc.plot_recurrence(rm)      # appears inline in Spyder
rc.save_figure(fig, "rp.png")     # writes to the configured output directory
```

Note that `save_figure` closes the figure by default. Pass `close=False` to
keep it on screen.

---

## 5. Where the output goes

```python
rc.set_config(output_dir="results")     # created if missing, relative to cwd
```

In Spyder the working directory is shown in the top-right corner and is often
not what you expect. Use an absolute path if in doubt:

```python
rc.set_config(output_dir=r"C:\work\recurra_results")
```

Every exported table writes three files: the table, `*_environment.csv` and,
where applicable, `*_provenance.csv`.

---

## 6. Running the examples

The example scripts are ordinary Python files:

```bash
python examples/example_01_pipeline.py
python examples/example_02_statespace.py
python examples/example_03_recurrence.py
python examples/example_04_windowed.py
python examples/example_05_comparability.py
```

To run them all in sequence with a summary of what each produced:

```bash
python examples/run_all.py
```

Nothing is generated until you run something: the package ships with no output
directory. It is created on first use.

`examples/example_06_cookbook.py` is different: it is split into Spyder
**cells** with `# %%` markers. Open it in Spyder and run cells one at a time
with **Ctrl+Enter**, or the whole file with **F5**. Each cell is a short,
self-contained recipe, and the file covers every command and configuration in
the library.

---

## 7. Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: recurra` | wrong interpreter, or kernel started before install | see section 2b: check `sys.executable`, install with `sys.executable -m pip`, restart the kernel |
| Install seemed to work but the import fails | Python 3.9 or older; `requires-python` error missed in the scroll | check `sys.version`, create a 3.10+ environment |
| `RuntimeWarning: _imaging extension was built for another version` | Pillow mixed between conda and pip | `conda install -c conda-forge --force-reinstall pillow`, or a clean environment |
| Long `autoreload of ... failed` tracebacks | Spyder's UMR reloading a broken package | fix that package; the tracebacks are a symptom |
| `BackendUnavailable: backend 'matplotlib' ...` | matplotlib missing | `pip install matplotlib` |
| Figures do not appear | backend set to Agg by something else | **Preferences → IPython console → Graphics → Inline**, restart kernel |
| Edits to the source have no effect | kernel holds the old module | restart the kernel (Ctrl+.) |
| `MemoryBudgetError` | the matrix does not fit the declared budget | `rc.set_config(memory_budget_gb=8)`, or `store="none"`, or window the record |
| Warnings about tail ratio or dominant blocks | the library telling you something real | read them; see `docs/DESIGN_DECISIONS.md` DD-02 and DD-28 |
| Output files not where expected | Spyder's working directory | set an absolute `output_dir` |

### Pillow built for a different version

```
RuntimeWarning: The _imaging extension was built for another version of Pillow
Core version: 9.4.0
Pillow version: 12.3.0
```

followed by `autoreload of PIL.Image failed`. This is a damaged Pillow
install, usually from pip putting a wheel on top of a conda build or the
reverse. The compiled extension and the Python package no longer match.

It does **not** affect the core of the library: numpy, scipy and pandas are
untouched, so thresholds, recurrence structures and metrics all work. But
matplotlib writes PNG through Pillow, so `save_figure` will fail.

```bash
conda install -c conda-forge --force-reinstall pillow
```

If that does not settle it, the environment has mixed package managers more
generally and the quickest route is a clean environment (section 2), letting
conda provide the compiled dependencies and pip provide only `recurra`:

```bash
conda create -n recurra python=3.11 numpy scipy pandas matplotlib pillow spyder-kernels
conda activate recurra
cd path/to/recurra
pip install -e ".[dev]"
```

Mixing the two managers for the same binary package is what causes this in the
first place, so it is worth keeping the split clean.

### Spyder's autoreload is noisy about a broken package

Long `autoreload of ... failed` tracebacks are a *symptom*, not the cause:
Spyder's user module reloader tries to reload a package that cannot import.
Fix the underlying package. To quieten it meanwhile, **Tools → Preferences →
Python interpreter → User Module Reloader → uncheck "Enable UMR"**, then
restart the kernel. Note that with UMR off you must restart the kernel to pick
up edits to the library source.

### Many warnings during a run

Messages such as

```
UserWarning: statespace: phase channel 'phase_theta' chosen as slowest (6 Hz)
UserWarning: statespace: block 'amp_gamma' has a much heavier tail than the others
```

are **not errors**. They are the library reporting what it did without being
asked: which channel it picked, a block that dominates the geometry, records
that are not comparable. During real analysis they are worth reading.

They carry their own category, so you can quieten the library without going
deaf to numpy and scipy [DD-46]:

```python
import warnings
import recurra as rc
warnings.simplefilter("ignore", rc.RecurraWarning)        # all of ours
warnings.simplefilter("ignore", rc.InferenceWarning)      # only one kind
```

The test suite already filters them through `pyproject.toml`, so `pytest`
output is clean.

Turning them off at the source is possible but not recommended, since the
choice stops being announced anywhere except the provenance trail:

```python
rc.set_config(warn_on_caveat=False)
```

---

## 8. Minimal first run

```python
import numpy as np
import recurra as rc
from recurra import statespace as sm

sig = rc.generate_cfc(modality="pac", duration=20.0, fs=500.0, alpha=0.8,
                      preferred_phase=np.pi/2, snr_db=15.0, seed=42)
rec = rc.analytic(rc.filterbank(sig.to_recording("sub01"),
                                {"theta": (4, 8), "gamma": (50, 70)}),
                  sources=["theta", "gamma"])
ss = sm.pac_space(rec, smooth=8.0).subsample(max_points=1500)
th = rc.threshold(ss, target_rr=0.05, theiler=1, rng=0)
rm = rc.recurrence_plot(ss, th, theiler=1)

print(rm.summary())
rc.plot_recurrence(rm)
```
