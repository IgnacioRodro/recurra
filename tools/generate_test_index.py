"""Generate docs/TESTS.md from the test files themselves.

    python tools/generate_test_index.py

Writing a 250-entry index by hand guarantees it will be wrong within a week
[DD-45], so it is derived: the ordering is pytest's own (files alphabetically,
tests in declaration order), the descriptions come from each test's docstring,
and parametrised tests are marked with their case count. Only the per-file
commentary below is written by hand.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# pytest collects files in alphabetical order, and tests in declaration order
# within each file. Reproducing that here needs no subprocess.
files = [f"tests/{p.name}" for p in sorted(TESTS.glob("test_*.py"))]

FILE_PURPOSE = {
    "test_capabilities.py": ("Capability system", "DD-03",
        "That a stage refuses to run without the data it needs, and says which capability is missing."),
    "test_recording.py": ("The Recording object", "DD-07, DD-04",
        "Copy-on-write sharing, read-only buffers, provenance accumulation."),
    "test_ingest.py": ("Ingestion", "DD-08, DD-06",
        "The input shapes, role assignment, quality checks, and the errors that name their own fix."),
    "test_preprocess.py": ("Filtering and Hilbert", "DD-11, DD-12",
        "Filter order from cycle counts, band isolation, edge policies, and the full chain end to end."),
    "test_scaling.py": ("The scaling policy", "DD-02",
        "The analytic result the default rests on, and that the balance holds across five orders of magnitude of gain."),
    "test_signals.py": ("The synthetic generator", "DD-13, DD-48",
        "That the generated coupling is monotone in alpha against the canonical indices, and reproducible."),
    "test_statespace.py": ("State space construction", "DD-21, DD-23, DD-25, DD-28",
        "The weight-folding identity, every constructor on all three routes, embedding estimation, scaling and transforms."),
    "test_threshold.py": ("Threshold selection", "DD-32, DD-35",
        "That every mode runs, that the target rate is met, and that Theiler-aware sampling behaves."),
    "test_recurrence.py": ("Recurrence structures", "DD-19, DD-20, DD-33, DD-34, DD-36",
        "Distance kernels, streaming equivalence bit for bit, Theiler bands, RP/CRP/JRP, memory refusal."),
    "test_windowed.py": ("Windowed analysis", "DD-39, DD-40, DD-41",
        "Window specification, threshold scope, laziness, every mode windowed, and the multi-record batch."),
    "test_comparability.py": ("Comparability and groups", "DD-42, DD-43, DD-44",
        "That the three silent failure modes are detected, and that groups separate window by window."),
    "test_viz.py": ("Figures", "DD-27, DD-30, DD-31, DD-38, DD-49..52",
        "That every registered figure renders from one object, that inputs are not mutated, and the language guards."),
    "test_export.py": ("CSV export", "DD-10",
        "That every table gets its provenance and environment sidecars."),
    "test_report.py": ("Run reports", "DD-37",
        "Sections, artefact registration, rendering, and that the report is in English."),
    "test_docs.py": ("The living documents", "DD-45",
        "That the documents have not drifted from the code: decision index, citations, module map, figure catalogue, counts, API coverage."),
}

lines = ['<picture><source media="(prefers-color-scheme: dark)" srcset="_static/recurra-mark-dark.svg">'
         '<img alt="recurra" src="_static/recurra-mark.svg" width="56" align="right"></picture>\n\n' + """# Test index

**Generated from the test files.** Regenerate with `python tools/generate_test_index.py`.

Ordered as pytest collects them: files alphabetically, tests in declaration
order within each file.

## Tests do not produce figures

Worth stating plainly, because it is a natural assumption. The suite renders
figures in memory to check that they draw, and closes them; it writes **no
files at all**. Every PNG and CSV comes from the example scripts:

```bash
python examples/run_all.py        # 54 figures, 45 tables, 3 run reports
```

| Example | Produces |
|---|---|
| `example_01_pipeline.py` | signal, phase-amplitude and quality figures; generator validation and corpus tables |
| `example_02_statespace.py` | attractor figures for every constructor, route comparison, embedding curves |
| `example_03_recurrence.py` | recurrence, cross- and joint-recurrence plots, density profiles, threshold diagnostics |
| `example_04_windowed.py` | window coverage and series, per-window recurrence plots |
| `example_05_comparability.py` | comparability reports, group comparison, feature matrix |
| `example_06_cookbook.py` | 39 figures into `cookbook_out/`, one per recipe |

---

## Summary
"""]
lines.append("| # | File | Covers | Decisions | Tests |")
lines.append("|---|---|---|---|---|")
total = 0
for i, rel in enumerate(files, 1):
    name = pathlib.Path(rel).name
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    n = sum(1 for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name.startswith("test_"))
    total += n
    purpose, dds, _ = FILE_PURPOSE.get(name, (name, "", ""))
    lines.append(f"| {i} | `{name}` | {purpose} | {dds} | {n} |")
lines.append(f"| | | | **total** | **{total} functions** |")
lines.append("")
lines.append("Parametrised tests expand to more cases; run `pytest -q` for the count.")
lines.append("")
lines.append("---")
lines.append("")

for i, rel in enumerate(files, 1):
    name = pathlib.Path(rel).name
    src = (ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    purpose, dds, blurb = FILE_PURPOSE.get(name, (name, "", ""))
    lines.append(f"## {i}. `{name}` — {purpose}")
    lines.append("")
    if blurb:
        lines.append(blurb)
        lines.append("")
    if dds:
        lines.append(f"Decisions: {dds}")
        lines.append("")
    lines.append("| Test | What it checks |")
    lines.append("|---|---|")
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        doc = ast.get_docstring(node)
        if doc:
            desc = " ".join(doc.split())
        else:
            desc = node.name.replace("test_", "").replace("_", " ")
            desc = desc[0].upper() + desc[1:] + "."
        params = [d for d in node.decorator_list
                  if isinstance(d, ast.Call) and "parametrize" in ast.dump(d)]
        marker = ""
        if params:
            try:
                cases = len(params[0].args[1].elts)
                marker = f" _(×{cases})_"
            except Exception:
                marker = " _(parametrised)_"
        desc = desc.replace("|", "\\|")
        if len(desc) > 200:
            desc = desc[:197] + "..."
        lines.append(f"| `{node.name}`{marker} | {desc} |")
    lines.append("")

(ROOT / "docs" / "TESTS.md").write_text("\n".join(lines), encoding="utf-8")
print("written, lines:", len(lines))
