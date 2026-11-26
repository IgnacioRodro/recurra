"""The living documents must stay in step with the code.

Design note (DD-45). Three documents are maintained alongside the library:
design decisions, architecture and examples. Keeping them current by
discipline alone does not work -- they drift the moment a phase lands and
nobody notices until someone reads a stale claim. So the parts that *can* be
checked mechanically are checked here:

* every decision cited in the source exists in the decisions document
* the decisions index and the decision sections agree
* every module in the package appears in the architecture map
* every registered figure appears in the figure catalogue
* the test count quoted in the documents matches reality
* the version agrees across the package, the changelog and the citation file
* every example script the README advertises exists and is runnable

This cannot verify that the prose is *true*. It can verify that it is not
obviously stale, which is where the rot starts.
"""
from __future__ import annotations

import pathlib
import re

import pytest

import recurra
from recurra.viz import COMPARISON_FIGURES, SINGLE_FIGURES

ROOT = pathlib.Path(recurra.__file__).parent.parent.parent
DOCS = ROOT / "docs"
SRC = pathlib.Path(recurra.__file__).parent

DECISIONS = DOCS / "DESIGN_DECISIONS.md"
ARCHITECTURE = DOCS / "ARCHITECTURE.md"
EXAMPLES = DOCS / "EXAMPLES.md"
README = ROOT / "README.md"
CITATION = ROOT / "CITATION.cff"


def _read(path: pathlib.Path) -> str:
    if not path.exists():
        pytest.skip(f"{path.name} not present in this layout")
    return path.read_text(encoding="utf-8")


# ------------------------------------------------------------- decisions
def test_decision_index_matches_sections():
    text = _read(DECISIONS)
    indexed = set(re.findall(r"^\| \[(DD-\d+)\]", text, re.M))
    anchored = {f"DD-{n}" for n in re.findall(r'^<a id="dd-(\d+)">', text, re.M)}
    assert indexed == anchored, (
        f"index and sections disagree; only in index: {sorted(indexed - anchored)}; "
        f"only as sections: {sorted(anchored - indexed)}"
    )


def test_decisions_are_numbered_without_gaps():
    text = _read(DECISIONS)
    numbers = sorted(int(n) for n in re.findall(r'^<a id="dd-(\d+)">', text, re.M))
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"decision numbering has gaps or duplicates: {numbers}"
    )


def test_every_decision_cited_in_source_exists():
    """A dangling [DD-nn] in a docstring is a promise the documents do not keep."""
    text = _read(DECISIONS)
    documented = {f"DD-{n}" for n in re.findall(r'^<a id="dd-(\d+)">', text, re.M)}
    cited: dict[str, set[str]] = {}
    for path in SRC.rglob("*.py"):
        for ref in re.findall(r"DD-(\d+)", path.read_text(encoding="utf-8")):
            cited.setdefault(f"DD-{ref}", set()).add(path.name)
    dangling = {k: sorted(v) for k, v in cited.items() if k not in documented}
    assert not dangling, f"source cites undocumented decisions: {dangling}"


def test_major_decisions_are_cited_somewhere_in_the_source():
    """A decision about the code should be traceable from the code."""
    text = _read(DECISIONS)
    documented = {f"DD-{n}" for n in re.findall(r'^<a id="dd-(\d+)">', text, re.M)}
    source = "\n".join(p.read_text(encoding="utf-8") for p in SRC.rglob("*.py"))
    tests = "\n".join(p.read_text(encoding="utf-8")
                      for p in (ROOT / "tests").rglob("*.py"))
    # Decisions about process rather than code need not appear in modules.
    # Decisions about packaging, process or documents, not about a module.
    process_only = {"DD-01", "DD-15", "DD-16", "DD-17", "DD-18", "DD-45",
                    "DD-60"}   # a measured result, not a rule the code follows
    orphans = sorted(d for d in documented - process_only
                     if d not in source and d not in tests)
    assert not orphans, (
        f"decisions with no trace in code or tests: {orphans}. Either cite them "
        "where they apply or mark them as process-only."
    )


# ---------------------------------------------------------- architecture
def test_every_module_appears_in_the_architecture_map():
    text = _read(ARCHITECTURE)
    missing = []
    for path in sorted(SRC.rglob("*.py")):
        name = path.name
        if name.startswith("_") and name != "_version.py":
            continue
        if name == "__init__.py":
            continue
        if name not in text:
            missing.append(str(path.relative_to(SRC)))
    assert not missing, f"modules absent from the architecture map: {missing}"


def _count_test_functions() -> int:
    """Declared test functions. Unambiguous, unlike expanded parametrisation."""
    return sum(len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), re.M))
               for p in sorted((ROOT / "tests").rglob("test_*.py")))


def test_architecture_quotes_the_real_test_count():
    """A stale count is the first sign a document stopped being maintained.

    The count is written in two places, the architecture document and the
    README's status line, and both are checked. Nowhere else quotes one: the
    README, INSTALL and ROADMAP once carried counts several versions old,
    because nothing checked them.
    """
    actual = _count_test_functions()
    for path in (ARCHITECTURE, README):
        text = _read(path)
        quoted = re.search(r"\*\*(\d+) test functions, all green\*\*", text)
        assert quoted, f"{path.name} must state '**N test functions, all green**'"
        assert int(quoted.group(1)) == actual, (
            f"{path.name} says {quoted.group(1)} test functions, the suite "
            f"declares {actual}. Update {path.name}."
        )


def test_figure_catalogue_lists_every_registered_figure():
    text = _read(ARCHITECTURE)
    for name in SINGLE_FIGURES:
        assert f"plot_{name}" in text, (
            f"single figure plot_{name} is registered but missing from the "
            "architecture catalogue"
        )
    for name in COMPARISON_FIGURES:
        assert f"compare_{name}" in text, (
            f"comparison figure compare_{name} is registered but missing from the "
            "architecture catalogue"
        )
    # The README's status table quoted 17 single figures long after there were 22.
    quoted = re.search(r"Figures: (\d+) single \+ (\d+) comparison", _read(README))
    assert quoted, "README.md must state 'Figures: N single + M comparison'"
    assert (int(quoted.group(1)), int(quoted.group(2))) == (
        len(SINGLE_FIGURES), len(COMPARISON_FIGURES)), (
        f"README.md says {quoted.group(1)} single and {quoted.group(2)} comparison "
        f"figures; {len(SINGLE_FIGURES)} and {len(COMPARISON_FIGURES)} are registered")


# --------------------------------------------------------------- examples
def test_example_scripts_referenced_in_the_readme_exist():
    readme = _read(ROOT / "README.md")
    for name in re.findall(r"(example_\d+_\w+\.py)", readme):
        assert (ROOT / "examples" / name).exists(), (
            f"the README advertises {name}, which does not exist"
        )


def test_every_example_script_is_referenced():
    readme = _read(ROOT / "README.md")
    scripts = sorted(p.name for p in (ROOT / "examples").glob("example_*.py"))
    missing = [s for s in scripts if s not in readme]
    assert not missing, f"example scripts nobody is told about: {missing}"


def test_examples_document_covers_the_public_api():
    """Every public entry point should appear somewhere in the examples."""
    text = _read(EXAMPLES)
    undocumented = []
    for name in recurra.__all__:
        if name.startswith("_") or name in {"__version__"}:
            continue
        if name[0].isupper() and name.endswith("Error"):
            continue
        if name not in text:
            undocumented.append(name)
    assert not undocumented, (
        f"public API absent from docs/EXAMPLES.md: {sorted(undocumented)}"
    )


# ------------------------------------------------------------- changelog
def test_changelog_mentions_the_current_version():
    """The newest heading says what the version is: a development version is
    `unreleased`, a release carries the date it was tagged."""
    text = _read(ROOT / "CHANGELOG.md")
    newest = re.search(r"^## \[(\d+\.\d+\.\d+(?:\.dev\d+)?)\] — (.+)$", text, re.M)
    assert newest, "no version heading of the form '## [X.Y.Z] — ...' found"
    version, label = newest.group(1), newest.group(2).strip()
    if ".dev" in version:
        assert label.startswith("unreleased"), (
            f"{version} is a development version, so its heading should say "
            f"'unreleased', not {label!r}")
    else:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", label), (
            f"release {version} should carry its date as YYYY-MM-DD, not {label!r}")


# --------------------------------------------------------------- cookbook
def test_cookbook_exercises_the_whole_public_api():
    """The cookbook is the acceptance test for the API surface [DD-45]."""
    path = ROOT / "examples" / "example_06_cookbook.py"
    if not path.exists():
        pytest.skip("cookbook not present")
    text = path.read_text(encoding="utf-8")
    missing = []
    for name in recurra.__all__:
        if name.startswith("_") or name == "__version__":
            continue
        if name not in text:
            missing.append(name)
    assert not missing, (
        f"public API not exercised by the cookbook: {sorted(missing)}"
    )


def test_cookbook_is_split_into_spyder_cells():
    path = ROOT / "examples" / "example_06_cookbook.py"
    if not path.exists():
        pytest.skip("cookbook not present")
    text = path.read_text(encoding="utf-8")
    cells = re.findall(r"^# %% (\d+) --", text, re.M)
    assert len(cells) >= 50, f"only {len(cells)} numbered cells"
    numbers = [int(c) for c in cells]
    assert numbers == sorted(numbers), "cell numbers are out of order"
    assert len(set(numbers)) == len(numbers), "duplicate cell numbers"


def test_test_index_is_current():
    """docs/TESTS.md is generated; a stale one is worse than none [DD-45]."""
    index = ROOT / "docs" / "TESTS.md"
    if not index.exists():
        pytest.skip("test index not generated")
    text = index.read_text(encoding="utf-8")
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        assert path.name in text, f"{path.name} missing from the index"
        for name in re.findall(r"^def (test_\w+)", path.read_text(encoding="utf-8"), re.M):
            assert f"`{name}`" in text, (
                f"{name} missing from docs/TESTS.md -- regenerate with "
                "python tools/generate_test_index.py"
            )


# ------------------------------------------------------- example index
def test_example_index_lists_every_script():
    """docs/EXAMPLE_INDEX.md must cover each example [DD-45]."""
    index = ROOT / "docs" / "EXAMPLE_INDEX.md"
    if not index.exists():
        pytest.skip("example index not present")
    text = index.read_text(encoding="utf-8")
    for path in sorted((ROOT / "examples").glob("example_*.py")):
        assert path.name in text, f"{path.name} missing from the example index"


def test_example_index_names_the_figures_the_scripts_save():
    """Every filename an example writes must be described in the index.

    Catches the drift that matters: a figure added to a script and never
    documented, or one documented after being removed.
    """
    index = ROOT / "docs" / "EXAMPLE_INDEX.md"
    if not index.exists():
        pytest.skip("example index not present")
    text = index.read_text(encoding="utf-8")

    saved, described = set(), set()
    for path in sorted((ROOT / "examples").glob("example_*.py")):
        src = path.read_text(encoding="utf-8")
        saved |= set(re.findall(r'"(fig_[a-z0-9_]+\.png)"', src))
        saved |= set(re.findall(r'f"(fig_[a-z0-9_]+)_?\{', src))
        saved |= set(re.findall(r'export_frame\([^,]+,\s*"([a-z0-9_]+)"', src))
        saved |= set(re.findall(r'\.export\([^,]+,\s*"([a-z0-9_]+)"', src))

    for name in sorted(saved):
        stem = name.replace(".png", "")
        if stem in text or name in text:
            described.add(name)
    missing = sorted(saved - described)
    assert not missing, (
        f"produced by an example but absent from docs/EXAMPLE_INDEX.md: {missing}"
    )


def test_documented_metrics_all_exist():
    """A metric named in the documents must exist in the code [DD-45].

    This check exists because it did not. Two implementations of the RQA layer
    both passed their own suites while exporting different metric sets: one
    with RTE, one with RATIO. The design document reported measured results for
    RTE, so the version without it contradicted its own documentation and
    nothing noticed. Naming a metric in prose is a claim that it exists.
    """
    from recurra.rqa.metrics import METRIC_NAMES

    known = set(METRIC_NAMES) | {
        # units and axis labels that appear in the leftmost column of a table
        "alpha", "lambda", "seconds", "metric",
        # specified for later phases, named in the documents as planned work
        "D2", "lambda_1", "K2", "TREND", "T1", "T2", "CLEAR",
        "transitivity", "clustering",
        # produced elsewhere in the library
        "independence_ratio", "recurrence_rate", "epsilon",
        # canonical coupling indices, from recurra.metrics.classic
        "MVL", "MI", "PLV", "AAC",
    }
    # A metric name is at least two characters and not a bare axis label such
    # as the "N" heading of a sample-size row.
    pattern = re.compile(r"^\| ?\**([A-Za-z_][A-Za-z0-9_]{1,})\**"
                         r" ?\| ?\**[\d.]+\**", re.M)
    for name in ("DESIGN_DECISIONS.md", "ARCHITECTURE.md", "EXAMPLES.md"):
        path = DOCS / name
        if not path.exists():
            continue
        for metric in set(pattern.findall(path.read_text(encoding="utf-8"))):
            if metric.isupper() or metric in known or "_" in metric:
                assert metric in known, (
                    f"{name} reports a result for {metric!r}, which is not a "
                    f"metric the code produces. Known: {sorted(known)}"
                )


def test_the_version_is_not_stale():
    """It sat at 0.1.0.dev0 through nineteen delivered builds. An installed
    copy could not be told apart from any other, which is exactly what a
    version string exists to prevent."""
    import re

    import recurra as rc

    assert re.fullmatch(r"\d+\.\d+\.\d+(\.dev\d+)?", rc.__version__)
    assert rc.__version__ != "0.1.0.dev0", (
        "the version has not moved since the first build")
    # and the changelog's newest entry must name it
    changelog = pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8")
    newest = re.search(r"^## \[([^\]]+)\]", text, re.M)
    assert newest and newest.group(1) == rc.__version__, (
        f"CHANGELOG heads at {newest.group(1) if newest else None} but the "
        f"package says {rc.__version__}")
    # and the citation file must name the latest release, with its date: it
    # said 0.25.0.dev0 with nothing checking it. Between releases it keeps
    # naming the last one, which is what a citation should point to.
    citation = changelog.parent / "CITATION.cff"
    if not citation.exists():
        pytest.skip("CITATION.cff not present in this layout")
    released = re.search(r"^## \[(\d+\.\d+\.\d+)\] — (\d{4}-\d{2}-\d{2})\s*$",
                         text, re.M)
    assert released, "the changelog has no dated release heading"
    cff = citation.read_text(encoding="utf-8")
    cited = re.search(r'^version:\s*"?([^"\n]+?)"?\s*$', cff, re.M)
    dated = re.search(r'^date-released:\s*"?([^"\n]+?)"?\s*$', cff, re.M)
    assert cited and cited.group(1) == released.group(1), (
        f"CITATION.cff cites {cited.group(1) if cited else None}; the latest "
        f"release in the changelog is {released.group(1)}")
    assert dated and dated.group(1) == released.group(2), (
        f"CITATION.cff dates the release {dated.group(1) if dated else None}; "
        f"the changelog says {released.group(2)}")


def test_install_lines_pin_the_latest_release():
    """README and INSTALL tell people which tag to install. In 0.25 they named
    the previous one for a while, with nothing to notice."""
    text = _read(ROOT / "CHANGELOG.md")
    released = re.search(r"^## \[(\d+\.\d+\.\d+)\] — \d{4}-\d{2}-\d{2}\s*$", text, re.M)
    assert released, "the changelog has no dated release heading"
    latest = released.group(1)
    for path in (README, ROOT / "docs" / "INSTALL.md"):
        pins = set(re.findall(r"recurra\.git@v(\d+\.\d+\.\d+)", _read(path)))
        assert pins <= {latest}, (
            f"{path.name} pins v{sorted(pins)}, the latest release is v{latest}")
    status = re.search(r"latest release (\d+\.\d+\.\d+)", _read(README))
    assert status and status.group(1) == latest, "README status line is stale"
