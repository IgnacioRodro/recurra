"""The command-line interface: every command runs and produces what it says."""
import pathlib

import pandas as pd

from recurra.cli import main


def test_doctor_lists_the_core_and_the_optional_backends(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "joblib"):
        assert name in out


def test_figures_lists_the_catalogue(capsys):
    assert main(["figures"]) == 0
    out = capsys.readouterr().out
    assert "plot_recurrence" in out and "comparison" in out.lower()


def test_generate_writes_the_signals_and_their_ground_truth(tmp_path, capsys):
    """Before 0.25 it wrote only the ground-truth table and discarded the
    signals it had generated."""
    out = tmp_path / "corpus"
    assert main(["generate", "--n", "3", "--duration", "2", "--out", str(out)]) == 0
    truth = pd.read_csv(out / "corpus_truth.csv")
    assert len(truth) == 3 and list(truth["signal"]) == [
        "signal_000", "signal_001", "signal_002"]
    series = sorted(pathlib.Path(out, "signals").rglob("*_timeseries.csv"))
    assert len(series) == 3
    first = pd.read_csv(series[0])
    assert len(first) == 2 * 500          # duration x the default fs
