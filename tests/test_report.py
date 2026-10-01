"""Tests for the run report [DD-37]."""
import os

import pandas as pd
import pytest

import recurra as rc
from recurra.report import RunReport


@pytest.fixture
def report(tmp_path):
    rc.set_config(output_dir=str(tmp_path))
    return RunReport("unit", title="unit test run")


def test_sections_and_notes(report):
    report.section("first").note("did a thing").result("count", 3)
    report.section("second").note("did another")
    text = report.render()
    assert "1. first" in text and "2. second" in text
    assert "did a thing" in text and "count: 3" in text


def test_export_registers_the_table(report, tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    path = report.section("s").export(df, "t", "a table")
    assert os.path.exists(path)
    assert "3 rows x 2 cols" in report.render()
    assert "a table" in report.render()


@pytest.mark.requires("matplotlib")
def test_figure_registration(report, tmp_path):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    report.section("s").save(fig, "f.png", "a figure")
    plt.close(fig)
    assert "f.png" in report.render()
    assert "FIGURES WRITTEN (1)" in report.render()


def test_warnings_are_collected(report):
    report.warn("something to know about")
    assert "something to know about" in report.render()
    assert "WARNINGS AND CAVEATS (1)" in report.render()


def test_write_produces_txt_and_index(report, tmp_path):
    report.section("s").export(pd.DataFrame({"a": [1]}), "t")
    path = report.write()
    assert path.endswith(".txt") and os.path.exists(path)
    assert os.path.exists(tmp_path / "unit_artifact_index.csv")
    content = open(path, encoding="utf-8").read()
    assert "UNIT TEST RUN" in content
    assert "recurra version" in content


def test_report_records_parameters(report):
    report.parameter(fs=500.0, duration=30.0)
    text = report.render()
    assert "PARAMETERS" in text and "fs" in text and "500.0" in text


def test_artifact_frame_columns(report):
    report.section("s").export(pd.DataFrame({"a": [1]}), "t", "desc")
    df = report.artifact_frame()
    assert {"section", "kind", "name", "rows", "description", "path"} <= set(df.columns)


def test_report_text_is_english(report):
    import re

    report.section("build").note("built a state space").export(
        pd.DataFrame({"a": [1]}), "t", "table of things")
    assert not re.search(r"[\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1]", report.render())
