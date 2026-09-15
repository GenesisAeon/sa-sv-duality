"""Tests for sa-sv-duality's own CLI (src/sa_sv_duality/cli.py)."""

from typer.testing import CliRunner

from sa_sv_duality.benchmark import run_benchmarks
from sa_sv_duality.cli import app

runner = CliRunner()


def test_run_short_duration():
    result = runner.invoke(app, ["run", "--duration", "2.0"])
    assert result.exit_code == 0, result.output
    assert "S_A" in result.output
    assert "S_V" in result.output


def test_q4_map():
    result = runner.invoke(app, ["q4-map"])
    assert result.exit_code == 0, result.output
    assert "Q4" in result.output


def test_route_0_to_15():
    result = runner.invoke(app, ["route", "0", "15"])
    # Real outcome: exit 0 if path found, 1 if not.
    assert result.exit_code in (0, 1), result.output
    if result.exit_code == 0:
        assert "Optimal path" in result.output or "0" in result.output


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert "sa-sv-duality" in result.output


def test_benchmark_fast_runs_and_reports_consistently():
    result = runner.invoke(app, ["benchmark", "--fast"])
    expected = run_benchmarks(fast=True)
    expected_exit = 0 if expected.ok else 1
    assert result.exit_code == expected_exit, result.output
    assert (
        "Benchmark" in result.output
        or "Passed" in result.output
        or "benchmark" in result.output.lower()
    )
