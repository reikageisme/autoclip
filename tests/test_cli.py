from __future__ import annotations

from typer.testing import CliRunner

from autoclip.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("ingest", "capcut-report", "extract", "spike", "concat-tools"):
        assert name in result.output


def test_capcut_report_on_empty_folder(tmp_path):
    out = tmp_path / "report.md"
    result = runner.invoke(app, ["capcut-report", str(tmp_path), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_extract_needs_target():
    result = runner.invoke(app, ["extract"])
    assert result.exit_code != 0
