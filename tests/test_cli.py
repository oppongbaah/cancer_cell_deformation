"""Tests for celldform.cli — one console command per non-HPC script, kept in sync with pyproject.toml."""

import sys
import tomllib
from pathlib import Path

import pytest

from celldform import cli

REPO = Path(__file__).resolve().parent.parent
EXCLUDED = {"hpc", "maintenance"}   # SLURM job scripts; one-off migrations
# Figure scripts written for the thesis text: they are run by path (python scripts/report/<name>.py), not installed as
# commands, so a public install does not gain a dozen thesis-only commands. Every other script must have a command.
RUN_BY_PATH = ("*_thesis.py", "*_ch4.py")


def _all_commands() -> dict[str, str]:
    return {c: rel for group in cli.SCRIPT_COMMANDS.values() for c, rel in group.items()}


# ---------------------------------------------------------------------------
# The command table matches scripts/
# ---------------------------------------------------------------------------

def test_every_script_has_a_command():
    import fnmatch
    scripts = {str(p.relative_to(REPO / "scripts")) for p in (REPO / "scripts").rglob("*.py")
               if p.relative_to(REPO / "scripts").parts[0] not in EXCLUDED
               and not any(fnmatch.fnmatch(p.name, pat) for pat in RUN_BY_PATH)}
    covered = set(_all_commands().values()) | {"data/organize_dataset.py"}   # built-in command
    assert scripts - covered == set(), "scripts without a celldform-* command (add them to SCRIPT_COMMANDS)"


def test_every_command_points_at_an_existing_script():
    missing = {c: rel for c, rel in _all_commands().items() if not (REPO / "scripts" / rel).is_file()}
    assert missing == {}


def test_command_names_follow_the_script_names():
    for c, rel in _all_commands().items():
        assert c == Path(rel).stem.replace("_", "-")


# ---------------------------------------------------------------------------
# pyproject.toml registers exactly these commands
# ---------------------------------------------------------------------------

def test_pyproject_entry_points_match_the_table():
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    expected = {f"celldform-{c}": f"celldform.cli:{c.replace('-', '_')}" for c in _all_commands()}
    expected |= {"celldform": "celldform.cli:main",
                 "celldform-organize-dataset": "celldform.cli:organize_dataset",
                 "celldform-hpc-submit": "celldform.cli:hpc_submit",
                 "celldform-demo": "celldform.cli:demo",
                 "celldform-shortcut": "celldform.cli:shortcut"}
    assert scripts == expected


def test_every_entry_point_resolves_to_a_callable():
    scripts = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["scripts"]
    for name, target in scripts.items():
        assert callable(getattr(cli, target.split(":")[1])), name


# ---------------------------------------------------------------------------
# Running a script command
# ---------------------------------------------------------------------------

def test_run_script_passes_arguments_and_exit_code():
    # summarize_cv.py needs only the standard library: --help exits 0, missing required args exit 2
    assert cli.run_script("evaluation/summarize_cv.py", ["--help"]) == 0
    assert cli.run_script("evaluation/summarize_cv.py", []) == 2


def test_run_script_reports_a_missing_script(capsys):
    assert cli.run_script("training/no_such_script.py", []) == 1
    assert "pip install -e ." in capsys.readouterr().err


def test_umbrella_lists_every_command(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["celldform"])
    cli.main()
    out = capsys.readouterr().out
    for c in [*_all_commands(), "organize-dataset", "hpc-submit"]:
        assert f"  {c} " in out


def test_umbrella_rejects_an_unknown_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["celldform", "no-such-command"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert "unknown command" in str(e.value)


def test_umbrella_runs_a_script_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["celldform", "summarize-cv", "--help"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 0


def test_every_script_parses_its_arguments():
    # `celldform-<command> --help` must print help, not run the script: a script without an
    # argument parser ignores --help and does its full job (eval_ensemble_cv.py did, until 2026-10-07).
    no_parser = [rel for rel in _all_commands().values()
                 if "ArgumentParser" not in (REPO / "scripts" / rel).read_text()]
    assert no_parser == []


def test_every_command_is_documented():
    doc = (REPO / "docs" / "cli.md").read_text()
    undocumented = [c for c in [*_all_commands(), "organize-dataset", "hpc-submit"]
                    if f"### `celldform-{c}`" not in doc]
    assert undocumented == [], "add a section to docs/cli.md"


def test_the_run_by_path_scripts_are_exactly_the_thesis_figure_scripts():
    """The exemption must not quietly swallow a real tool: only report/ figure scripts may use it."""
    import fnmatch
    exempt = [p.relative_to(REPO / "scripts") for p in (REPO / "scripts").rglob("*.py")
              if any(fnmatch.fnmatch(p.name, pat) for pat in RUN_BY_PATH)]
    assert exempt and all(p.parts[0] == "report" for p in exempt), exempt
