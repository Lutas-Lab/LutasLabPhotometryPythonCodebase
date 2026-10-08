import sys
import tomllib
from importlib import import_module
from pathlib import Path

import pytest

EXPECTED_COMMANDS = {
    "lutaslab-build-figure4-dopamine-input",
    "lutaslab-check-install",
    "lutaslab-compare-figure5-reanalysis",
    "lutaslab-fit-dopamine-pka-transfer",
    "lutaslab-plot-figure5-reanalysis",
    "lutaslab-run-figure5-bout-analysis",
    "lutaslab-run-figure5-reanalysis",
    "lutaslab-run-forecasting",
    "lutaslab-run-lickbout-delivery-analysis",
    "lutaslab-run-lifetime-workflow",
    "lutaslab-run-multitastant-analysis",
    "lutaslab-run-preprocess",
    "lutaslab-run-preprocess-batch",
    "lutaslab-run-psth",
    "lutaslab-run-psth-statistics",
    "lutaslab-validate-figure5-glm",
    "lutaslab-validate-notebooks",
}


def _declared_commands():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    with pyproject.open("rb") as stream:
        return tomllib.load(stream)["project"]["scripts"]


def test_expected_console_commands_are_declared():
    assert set(_declared_commands()) == EXPECTED_COMMANDS


@pytest.mark.parametrize("command", sorted(EXPECTED_COMMANDS))
def test_console_command_starts_and_handles_help(command, monkeypatch, capsys):
    module_name, function_name = _declared_commands()[command].split(":", maxsplit=1)
    main = getattr(import_module(module_name), function_name)

    if command == "lutaslab-check-install":
        main()
    else:
        monkeypatch.setattr(sys, "argv", [command, "--help"])
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 0

    output = capsys.readouterr()
    assert output.out or output.err
