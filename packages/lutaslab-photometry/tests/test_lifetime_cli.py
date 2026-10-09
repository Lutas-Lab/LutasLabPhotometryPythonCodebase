import pytest

import lutaslab_photometry.cli.run_lifetime_workflow as cli


@pytest.fixture
def workflow_stubs(monkeypatch):
    calls = []
    rows = [
        {
            "mouse": "M1",
            "date": "260101",
            "run": 1,
            "group": "Control",
            "condition": "Baseline",
        },
        {
            "mouse": "M1",
            "date": "260101",
            "run": 2,
            "group": "Control",
            "condition": "Ensure",
        },
    ]
    output = "result.npz"

    monkeypatch.setattr(cli, "read_lifetime_manifest", lambda workflow, manifest: rows)

    def record(name):
        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return [output]

        return fake

    monkeypatch.setattr(cli, "preprocess_lifetime_sessions", record("preprocess"))
    monkeypatch.setattr(cli, "run_lifetime_psth", record("psth"))
    monkeypatch.setattr(cli, "run_lifetime_glm", record("glm"))
    return calls, rows, output


def _common_args(action, workflow="iflip3"):
    return [
        action,
        "--workflow",
        workflow,
        "--manifest",
        "sessions.csv",
        "--data-root",
        "data",
        "--output-dir",
        "output",
    ]


def test_lifetime_cli_dispatches_preprocess(workflow_stubs, capsys):
    calls, rows, output = workflow_stubs

    arguments = [
        "preprocess",
        "--workflow",
        "iflip3",
        "--manifest",
        "sessions.csv",
        "--data-root",
        "data",
        "--overwrite",
    ]
    assert cli.main(arguments) == 0

    name, args, kwargs = calls.pop()
    assert name == "preprocess"
    assert args == ("iflip3", rows, "data")
    assert kwargs == {"overwrite": True, "iflip3_fit_settings": None}
    assert str(output) in capsys.readouterr().out


def test_lifetime_cli_dispatches_psth_options(workflow_stubs):
    calls, rows, _ = workflow_stubs
    arguments = _common_args("psth", workflow="fluopulse") + [
        "--signal",
        "tau",
        "--event",
        "ensure",
        "--window",
        "-2",
        "4",
        "--baseline",
        "-2",
        "0",
        "--dt",
        "0.2",
        "--no-heatmaps",
    ]

    assert cli.main(arguments) == 0

    name, args, kwargs = calls.pop()
    assert name == "psth"
    assert args == ("fluopulse", rows, "data", "output")
    assert kwargs["signal"] == "tau"
    assert kwargs["event"] == "ensure"
    assert kwargs["window"] == (-2.0, 4.0)
    assert kwargs["baseline"] == (-2.0, 0.0)
    assert kwargs["dt"] == 0.2
    assert kwargs["heatmaps"] is False
    assert callable(kwargs["progress_callback"])


def test_lifetime_cli_dispatches_glm_options(workflow_stubs):
    calls, rows, _ = workflow_stubs
    arguments = _common_args("glm") + [
        "--signal",
        "mpet",
        "--lick-kernel-seconds",
        "8",
        "--ensure-kernel-seconds",
        "16",
    ]

    assert cli.main(arguments) == 0

    name, args, kwargs = calls.pop()
    assert name == "glm"
    assert args == ("iflip3", rows, "data", "output")
    assert kwargs == {
        "signal": "mpet",
        "lick_kernel_seconds": 8.0,
        "ensure_kernel_seconds": 16.0,
    }


def test_lifetime_cli_filters_manifest_rows(workflow_stubs):
    calls, rows, _ = workflow_stubs
    arguments = _common_args("glm") + [
        "--signal",
        "mpet",
        "--group",
        "control",
        "--condition",
        "ensure",
    ]

    assert cli.main(arguments) == 0

    name, args, _ = calls.pop()
    assert name == "glm"
    assert args[1] == [rows[1]]


@pytest.mark.parametrize(
    "arguments,message",
    [
        (_common_args("psth") + ["--signal", "mpet"], "PSTH requires"),
        (_common_args("glm"), "GLM requires"),
    ],
)
def test_lifetime_cli_rejects_missing_action_options(workflow_stubs, arguments, message):
    with pytest.raises(ValueError, match=message):
        cli.main(arguments)
