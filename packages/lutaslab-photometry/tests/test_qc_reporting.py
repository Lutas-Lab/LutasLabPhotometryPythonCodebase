import warnings

from lutaslab_core.nidaq import RunningDataExcludedWarning

from lutaslab_photometry.qc_reporting import (
    concise_qc_warnings,
    parse_qc_warning_line,
)


def test_expected_running_warning_is_encoded_without_python_location(capsys):
    with concise_qc_warnings():
        warnings.warn(
            RunningDataExcludedWarning(
                "Running speed is a flat line at zero; running data were excluded.",
                r"Z:\Photometry\M1\running.mat",
            ),
            stacklevel=1,
        )

    captured = capsys.readouterr()
    record = parse_qc_warning_line(captured.out.strip())
    assert record == {
        "kind": "running_data_excluded",
        "reason": (
            "Running speed is a flat line at zero; running data were excluded."
        ),
        "path": r"Z:\Photometry\M1\running.mat",
    }
    assert captured.err == ""
    assert "test_qc_reporting.py" not in captured.out


def test_unexpected_warning_keeps_standard_diagnostic_output(capsys):
    with concise_qc_warnings():
        warnings.warn("unexpected diagnostic", UserWarning, stacklevel=1)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "UserWarning: unexpected diagnostic" in captured.err
    assert "test_qc_reporting.py" in captured.err


def test_malformed_or_unrecognized_qc_lines_remain_console_output():
    assert parse_qc_warning_line("ordinary output") is None
    assert parse_qc_warning_line("QC_WARNING\tnot json") is None
    assert parse_qc_warning_line('QC_WARNING\t{"kind":"other"}') is None
