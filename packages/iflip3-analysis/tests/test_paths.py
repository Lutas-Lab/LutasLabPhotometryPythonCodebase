from pathlib import Path

import pytest
from iflip3.paths import session_paths


def test_session_paths_use_laboratory_data_roots():
    paths = session_paths("AL164", "260923", 4)

    assert paths.iflip == Path("Z:/FLIM FLIP/AL164/AL164_260923/AL164_260923_004.iFLiP3")
    assert paths.iflip_legacy == Path("Z:/FLIM FLIP/AL164/AL164_260923/AL164_260923004.iFLiP3")
    assert paths.nidaq == Path("Z:/Photometry/AL164/AL164_260923/AL164-260923-004-nidaq.mat")


def test_session_paths_fall_back_to_an_existing_legacy_iflip(monkeypatch):
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: path.name == "AL164_260923004.iFLiP3",
    )
    paths = session_paths("AL164", "260923", 4)
    assert paths.iflip == paths.iflip_legacy
    assert paths.running == Path("Z:/Photometry/AL164/AL164_260923/AL164-260923-004-running.mat")


def test_session_paths_support_an_explicit_root_and_preserve_date_text():
    paths = session_paths("M1", "001234", 2, data_root="data-root")
    assert paths.nidaq == Path("data-root/Photometry/M1/M1_001234/M1-001234-002-nidaq.mat")


@pytest.mark.parametrize("mouse,date", [("", "260101"), ("M/1", "260101")])
def test_session_paths_reject_invalid_identity(mouse, date):
    with pytest.raises(ValueError):
        session_paths(mouse, date, 1)


@pytest.mark.parametrize("run", [True, -1, 1.5, "not-a-run"])
def test_session_paths_reject_invalid_run(run):
    with pytest.raises(ValueError, match="run must be a nonnegative integer"):
        session_paths("M1", "260101", run)
