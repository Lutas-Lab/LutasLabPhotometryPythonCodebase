from pathlib import Path

from fluopulse_analysis.paths import (
    find_doric_files,
    infer_session_identity,
    nidaq_paths_for_doric,
)


def test_infer_identity_and_nidaq_paths():
    source = Path("260923_SC81_run6_0000.doric")
    identity = infer_session_identity(source)
    assert (identity.mouse, identity.date, identity.run) == ("SC81", "260923", 6)
    paths = nidaq_paths_for_doric(source)
    assert paths.nidaq == Path(
        "Z:/Photometry/SC81/SC81_260923/SC81-260923-006-nidaq.mat"
    )
    assert paths.running == Path(
        "Z:/Photometry/SC81/SC81_260923/SC81-260923-006-running.mat"
    )


def test_find_doric_files_sorts_runs_and_ignores_other_files(monkeypatch):
    run2 = Path("260923_SC81_run2_0000.doric")
    run1 = Path("nested/SC81_260923_run1_0000.doric")
    description = Path("SC81_260923_10minBaseline.doric")
    candidates = [run2, Path("notes.doric"), description, run1]
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(Path, "rglob", lambda self, pattern: candidates)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    assert find_doric_files("SC81", "260923", doric_root="root") == [
        run1,
        run2,
        description,
    ]
