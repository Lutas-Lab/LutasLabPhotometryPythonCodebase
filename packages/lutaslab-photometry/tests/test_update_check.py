import io
import json
from urllib.error import URLError

import pytest

from lutaslab_photometry import update_check

COMMIT = "a" * 40


def _make_checkout(tmp_path, *, packed=False):
    git_directory = tmp_path / ".git"
    git_directory.mkdir()
    (git_directory / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    if packed:
        (git_directory / "packed-refs").write_text(
            f"# pack-refs with: peeled fully-peeled\n{COMMIT} refs/heads/main\n",
            encoding="utf-8",
        )
    else:
        reference = git_directory / "refs" / "heads" / "main"
        reference.parent.mkdir(parents=True)
        reference.write_text(f"{COMMIT}\n", encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("packed", [False, True])
def test_local_git_commit_reads_checkout_without_git_executable(tmp_path, packed):
    checkout = _make_checkout(tmp_path, packed=packed)

    assert update_check.local_git_commit(checkout) == COMMIT


def test_local_git_commit_returns_none_for_zip_copy(tmp_path):
    assert update_check.local_git_commit(tmp_path) is None


@pytest.mark.parametrize(
    ("github_status", "ahead_by", "expected_state"),
    [
        ("identical", 0, "up_to_date"),
        ("ahead", 3, "update_available"),
        ("behind", 0, "local_ahead"),
        ("diverged", 2, "diverged"),
    ],
)
def test_check_for_update_interprets_github_comparison(
    monkeypatch, tmp_path, github_status, ahead_by, expected_state
):
    _make_checkout(tmp_path)

    def fake_urlopen(request, timeout):
        assert COMMIT in request.full_url
        assert timeout == 1.0
        return io.BytesIO(
            json.dumps({"status": github_status, "ahead_by": ahead_by}).encode()
        )

    monkeypatch.setattr(update_check, "urlopen", fake_urlopen)

    result = update_check.check_for_update(tmp_path, timeout_seconds=1.0)

    assert result.state == expected_state
    assert result.local_commit == COMMIT
    if expected_state == "update_available":
        assert result.commits_behind == ahead_by


def test_check_for_update_handles_network_failure(monkeypatch, tmp_path):
    _make_checkout(tmp_path)

    def fail_urlopen(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr(update_check, "urlopen", fail_urlopen)

    result = update_check.check_for_update(tmp_path)

    assert result.state == "check_failed"
    assert result.local_commit == COMMIT
