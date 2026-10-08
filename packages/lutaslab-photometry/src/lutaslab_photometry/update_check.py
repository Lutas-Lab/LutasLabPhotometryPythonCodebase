"""Read-only update checks for repository-based installations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GITHUB_COMPARE_URL = (
    "https://api.github.com/repos/Lutas-Lab/"
    "LutasLabPhotometryPythonCodebase/compare/{commit}...main?per_page=1"
)
GITHUB_REPOSITORY_URL = (
    "https://github.com/Lutas-Lab/LutasLabPhotometryPythonCodebase"
)
_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40,64}")

UpdateState = Literal[
    "up_to_date",
    "update_available",
    "local_ahead",
    "diverged",
    "not_a_clone",
    "check_failed",
]


@dataclass(frozen=True)
class UpdateStatus:
    """Result of comparing the local checkout with GitHub's ``main`` branch."""

    state: UpdateState
    local_commit: str | None = None
    commits_behind: int | None = None


def _git_directory(project_root: Path) -> Path | None:
    marker = project_root / ".git"
    if marker.is_dir():
        return marker
    if not marker.is_file():
        return None
    try:
        marker_text = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    prefix = "gitdir:"
    if not marker_text.lower().startswith(prefix):
        return None
    git_directory = Path(marker_text[len(prefix) :].strip())
    if not git_directory.is_absolute():
        git_directory = marker.parent / git_directory
    return git_directory.resolve()


def local_git_commit(project_root: str | Path) -> str | None:
    """Return the checkout's full commit hash without requiring Git on PATH."""

    git_directory = _git_directory(Path(project_root))
    if git_directory is None:
        return None
    try:
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if _COMMIT_PATTERN.fullmatch(head):
        return head.lower()
    prefix = "ref:"
    if not head.lower().startswith(prefix):
        return None
    reference = head[len(prefix) :].strip()
    try:
        commit = (git_directory / reference).read_text(encoding="utf-8").strip()
    except OSError:
        commit = ""
    if _COMMIT_PATTERN.fullmatch(commit):
        return commit.lower()
    try:
        packed_references = (git_directory / "packed-refs").read_text(
            encoding="utf-8"
        )
    except OSError:
        return None
    for line in packed_references.splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        fields = line.split(" ", maxsplit=1)
        if len(fields) == 2 and fields[1] == reference:
            return fields[0].lower() if _COMMIT_PATTERN.fullmatch(fields[0]) else None
    return None


def check_for_update(
    project_root: str | Path,
    *,
    timeout_seconds: float = 3.0,
) -> UpdateStatus:
    """Compare this checkout with public GitHub ``main`` without credentials."""

    local_commit = local_git_commit(project_root)
    if local_commit is None:
        return UpdateStatus("not_a_clone")

    request = Request(
        GITHUB_COMPARE_URL.format(commit=local_commit),
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "LutasLabPhotometry-update-check",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = json.load(response)
    except (HTTPError, URLError, OSError, TimeoutError, ValueError):
        return UpdateStatus("check_failed", local_commit)

    comparison = payload.get("status")
    if comparison == "identical":
        return UpdateStatus("up_to_date", local_commit, 0)
    if comparison == "ahead":
        commits_behind = payload.get("ahead_by")
        if not isinstance(commits_behind, int):
            commits_behind = None
        return UpdateStatus("update_available", local_commit, commits_behind)
    if comparison == "behind":
        return UpdateStatus("local_ahead", local_commit)
    if comparison == "diverged":
        return UpdateStatus("diverged", local_commit)
    return UpdateStatus("check_failed", local_commit)
