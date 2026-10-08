"""Laboratory path conventions for paired iFLiP and NI-DAQ sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path("Z:/")
PHOTOMETRY_FOLDER = "Photometry"
IFLIP_FOLDER = "FLIM FLIP"


def _is_file(path: Path) -> bool:
    """Return false for absent or currently inaccessible mapped-drive paths."""

    try:
        return path.is_file()
    except OSError:
        return False


@dataclass(frozen=True)
class SessionPaths:
    """Conventional input paths for one mouse/date/run session."""

    mouse: str
    date: str
    run: int
    iflip_preferred: Path
    iflip_legacy: Path
    nidaq: Path
    running: Path

    @property
    def iflip_candidates(self) -> tuple[Path, Path]:
        """Return preferred and legacy iFLiP filenames, in that order."""

        return self.iflip_preferred, self.iflip_legacy

    @property
    def iflip(self) -> Path:
        """Choose the preferred iFLiP file, with automatic legacy fallback."""

        if _is_file(self.iflip_preferred) or not _is_file(self.iflip_legacy):
            return self.iflip_preferred
        return self.iflip_legacy

    def validate(self, *, require_running: bool = True) -> None:
        """Raise a single informative error listing missing session inputs."""

        missing: list[str] = []
        if not any(_is_file(path) for path in self.iflip_candidates):
            missing.append(
                f"iFLiP: {self.iflip_preferred} (preferred) or {self.iflip_legacy} (legacy)"
            )
        required = {"NI-DAQ": self.nidaq}
        if require_running:
            required["running"] = self.running
        missing.extend(f"{label}: {path}" for label, path in required.items() if not _is_file(path))
        if missing:
            raise FileNotFoundError("Session input files not found:\n" + "\n".join(missing))


def session_paths(
    mouse: str,
    date: str,
    run: int,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
) -> SessionPaths:
    """Construct paired iFLiP, NI-DAQ, and running paths by session identity.

    ``date`` is treated as text so leading zeroes are preserved.
    """

    mouse = str(mouse).strip()
    date = str(date).strip()
    if not mouse or not date:
        raise ValueError("mouse and date must not be empty")
    if any(character in mouse or character in date for character in ("/", "\\")):
        raise ValueError("mouse and date must not contain path separators")
    if isinstance(run, bool):
        raise ValueError("run must be a nonnegative integer")
    try:
        numeric_run = float(run)
    except (TypeError, ValueError) as exc:
        raise ValueError("run must be a nonnegative integer") from exc
    if not numeric_run.is_integer() or numeric_run < 0:
        raise ValueError("run must be a nonnegative integer")
    run = int(numeric_run)

    root = Path(data_root)
    dated_folder = f"{mouse}_{date}"
    photometry_folder = root / PHOTOMETRY_FOLDER / mouse / dated_folder
    iflip_folder = root / IFLIP_FOLDER / mouse / dated_folder
    photometry_stem = f"{mouse}-{date}-{run:03d}"
    preferred_iflip_stem = f"{mouse}_{date}_{run:03d}"
    legacy_iflip_stem = f"{mouse}_{date}{run:03d}"

    return SessionPaths(
        mouse=mouse,
        date=date,
        run=run,
        iflip_preferred=iflip_folder / f"{preferred_iflip_stem}.iFLiP3",
        iflip_legacy=iflip_folder / f"{legacy_iflip_stem}.iFLiP3",
        nidaq=photometry_folder / f"{photometry_stem}-nidaq.mat",
        running=photometry_folder / f"{photometry_stem}-running.mat",
    )
