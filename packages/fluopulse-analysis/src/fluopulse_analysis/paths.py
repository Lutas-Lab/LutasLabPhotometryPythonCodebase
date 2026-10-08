"""Laboratory file naming conventions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PHOTOMETRY_ROOT = Path("Z:/Photometry")
DEFAULT_DORIC_ROOT = Path("Z:/FLIM FLIP")
_DORIC_NAMES = (
    re.compile(
        r"^(?P<date>\d{6})[_-](?P<mouse>.+?)[_-]run(?P<run>\d+)"
        r"(?:[_-].*)?\.doric$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<mouse>.+?)[_-](?P<date>\d{6})[_-]run(?P<run>\d+)"
        r"(?:[_-].*)?\.doric$",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class SessionIdentity:
    mouse: str
    date: str
    run: int


@dataclass(frozen=True)
class NIDAQPaths:
    nidaq: Path
    running: Path

    def validate(self, *, require_running: bool = True) -> None:
        required = {"NI-DAQ": self.nidaq}
        if require_running:
            required["running"] = self.running
        missing = [
            f"{name}: {path}" for name, path in required.items() if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("Session files not found:\n" + "\n".join(missing))


def infer_session_identity(doric_path: str | Path) -> SessionIdentity:
    """Infer mouse, date, and run from a standard Doric filename."""

    name = Path(doric_path).name
    match = None
    for pattern in _DORIC_NAMES:
        match = pattern.match(name)
        if match is not None:
            break
    if match is None:
        raise ValueError(
            "Doric filename must resemble '260923_SC81_run1_0000.doric' "
            "or 'SC81_260923_run1_0000.doric'"
        )
    return SessionIdentity(
        mouse=match.group("mouse"),
        date=match.group("date"),
        run=int(match.group("run")),
    )


def nidaq_paths(
    mouse: str,
    date: str,
    run: int,
    *,
    photometry_root: str | Path = DEFAULT_PHOTOMETRY_ROOT,
) -> NIDAQPaths:
    """Construct corresponding NI-DAQ and running paths."""

    mouse = str(mouse).strip()
    date = str(date).strip()
    run = int(run)
    folder = Path(photometry_root) / mouse / f"{mouse}_{date}"
    stem = f"{mouse}-{date}-{run:03d}"
    return NIDAQPaths(
        nidaq=folder / f"{stem}-nidaq.mat",
        running=folder / f"{stem}-running.mat",
    )


def nidaq_paths_for_doric(
    doric_path: str | Path,
    *,
    photometry_root: str | Path = DEFAULT_PHOTOMETRY_ROOT,
) -> NIDAQPaths:
    identity = infer_session_identity(doric_path)
    return nidaq_paths(
        identity.mouse,
        identity.date,
        identity.run,
        photometry_root=photometry_root,
    )


def find_doric_files(
    mouse: str,
    date: str,
    *,
    doric_root: str | Path = DEFAULT_DORIC_ROOT,
) -> list[Path]:
    """Find standard Doric recordings for one mouse and date, sorted by run."""

    mouse = str(mouse).strip()
    date = str(date).strip().zfill(6)
    root = Path(doric_root)
    folders = (
        root / mouse / f"{mouse}_{date}",
        root / f"{mouse}_{date}",
        root / mouse / date,
        root / mouse,
    )
    matched: list[tuple[tuple[int, int | str], Path]] = []
    candidates: dict[str, Path] = {}
    for folder in folders:
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file() and path.suffix.casefold() == ".doric":
                    candidates[str(path.resolve()).casefold()] = path
    for path in candidates.values():
        try:
            identity = infer_session_identity(path)
        except ValueError:
            name = path.stem.casefold()
            valid_prefixes = (
                f"{date}_{mouse}".casefold(),
                f"{mouse}_{date}".casefold(),
            )
            if name.startswith(valid_prefixes):
                matched.append(((1, name), path))
            continue
        if identity.mouse.casefold() == mouse.casefold() and identity.date == date:
            matched.append(((0, identity.run), path))
    return [path for _, path in sorted(matched)]
