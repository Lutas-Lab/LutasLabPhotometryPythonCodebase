from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import subprocess
import warnings

import numpy as np


SCHEMA_VERSION = "1.0"
REQUIRED_SESSION_KEYS = {
    "mouse",
    "date",
    "run",
    "photo_time_465_ch1",
    "photometry_465_ch1",
    "dff_ch1",
    "locomotion_time",
    "processed_locomotion",
}


def validate_session(session, *, require_current_schema=False):
    """Validate keys needed by downstream session consumers."""
    missing = REQUIRED_SESSION_KEYS.difference(session)
    if missing:
        raise ValueError(f"Processed session is missing keys: {sorted(missing)}")

    schema = str(session.get("processed_schema_version", "legacy"))
    if require_current_schema and schema != SCHEMA_VERSION:
        raise ValueError(
            f"Processed session schema {schema!r} is not supported; expected {SCHEMA_VERSION!r}."
        )
    return schema


def _package_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def _git_commit():
    project_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


# ============================================================
# Save processed session
# ============================================================

def save_session(
    session,
    output_dir=None
):
    """
    Save a processed session as a compressed .npz file.

    By default, the file is saved in the same directory as
    the original photometry file.

    Parameters
    ----------
    session : dict
        Processed session dictionary.

    output_dir : str or Path, optional
        Directory in which to save the processed session.

        If None, the directory containing
        session["photometry_path"] is used.

    Returns
    -------
    save_path : Path
        Path to the saved processed session.
    """

    schema = validate_session(session)

    # --------------------------------------------------------
    # Session identifiers
    # --------------------------------------------------------

    mouse = session["mouse"]
    date = session["date"]
    run = int(session["run"])

    # --------------------------------------------------------
    # Determine output directory
    # --------------------------------------------------------

    if output_dir is None:

        photometry_path = Path(
            session["photometry_path"]
        )

        output_dir = photometry_path.parent

    else:

        output_dir = Path(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    filename = (
        f"{mouse}-"
        f"{date}-"
        f"{run:03d}-"
        f"processed.npz"
    )

    save_path = (
        output_dir
        / filename
    )

    # --------------------------------------------------------
    # Prepare dictionary for NumPy
    # --------------------------------------------------------

    save_dict = {}

    for key, value in session.items():

        # Convert Path objects to strings
        if isinstance(value, Path):

            save_dict[key] = str(value)

        else:

            save_dict[key] = value

    save_dict["processed_schema_version"] = schema
    save_dict["processing_utc"] = datetime.now(timezone.utc).isoformat()
    save_dict["code_commit"] = _git_commit()
    save_dict["numpy_version"] = _package_version("numpy")
    save_dict["scipy_version"] = _package_version("scipy")
    save_dict["pynapple_version"] = _package_version("pynapple")
    save_dict["nemos_version"] = _package_version("nemos")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez_compressed(
        save_path,
        **save_dict
    )

    print(
        f"Saved processed session:\n"
        f"{save_path}"
    )

    return save_path


# ============================================================
# Load processed session directly from a file
# ============================================================

def load_session(
    path
):
    """
    Load a processed session directly from an .npz file.

    This is the most portable loading method and works on
    Windows, Linux, Biowulf, etc.

    Parameters
    ----------
    path : str or Path
        Full path to the processed .npz file.

    Returns
    -------
    session : dict
        Loaded processed session.
    """

    load_path = Path(
        path
    )

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not load_path.exists():

        raise FileNotFoundError(
            f"Processed session not found:\n"
            f"{load_path}"
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    loaded = np.load(
        load_path,
        allow_pickle=False
    )

    session = {}

    for key in loaded.files:

        value = loaded[key]

        # Convert zero-dimensional arrays back
        # into regular Python scalars
        if value.ndim == 0:

            value = value.item()

        session[key] = value

    loaded.close()

    # --------------------------------------------------------
    # Store where THIS processed file was loaded from
    #
    # This is separate from session["photometry_path"],
    # which may still contain the original Windows path.
    # --------------------------------------------------------

    schema = validate_session(session)
    if schema == "legacy":
        warnings.warn(
            "Loaded a legacy processed session without a schema version.",
            UserWarning,
            stacklevel=2,
        )

    session["processed_session_path"] = str(
        load_path
    )

    print(
        f"Loaded processed session:\n"
        f"{load_path}"
    )

    return session


# ============================================================
# Load processed session by mouse/date/run
# ============================================================

def load_session_by_id(
    mouse,
    date,
    run,
    data_root
):
    """
    Load a processed session using mouse, date, and run.

    Unlike the old version, this function does NOT assume
    Z:\\Photometry. The data root must be supplied explicitly.

    Parameters
    ----------
    mouse : str
        Mouse identifier.

    date : str
        Session date.

    run : int
        Run number.

    data_root : str or Path
        Root directory containing the mouse/session folders.

        Example on Windows:
            r"Z:\\Photometry"

        Example on Linux/Biowulf:
            "/data/USERNAME/photometry"

    Returns
    -------
    session : dict
        Loaded processed session.
    """

    data_root = Path(
        data_root
    )

    run = int(run)

    # --------------------------------------------------------
    # Reconstruct session directory
    # --------------------------------------------------------

    session_dir = (
        data_root
        / mouse
        / f"{mouse}_{date}"
    )

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    filename = (
        f"{mouse}-"
        f"{date}-"
        f"{run:03d}-"
        f"processed.npz"
    )

    load_path = (
        session_dir
        / filename
    )

    # --------------------------------------------------------
    # Use portable loader
    # --------------------------------------------------------

    return load_session(
        load_path
    )
