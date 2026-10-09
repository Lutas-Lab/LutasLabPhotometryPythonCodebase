import json
import warnings
from pathlib import Path

import numpy as np

from .load_data import DEFAULT_CHANNEL_MAP
from .processed_provenance import build_processed_provenance, provenance_json

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
PREPROCESSING_PARAMETER_KEYS = (
    "photometry_edge",
    "irls_constant",
    "ttl_threshold",
    "cue_max_pulse_gap",
    "locomotion_min_width",
    "locomotion_max_width",
    "locomotion_invert",
    "lick_bout_interval",
    "minimum_bout_licks",
    "post_cue_window",
)
CONVENTIONAL_PREPROCESSING_DEFAULTS = {
    "channel_map": DEFAULT_CHANNEL_MAP,
    "photometry_edge": 3,
    "irls_constant": 1.4,
    "ttl_threshold": 1.5,
    "cue_max_pulse_gap": 0.5,
    "locomotion_min_width": 4,
    "locomotion_max_width": 6,
    "locomotion_invert": True,
    "lick_bout_interval": 1.0,
    "minimum_bout_licks": 3,
    "post_cue_window": 2.0,
}


def conventional_preprocessing_parameters(overrides=None):
    """Return the maintained conventional preprocessing defaults plus overrides."""

    parameters = dict(CONVENTIONAL_PREPROCESSING_DEFAULTS)
    parameters.update(overrides or {})
    return parameters


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

    parameters = conventional_preprocessing_parameters(
        {
            key: session[key]
            for key in PREPROCESSING_PARAMETER_KEYS
            if key in session
        }
    )
    if "channel_map_keys" in session and "channel_map_rows" in session:
        parameters["channel_map"] = {
            str(key): int(row)
            for key, row in zip(
                session["channel_map_keys"],
                session["channel_map_rows"],
                strict=True,
            )
        }
    provenance = build_processed_provenance(
        "conventional",
        parameters,
        processed_schema_version=schema,
    )
    packages = provenance["software"]["packages"]
    save_dict["processed_schema_version"] = schema
    save_dict["provenance_json"] = provenance_json(provenance)
    # Retain the original top-level provenance fields for older consumers.
    save_dict["processing_utc"] = provenance["processing_utc"]
    save_dict["code_commit"] = provenance["code"]["commit"] or "unknown"
    save_dict["numpy_version"] = packages["numpy"] or "unknown"
    save_dict["scipy_version"] = packages["scipy"] or "unknown"
    save_dict["pynapple_version"] = packages["pynapple"] or "unknown"
    save_dict["nemos_version"] = packages["nemos"] or "unknown"

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

    if "provenance_json" in session:
        session["provenance"] = json.loads(str(session["provenance_json"]))

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
