from pathlib import Path

import numpy as np


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
            "/data/lutasa2/photometry"

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