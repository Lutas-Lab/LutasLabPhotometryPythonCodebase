from pathlib import Path

from lutaslab_core.manifest import load_session_manifest

REQUIRED_COLUMNS = ("mouse", "date", "run")


def resolve_session_channel(session, channel="manifest"):
    """Resolve a manifest channel or an explicit all-session override."""
    if channel in (None, "manifest"):
        raw_channel = session.get("channel", 1)
        if raw_channel in (None, ""):
            raw_channel = 1
    else:
        raw_channel = channel
    if isinstance(raw_channel, bool):
        raise ValueError("Photoreceiver channel must be 1 or 2, not a boolean.")
    try:
        selected = int(raw_channel)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid photoreceiver channel {raw_channel!r} for "
            f"{session.get('mouse', 'session')}."
        ) from error
    if selected not in (1, 2):
        raise ValueError(
            f"Photoreceiver channel must be 1 or 2; got {selected} for "
            f"{session.get('mouse', 'session')}."
        )
    return selected


def processed_session_path(data_root, session):
    """Return the conventional processed-session path for one manifest row."""
    mouse = str(session["mouse"])
    date = str(session["date"])
    run = int(session["run"])
    return (
        Path(data_root)
        / mouse
        / f"{mouse}_{date}"
        / f"{mouse}-{date}-{run:03d}-processed.npz"
    )


__all__ = [
    "REQUIRED_COLUMNS",
    "load_session_manifest",
    "processed_session_path",
    "resolve_session_channel",
]
