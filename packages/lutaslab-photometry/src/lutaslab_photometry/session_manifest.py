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


def filter_manifest_sessions(sessions, *, group=None, condition=None):
    """Select manifest rows by exact, case-insensitive group and condition labels."""

    selected_group = None if group is None else str(group).strip().casefold()
    selected_condition = (
        None if condition is None else str(condition).strip().casefold()
    )
    filtered = [
        session
        for session in sessions
        if (
            selected_group is None
            or str(session.get("group", "") or "").strip().casefold()
            == selected_group
        )
        and (
            selected_condition is None
            or str(session.get("condition", "") or "").strip().casefold()
            == selected_condition
        )
    ]
    if not filtered:
        criteria = []
        if group is not None:
            criteria.append(f"group={group!r}")
        if condition is not None:
            criteria.append(f"condition={condition!r}")
        raise ValueError(
            "No manifest sessions match " + " and ".join(criteria) + "."
        )
    return filtered


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
    "filter_manifest_sessions",
    "load_session_manifest",
    "processed_session_path",
    "resolve_session_channel",
]
