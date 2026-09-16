import csv
from pathlib import Path


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


def load_session_manifest(path):
    """Load and validate a CSV containing mouse, date, and run columns."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Session manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("Session manifest is missing a header row.")

        fieldnames = tuple(name.strip() for name in reader.fieldnames)
        missing = set(REQUIRED_COLUMNS).difference(fieldnames)
        if missing:
            raise ValueError(
                f"Session manifest is missing columns: {sorted(missing)}"
            )

        sessions = []
        seen = set()
        for line_number, row in enumerate(reader, start=2):
            row = {(key or "").strip(): value for key, value in row.items()}
            if not any((value or "").strip() for value in row.values()):
                continue

            mouse = (row.get("mouse") or "").strip()
            date = (row.get("date") or "").strip()
            run_text = (row.get("run") or "").strip()

            if not mouse or not date or not run_text:
                raise ValueError(
                    f"Session manifest line {line_number} has an empty required value."
                )

            try:
                run = int(run_text)
            except ValueError as error:
                raise ValueError(
                    f"Session manifest line {line_number} has invalid run {run_text!r}."
                ) from error
            if run < 0:
                raise ValueError(
                    f"Session manifest line {line_number} has a negative run number."
                )

            identifier = (mouse, date, run)
            if identifier in seen:
                raise ValueError(
                    "Session manifest contains a duplicate session: "
                    f"{mouse} {date} run {run}."
                )
            seen.add(identifier)
            session = {
                key: (value or "").strip()
                for key, value in row.items()
                if key not in REQUIRED_COLUMNS and (value or "").strip()
            }
            session.update({"mouse": mouse, "date": date, "run": run})
            sessions.append(session)

    if not sessions:
        raise ValueError("Session manifest contains no sessions.")

    return sessions


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
