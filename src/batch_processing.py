from pathlib import Path

from .load_data import get_session_paths, load_session_data
from .preprocess import preprocess_session
from .save_sessiondata import save_session
from .session_manifest import processed_session_path


def preprocess_manifest_sessions(
    sessions,
    data_root,
    *,
    overwrite=False,
    continue_on_error=False,
    channel_map=None,
    preprocessing_options=None,
):
    """Preprocess manifest sessions and save each result beside its raw data."""
    data_root = Path(data_root)
    preprocessing_options = dict(preprocessing_options or {})
    results = []

    for info in sessions:
        identifier = (
            f"{info['mouse']} {info['date']} run {int(info['run'])}"
        )
        destination = processed_session_path(data_root, info)

        if destination.exists() and not overwrite:
            results.append(
                {
                    **info,
                    "status": "skipped",
                    "path": destination,
                    "error": None,
                }
            )
            continue

        try:
            paths = get_session_paths(
                info["mouse"], info["date"], info["run"], data_root
            )
            raw_session = load_session_data(paths, channel_map=channel_map)
            processed = preprocess_session(raw_session, **preprocessing_options)
            save_path = save_session(processed)
        except Exception as error:
            if not continue_on_error:
                raise RuntimeError(f"Failed to preprocess {identifier}.") from error
            results.append(
                {
                    **info,
                    "status": "failed",
                    "path": destination,
                    "error": str(error),
                }
            )
        else:
            results.append(
                {
                    **info,
                    "status": "processed",
                    "path": save_path,
                    "error": None,
                }
            )

    return results
