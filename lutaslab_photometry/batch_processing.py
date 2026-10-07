from pathlib import Path

from lutaslab_core.batch import run_batch

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

    def process(info):
        paths = get_session_paths(
            info["mouse"], info["date"], info["run"], data_root
        )
        raw_session = load_session_data(paths, channel_map=channel_map)
        processed = preprocess_session(raw_session, **preprocessing_options)
        return save_session(processed)

    return run_batch(
        sessions,
        process,
        destination_for=lambda info: processed_session_path(data_root, info),
        identifier_for=lambda info: (
            f"{info['mouse']} {info['date']} run {int(info['run'])}"
        ),
        overwrite=overwrite,
        continue_on_error=continue_on_error,
    )
