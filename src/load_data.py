from pathlib import Path

import numpy as np
from scipy.io import loadmat


DEFAULT_CHANNEL_MAP = {
    "raw_photometry_ch1": 0,
    "locomotion_ttlpulses": 1,
    "raw_photometry_ch2": 2,
    "licking": 3,
    "visual_cue": 4,
    "ttl_465": 5,
    "ttl_405": 6,
    "solenoid_opening": 7,
}


def get_session_paths(mouse_name, date, run, data_root):
    """Construct raw-data paths for one session."""
    base = Path(data_root)
    run = int(run)
    folder = base / mouse_name / f"{mouse_name}_{date}"
    photometry_path = folder / f"{mouse_name}-{date}-{run:03d}-nidaq.mat"
    locomotion_path = folder / f"{mouse_name}-{date}-{run:03d}-running.mat"

    session = {
        "mouse": mouse_name,
        "date": date,
        "run": run,
        "photometry_path": photometry_path,
        "locomotion_path": locomotion_path,
    }

    return session

def load_session_data(session, channel_map=None):
    """Load and validate the raw MATLAB files for one session."""
    for key in ("photometry_path", "locomotion_path"):
        path = Path(session[key])
        if not path.is_file():
            raise FileNotFoundError(f"Session input not found: {path}")

    mat = loadmat(session["photometry_path"])
    required = {"data", "timestamps", "Fs"}
    missing = required.difference(mat)
    if missing:
        raise KeyError(f"Photometry file is missing variables: {sorted(missing)}")

    data = mat["data"]
    channel_map = DEFAULT_CHANNEL_MAP if channel_map is None else dict(channel_map)
    missing_channels = set(DEFAULT_CHANNEL_MAP).difference(channel_map)
    if missing_channels:
        raise ValueError(f"channel_map is missing entries: {sorted(missing_channels)}")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in channel_map.values()
    ):
        raise ValueError("channel_map values must be nonnegative integer row indices.")

    if data.ndim != 2 or data.shape[0] <= max(channel_map.values()):
        raise ValueError(
            "Photometry 'data' does not contain every configured channel row."
        )

    # Photoreceiver signals
    session["raw_photometry_ch1"] = data[channel_map["raw_photometry_ch1"], :]
    session["raw_photometry_ch2"] = data[channel_map["raw_photometry_ch2"], :]

    # Behavioral signals
    session["visual_cue"] = data[channel_map["visual_cue"], :]
    session["licking"] = data[channel_map["licking"], :]
    session["solenoid_opening"] = data[channel_map["solenoid_opening"], :]

    # LED timing signals
    session["ttl_465"] = data[channel_map["ttl_465"], :]
    session["ttl_405"] = data[channel_map["ttl_405"], :]

    # Acquisition information
    session["timestamps"] = mat["timestamps"].squeeze()
    fs_value = np.asarray(mat["Fs"]).squeeze()
    if fs_value.ndim != 0 or not np.isfinite(fs_value) or fs_value <= 0:
        raise ValueError("Fs must contain one finite positive sampling rate.")
    session["fs"] = float(fs_value)
    if session["timestamps"].ndim != 1:
        raise ValueError("timestamps must be one-dimensional.")
    if data.shape[1] != len(session["timestamps"]):
        raise ValueError("Photometry data columns do not match timestamps.")

    # Locomotion synchronization
    session["locomotion_ttlpulses"] = data[channel_map["locomotion_ttlpulses"], :]
    session["channel_map_keys"] = np.array(list(channel_map), dtype=str)
    session["channel_map_rows"] = np.array(list(channel_map.values()), dtype=int)

    loco_mat = loadmat(session["locomotion_path"])
    if "speed" not in loco_mat:
        raise KeyError("Locomotion file is missing the 'speed' variable.")
    session["locomotion"] = loco_mat["speed"].squeeze()

    return session
