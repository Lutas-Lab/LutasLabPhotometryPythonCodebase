from pathlib import Path
from scipy.io import loadmat

def get_session_paths(mouse_name, date, run):
    base=Path(r"Z:\Photometry")
    folder = base/mouse_name/f"{mouse_name}_{date}"
    photometry_path=folder/f"{mouse_name}-{date}-{run:03d}-nidaq.mat"
    locomotion_path=folder/f"{mouse_name}-{date}-{run:03d}-running.mat"

    session = {
        "mouse": mouse_name,
        "date":date,
        "run": run,
        "photometry_path":
photometry_path,
        "locomotion_path":
locomotion_path
    }
    
    return session

def load_session_data(session):
    mat = loadmat(session["photometry_path"])
    data = mat["data"]

    # Photoreceiver signals
    session["raw_photometry_ch1"] = data[0, :]
    session["raw_photometry_ch2"] = data[2, :]

    # Behavioral signals
    session["visual_cue"] = data[4, :]
    session["licking"] = data[3, :]
    session["solenoid_opening"] = data[7, :]

    # LED timing signals
    session["ttl_465"] = data[5, :]
    session["ttl_405"] = data[6, :]

    # Acquisition information
    session["timestamps"] = mat["timestamps"].squeeze()
    session["fs"] = mat["Fs"].squeeze()

    # Locomotion synchronization
    session["locomotion_ttlpulses"] = data[1, :]

    loco_mat = loadmat(session["locomotion_path"])
    session["locomotion"] = loco_mat["speed"].squeeze()

    return session