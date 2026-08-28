import numpy as np
import pynapple as nap


def session_to_pynapple(session):
    """
    Convert processed session data into Pynapple objects.

    Returns
    -------
    data : dict
        Dictionary containing Pynapple objects.
    """

    data = {}

    # -------------------------
    # Photometry
    # -------------------------

    data["dff_ch1"] = nap.Tsd(
        t=session["photo_time_465_ch1"],
        d=session["dff_ch1"],
        time_units="s"
    )

    data["dff_ch2"] = nap.Tsd(
        t=session["photo_time_465_ch2"],
        d=session["dff_ch2"],
        time_units="s"
    )


    # -------------------------
    # Locomotion
    # -------------------------

    data["locomotion"] = nap.Tsd(
        t=session["locomotion_time"],
        d=session["processed_locomotion"],
        time_units="s"
    )


    # -------------------------
    # Individual licks
    # -------------------------

    data["licks"] = nap.Ts(
        t=session["lick_times"],
        time_units="s"
    )


    # -------------------------
    # Visual cues
    # -------------------------

    data["visual_cues"] = nap.IntervalSet(
        start=session["cue_onset"],
        end=session["cue_offset"],
        time_units="s"
    )


    # -------------------------
    # Solenoid openings
    # -------------------------

    data["solenoid"] = nap.IntervalSet(
        start=session["solenoid_onset"],
        end=session["solenoid_offset"],
        time_units="s"
    )


    # -------------------------
    # Lick bouts
    # -------------------------

    data["lick_bouts"] = nap.IntervalSet(
        start=session["lick_bout_onset"],
        end=session["lick_bout_offset"],
        time_units="s"
    )


    return data