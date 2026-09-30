import numpy as np
from lutaslab_core.session import (
    AlignedSession,
    ContinuousSignal,
    EventSeries,
    IntervalSeries,
)


def session_to_core(session, session_id=None):
    """Convert a processed photometry dictionary to the common session model."""

    metadata = dict(session.get("metadata", {}))
    metadata.setdefault("sensor", "fiber_photometry")
    return AlignedSession(
        session_id=session_id or str(metadata.get("session_id", "photometry-session")),
        continuous={
            "dff_ch1": ContinuousSignal(
                session["photo_time_465_ch1"],
                session["dff_ch1"],
                "dF/F",
            ),
            "dff_ch2": ContinuousSignal(
                session["photo_time_465_ch2"],
                session["dff_ch2"],
                "dF/F",
            ),
            "locomotion": ContinuousSignal(
                session["locomotion_time"],
                session["processed_locomotion"],
                "a.u.",
            ),
        },
        events={"licks": EventSeries(session["lick_times"], "licks")},
        intervals={
            "visual_cues": IntervalSeries(
                session["cue_onset"],
                session["cue_offset"],
                "visual_cues",
            ),
            "solenoid": IntervalSeries(
                session["solenoid_onset"],
                session["solenoid_offset"],
                "solenoid",
            ),
            "lick_bouts": IntervalSeries(
                session["lick_bout_onset"],
                session["lick_bout_offset"],
                "lick_bouts",
            ),
        },
        metadata=metadata,
    )


def session_to_pynapple(session):
    """
    Convert processed session data into Pynapple objects.

    Returns
    -------
    data : dict
        Dictionary containing Pynapple objects.
    """

    try:
        import pynapple as nap
    except ImportError as error:
        raise ImportError("Pynapple conversion requires pynapple") from error

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
