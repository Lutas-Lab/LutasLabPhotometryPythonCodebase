import sys
import types

import numpy as np

from lutaslab_core.session import AlignedSession, ContinuousSignal, EventSeries, IntervalSeries


def test_session_converts_common_types_to_pynapple(monkeypatch):
    fake = types.SimpleNamespace(
        Tsd=lambda **kwargs: ("Tsd", kwargs),
        Ts=lambda **kwargs: ("Ts", kwargs),
        IntervalSet=lambda **kwargs: ("IntervalSet", kwargs),
    )
    monkeypatch.setitem(sys.modules, "pynapple", fake)
    session = AlignedSession(
        "mouse-date-run",
        continuous={"signal": ContinuousSignal([0, 1], [2, 3], "ns")},
        events={"licks": EventSeries([0.25])},
        intervals={"cue": IntervalSeries([0.5], [0.75])},
    )
    converted = session.to_pynapple()
    assert converted["signal"][0] == "Tsd"
    assert converted["licks"][0] == "Ts"
    assert converted["cue"][0] == "IntervalSet"


def test_continuous_signals_allow_missing_samples_but_not_infinity():
    signal = ContinuousSignal([0, 1], [2, np.nan], "ns")
    assert np.isnan(signal.values[1])
    with np.testing.assert_raises_regex(ValueError, "non-infinite"):
        ContinuousSignal([0, 1], [2, np.inf], "ns")
