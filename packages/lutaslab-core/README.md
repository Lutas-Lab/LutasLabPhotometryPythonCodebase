# Lutas Lab Core

Shared, sensor-independent building blocks for Lutas Lab time-series analyses.

The package owns laboratory NI-DAQ loading, TTL and behavioral event
representations, affine clock alignment, common time-series containers, and
conversion to Pynapple. Sensor-specific preprocessing remains in the
photometry, FluoPulse, and iFLiP3 packages.

The public API uses seconds for all timestamps and durations. Continuous data
are represented by `ContinuousSignal`, timestamp events by `EventSeries`, and
bounded events by `IntervalSeries`.
