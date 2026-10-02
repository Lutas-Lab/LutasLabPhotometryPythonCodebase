# Lutas Lab Core

Shared, sensor-independent building blocks for Lutas Lab time-series analyses.

The package also provides `PublicationBundle` for exporting small, versioned,
compressed source-data deposits with figure mappings and integrity checks. See
[`docs/publication-bundles.md`](../../docs/publication-bundles.md) for the data
retention policy and an example.

The package owns laboratory NI-DAQ loading, TTL and behavioral event
representations, affine clock alignment, common time-series containers, and
conversion to Pynapple. Sensor-specific preprocessing remains in the
photometry, FluoPulse, and iFLiP3 packages.

Reusable modeling utilities include raised-cosine temporal bases, trial-reset
design matrices, grouped ridge cross-validation, sampled lick-bout features,
and causal gamma-cascade transfer functions. A data-independent example is in
[`examples/event_glm_and_transfer.py`](../../examples/event_glm_and_transfer.py).

The public API uses seconds for all timestamps and durations. Continuous data
are represented by `ContinuousSignal`, timestamp events by `EventSeries`, and
bounded events by `IntervalSeries`.
