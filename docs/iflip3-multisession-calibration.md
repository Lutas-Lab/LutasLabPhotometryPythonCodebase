# iFLIP3 multi-session calibration

## Purpose

Multi-session calibration estimates one decay-shape basis for a compatible set
of iFLIP3 recordings. The calibration shares short and long lifetimes, decay
IRF width (`sigma`) across the selected sessions and estimates one decay start
(`t0`) for each recording. During subsequent preprocessing those shape values
and the recording's `t0` remain fixed. Only the nonnegative short- and
long-component photon counts and a nonnegative flat-background coefficient are
solved at each photometry time point.

The parameter-sharing strategy follows the global fluorescence-decay analysis
concept introduced by Jay R. Knutson, Joseph M. Beechem, and Ludwig Brand:
Knutson JR, Beechem JM, Brand L. “Simultaneous analysis of multiple fluorescence
decay curves: a global approach.” *Chemical Physics Letters* 102:501-507 (1983).
[doi:10.1016/0009-2614(83)87454-5](https://doi.org/10.1016/0009-2614(83)87454-5).
The repository uses its own variable-projection implementation rather than code
from the original work.

The optional measured-background recording and configured afterpulse correction
are applied first. The fit then includes a nonnegative flat nuisance basis. In
calibration, each session aggregate has its own flat coefficient; in
preprocessing, every temporal bin has its own coefficient. This protects the
long component from a simple DC mismatch while retaining the shared lifetime
shapes. It does not correct shaped tissue autofluorescence, which would require
an experimentally measured autofluorescence basis such as no-sensor controls.

## Recommended workflow

1. Complete and validate the iFLIP3 session table in the GUI.
2. Open **Preprocess and align**, then find **Multi-session calibration**.
3. Start with all sessions that used the same acquisition configuration.
4. Click **Run multi-session calibration**.
5. Review the shared parameters, decay overlays, residual heatmap, and
   per-session QC table.
6. If one or more sessions have structured residuals or markedly poorer QC,
   investigate them. Exclude a session only for a documented technical reason,
   then rerun the calibration.
7. Review the proposed descriptive, timestamped JSON path and click
   **Save calibration**. You may replace `shared.json` in the path field with a
   custom descriptive stem before saving.
8. Enable **Apply this saved calibration during preprocessing**.
9. Run preprocessing. Enable overwrite only for existing processed sessions
   that should be regenerated with the calibration.

Saved calibrations are ordinary versioned JSON files. The GUI creates a new file
by default and expands the placeholder `shared.json` to a name containing the
pulse frequency, session count, and UTC timestamp. A custom stem such as
`80mhz-standard` is useful when multiple optical or detector configurations use
the same repetition rate. Replacing an exact existing JSON requires disabling
versioned saving and confirming the replacement warning.

Current calibrations use schema 2.1, which records that the global fit included
independent nonnegative flat coefficients. Schema 2.0 and 1.0 files remain
loadable for reproducibility, but their global lifetimes were estimated without
that term. The GUI labels them as legacy. Current preprocessing still fits a
flat coefficient in every temporal bin when one of those older calibrations is
applied; rerun calibration to make the global lifetime/IRF estimation use the
current model as well.

## How the fit is performed

For each selected session, the software:

1. loads the iFLIP3 recording and its optional measured-background recording;
2. applies measured-background and afterpulse corrections;
3. restricts the data to the configured single-photon-counting window;
4. sums corrected decay curves across photometry time points;
5. divides the aggregate decay by its positive area so each session has
   comparable influence; and
6. contributes that normalized decay to a joint two-component-plus-flat fit.

The joint fit shares `tau1`, `tau2`, and IRF `sigma`. Each calibration session
receives its own `t0`, two aggregate sensor-component coefficients, and one
nonnegative flat-background coefficient. These linear coefficients are solved
again at every nonlinear optimization step, so the flat term directly affects
the global lifetime and IRF estimates. Unweighted residuals are used because
corrected decays may contain negative bins and the session normalization is
intended to balance session influence.

The calibration records its acquisition time axis and excitation pulse
interval. Preprocessing rejects a recording that does not match either value,
rather than silently applying an incompatible basis.

The sync rate stored in an iFLIP3 header can be a measured value, so nominally
identical hardware may report very small differences. Pulse intervals within
0.1% are treated as the same acquisition rate. Material changes, such as 80 MHz
(12.5 ns) versus 40 MHz (25 ns), are rejected and require separate calibration
files. A rejection message reports both pulse intervals and sync rates.

## Choosing calibration sessions

Using all technically valid compatible sessions is the recommended default.
It covers different mice and dates and makes the shared estimate less dependent
on one unusually bright or clean recording. Area normalization prevents a long
or bright session from dominating solely because it contains more photons.

Use the custom selector when:

- detector, laser, filter, optical path, or timing settings changed;
- recordings have different lifetime-bin axes or pulse intervals;
- a hardware service or alignment change divides the experiment into distinct
  acquisition periods;
- a session has saturation, clipping, a malformed decay, missing correction
  data, or a clear structured residual pattern; or
- a small pilot calibration is needed before the full dataset is available.

Do not select sessions based on whether they produce a preferred biological
result. Exclusions should be based on acquisition compatibility or technical
QC and should be recorded with the analysis notes. Different acquisition
configurations should have separate calibration files.

A calibration requires at least two sessions. For a definitive experiment-wide
calibration, include multiple mice and recording dates where possible.

## Interpreting QC

The decay overlay shows normalized observed curves and their fitted curves. The
residual heatmap makes systematic late-bin offsets easier to detect. The table
reports each session's decay peak, fitted `t0`, offset from the cohort median,
normalized RMSE, R-squared, aggregate component fractions, maximum absolute
residual relative to the decay peak, RMSE relative to the peak, and absolute
residual area relative to total decay area. The table also reports the aggregate
flat-background fraction. Large `t0` offsets are highlighted for review; they
are retained in the fit and are not silently excluded.

Pay particular attention to:

- residuals with the same shape in most sessions, which suggest model or
  correction misspecification;
- a single session with a shifted residual pattern, which may indicate a timing
  or configuration difference;
- persistent late-bin offsets, which may indicate incomplete background
  correction; and
- sessions with unusually large flat-background fractions, especially when
  those fractions differ systematically by acquisition day;
- strong covariance between the flat term and a very long `tau2`, which can
  occur when a long periodic decay looks nearly constant over one pulse cycle;
- sessions without a measured-background recording, because the flat term can
  absorb only a DC offset and cannot reproduce a shaped detector background.

High R-squared alone is not sufficient. Structured residuals can remain even
when the overall fit statistic is high.

## Applying and preserving a calibration

The calibration JSON stores the shared parameters, correction settings,
compatible time axis, pulse interval, contributing session identities,
session-level QC, code commit, and creation time. Keep this file with the
analysis configuration and do not edit values manually.

When a saved calibration is selected, preprocessing bypasses the per-session
lifetimes/IRF fit. It fixes `tau1`, `tau2`, and IRF `sigma`, uses the saved `t0`
for a contributing calibration session, or estimates one bounded `t0` from a
new session's aggregate decay. That session-level `t0` is then fixed while only
short-component, long-component, and nonnegative flat-background counts are
solved at each time point. Direct MPET remains calculated from corrected photon
histograms; fitted sensor MPET is reconstructed after omitting the flat
term. The calibration identity, fitted session `t0`, flat-background policy,
and parameters are embedded in each processed file's provenance.

Existing processed files are not changed unless **Overwrite existing processed
files** is enabled. This permits incremental batch processing, but sessions
processed with an older fitting policy must be selectively overwritten if they
should use the new calibration.

## Single-session preview

The default single-session diagnostic applies the current shared calibration to
the selected recording. It therefore displays the same fixed-basis model used
during calibrated preprocessing: shared lifetimes and IRF width; one fixed
recording-level `t0`; and independently varying nonnegative short, long, and
flat-background coefficients. The inspector plots the flat-background fraction
and compares direct MPET with fitted sensor MPET.

An **Advanced: unconstrained single-session fit** expander remains available to
diagnose an outlier or support a legacy dataset without a multi-session
calibration. Differences between its locally fitted parameters and the shared
calibration are diagnostic; they do not override the saved calibration. When a
saved calibration is enabled, preprocessing ignores all advanced local-fit
controls.
