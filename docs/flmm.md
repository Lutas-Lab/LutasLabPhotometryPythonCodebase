# Functional mixed-effects models

The FLMM pilot is a group-level, event-aligned analysis shared by conventional
photometry, FluoPulse, and iFLIP3. It preserves each complete trial timecourse and
estimates a coefficient curve at every aligned time point while accounting for
repeated observations from the same mouse.

It complements rather than replaces the existing GLMs. The FLMM asks whether a
scalar trial label or trend is associated with the shape of an aligned response.
The conventional Behavioral GLM and Lifetime GLM ask predictive lag/kernel
questions using continuous recording data. All FLMM coefficients are
associational.

## Supported pilot models

The GUI and `lutaslab-run-flmm` expose three guided effects:

- **Condition:** condition plus standardized trial and session progress, with a
  mouse random intercept.
- **Group:** group plus standardized trial and session progress, with a mouse
  random intercept.
- **Within-session trial progress:** standardized trial and session progress,
  with a mouse random intercept.

Condition and group models require at least two retained levels and at least two
mice represented in every level. The selected manifest field must be populated
for every retained session; blank values are not treated as a scientific level.
Every model requires at least two mice; fewer than five mice produces a preflight
warning.
Only complete, finite event windows are retained. Boundary and normalization
failures are written to `flmm_exclusions.csv` rather than imputed.

Trial and session progression adjustments are included only when the retained
dataset contains variation in those covariates. A progression model therefore
cannot run when every retained session contributes only one usable event.

The concurrent/time-varying FLMM from the newer development fork is deliberately
not enabled. It needs a separate validation effort and would represent same-time
association, not a lagged behavioral response.

## Optional R installation

Dataset preparation is Python-only. Fitting requires R and the released
`fastFMM` package. The application detects `Rscript` from `PATH`, `R_HOME`, the
Windows registry, and standard Windows R installation folders. In an R or
RStudio console run:

```r
install.packages("fastFMM", dependencies = TRUE)
```

Restart or refresh the GUI after installing R or `fastFMM`; backend checks are
cached briefly. The base Python/GUI installer intentionally does not install or
embed R. The fit records the detected R and `fastFMM` versions in its output.

Check the optional backend from PowerShell with:

```powershell
uv run lutaslab-check-install --flmm
```

## GUI workflow

1. Preprocess the selected sessions as usual.
2. Open **Functional mixed model** and choose the response, event, window,
   normalization, and primary effect.
3. Avoid filtering to one condition when estimating condition, or one group when
   estimating group.
4. Choose **Prepare FLMM dataset** and inspect `flmm_preflight.json`,
   `flmm_exclusions.csv`, and `flmm_specification.json`.
5. When the design and exclusions are acceptable, choose **Run functional mixed
   model**.

After preparation, the tab reports retained mice, sessions, trials, exclusions,
and per-level replication. After fitting, it shows three complementary views:

- **Modeled responses:** the reference response plus each reconstructed condition
  or group response, or earlier/middle/later-trial predictions;
- **Population response and direct effects:** the intercept and difference or
  progression coefficient curves with pointwise and joint confidence bands; and
- **Mouse-to-mouse variation:** estimated random-intercept deviation curves for
  each mouse.

Use the direct coefficient and its joint band for formal whole-window inference.
The reconstructed response curves are descriptive because their uncertainty
would require the cross-coefficient covariance, which the export does not yet
carry through.

## Command line

Preparation without R:

```powershell
uv run lutaslab-run-flmm --workflow conventional `
  --manifest analysis/sessions.csv --data-root Z:\Photometry `
  --output-dir analysis/conventional/flmm --signal dff `
  --event cue_onset --effect condition --prepare-only
```

Remove `--prepare-only` to fit. For lifetime workflows, select `fluopulse` or
`iflip3` and use one of that workflow's processed signals and events.

## Outputs

Preparation writes:

- `flmm_input.csv`: frozen trial metadata, covariates, and outcome columns;
- `flmm_time.csv`: physical aligned time in seconds;
- `flmm_exclusions.csv`: every omitted event and its reason;
- `flmm_specification.json`: formula and preprocessing choices; and
- `flmm_preflight.json`: counts, levels, and sample-size warnings.

A successful fit adds the raw `flmm_fit.rds`, `flmm_coefficients.csv`, backend
version information, the R log, and PNG/SVG coefficient plots. It also writes
`flmm_fitted_responses.csv` and plots of the reconstructed modeled responses, plus
`flmm_mouse_random_effects.csv` and mouse-deviation plots when `fastFMM` returns
the requested random effects. Pointwise and joint 95% confidence bands are
exported separately. Prefer the joint band for claims about whether an effect
occurs anywhere across the full modeled time window.

This branch is an evaluation implementation. Before using it for a definitive
analysis, run a synthetic/direct-R parity check and a prespecified real dataset
from each acquisition method as described in
[the integration plan](flmm-integration-plan.md).
