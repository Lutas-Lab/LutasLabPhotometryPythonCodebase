# Functional mixed-model integration plan

Status: scalar FLMM pilot implemented on `codex/flmm-exploration`; concurrent FLMM
remains deferred.

The implemented pilot includes the common complete-trial contract, adapters for all
three acquisition workflows, preparation-only and fit CLI paths, an isolated Rscript
backend, stable coefficient tables/plots, exclusion and preflight artifacts, and a
shared fifth GUI tab. The currently exposed presets are condition, group, and linear
within-session trial progress, each adjusted for trial/session progress with a mouse
random intercept. The richer interactions, arbitrary scalar covariates, result browser,
simulation recovery, direct-R parity, and real-data acceptance runs below remain future
validation work rather than silently expanded production claims.

## Recommendation

Add functional linear mixed models (FLMMs) as a shared, trial-aligned group-analysis
workflow for conventional photometry, FluoPulse, and iFLIP3. Implement it in two
stages:

1. support the established non-concurrent FLMM for scalar trial/session covariates;
2. add concurrent FLMMs for time-varying covariates behind an explicit
   **Experimental** label after the scalar path and its validation are complete.

Do not replace the existing PSTH or temporal-basis ridge GLM workflows. FLMM answers
when a trial-level covariate is associated with a trial-aligned signal while accounting
for animal-level heterogeneity. The existing GLMs answer complementary predictive and
lag/kernel questions. In particular, a concurrent FLMM is instantaneous and should not
be presented as a substitute for a lagged behavioral GLM.

## What the external work provides

The original method fits a mixed model independently at every point on a shared
functional domain, smooths the coefficient curves, and constructs pointwise and joint
confidence bands. Its main practical benefit is retention of trial-level time courses
and animal-level random effects rather than reducing each trial to an AUC or peak.

The newer concurrent extension allows a covariate such as licking, speed, location, or
another signal to vary over time within each trial. Its coefficient at time `t` describes
the association between the outcome and covariate across trials/animals at that same
time `t`. It does **not** estimate a within-trial lagged response or convolution kernel.

Important implementation facts:

- CRAN `fastFMM` / `gloewing/fastFMM` contains the established non-concurrent method.
- The concurrent implementation used by the 2026 reviewed preprint is currently in
  `awqx/fastFMM`, a development fork, not the linked main repository or CRAN release.
- `fast-fmm-rpy2` still requires a separately installed R runtime and R package. Its
  public convenience call exposes only a narrow subset of `fui()` arguments, so it is
  not a sufficient backend contract for concurrent models without additional bridge
  code.
- The current lab workstation/environment has neither R, `rpy2`, nor
  `fast-fmm-rpy2` installed.
- `fastFMM` is GPL-3-or-later; the Python wrapper/tutorial repository is CC0. Keep the
  R package as an external, pinned runtime dependency and do not copy its source into
  this repository.

## Scientific scope and defaults

### Common trial definition

Every model uses one row per event-aligned trial and one column per common time point.
The Python layer must create and save a frozen analysis table before invoking R:

- identity: mouse, session, within-session trial index, source event index;
- manifest fields: group and condition;
- outcome matrix: `n_trials x n_timepoints`;
- scalar covariates: condition, group, centered trial/session number, behavioral
  summaries, and user-approved interactions;
- optional functional covariates: one `n_trials x n_timepoints` matrix per covariate;
- time grid, alignment event, preprocessing/normalization choices, and provenance.

Use seconds as the functional domain, matching `lutaslab-core`. Require a regular,
common grid within a model. Never pool different sensor outcomes into the same response
model merely because they share a clock.

The current PSTH option to retain partial boundary windows produces missing values.
Analytic `fastFMM` drops a whole trial when its outcome contains missing points, while
its FPCA imputation is documented as insufficiently tested. Therefore the first release
should default to complete windows, report every excluded trial, and not enable
imputation. A later option may trim all trials to their common supported domain.

### Outcome and QC mapping by acquisition method

| Workflow | Primary outcomes | Secondary/QC use |
| --- | --- | --- |
| Conventional | selected-channel dF/F; raw 465 as a sensitivity analysis | 405/reference signal can be explored as a concurrent nuisance covariate, not silently regressed twice |
| FluoPulse | `tau` | amplitude as a secondary outcome/covariate; fit R-squared as a prespecified QC filter or covariate |
| iFLIP3 | `mpet`; optionally long-lifetime fraction | raw intensity and component counts as secondary outcomes/covariates; lifetime-fit RMSE as QC |

Defaults should be scientifically conservative: analyze one named outcome at a time,
record the exact processed source, and apply the same trial-building code after the
sensor-specific loader has produced aligned data.

### Initial model presets

Expose guided presets rather than making an R formula the only interface:

- **Condition difference:** condition plus centered trial/session number, random mouse
  intercept.
- **Continuous trial covariate:** a selected scalar behavioral value plus centered
  trial/session number, random mouse intercept.
- **Condition by trial/session:** condition, centered progress variable, and their
  interaction, random mouse intercept.
- **Instantaneous behavior (experimental):** a time-varying covariate such as lick
  rate or running speed, with scalar session/trial adjustment and a justified random
  slope when the dataset supports it.

The formula builder should always show the resulting formula and reference levels.
An advanced raw-formula field can follow later. Do not automatically choose a maximal
random-effects structure: warn on low mouse/session counts and singular or failed
timepoint fits. A nested session-within-mouse sensitivity model can be offered when the
design supports it.

## Proposed code architecture

### 1. Sensor-independent trial contract in `lutaslab-core`

Add `lutaslab_core.functional_trials` with NumPy-based dataclasses such as:

- `FunctionalTrialDataset` (`time`, `outcome`, identities, scalar covariates,
  functional covariates, exclusions, metadata);
- `FunctionalModelSpec` (terms, interactions, random-effects preset, inference and
  smoothing settings);
- validators for shape, common grid, finite values, reference levels, identifiability,
  and per-timepoint covariate variance.

Keep this layer free of R, `rpy2`, Streamlit, and sensor-specific names. Reuse
`extract_perievent_trials` for interpolation and boundary logic, but retain trial
identity rather than returning only a pooled matrix.

### 2. Acquisition adapters in `lutaslab-photometry`

Add `lutaslab_photometry.flmm_dataset` as the orchestration layer because this package
already owns the shared CLI/GUI workflows:

- conventional adapter for the current processed-session dictionary;
- FluoPulse and iFLIP3 adapters using `load_processed_lifetime_session` and
  `AlignedSession`;
- one signal/event registry used by both the CLI and GUI;
- scalar feature builders for trial number, session number, event latency, lick count,
  lick rate, bout properties, group, and condition;
- functional feature builders for binned licking and interpolated running speed, with
  signal-specific additions only when requested.

Longer term, conventional processed sessions can gain an `AlignedSession` adapter, but
that migration is not required to prove the FLMM workflow.

### 3. Isolated R backend

Prefer a small `Rscript` adapter over embedding R in the Streamlit process for the
first implementation. This avoids making `rpy2` ABI/loading failures fatal to the GUI
and allows the Python test suite to mock the process boundary.

The backend contract should:

1. write the frozen wide analysis table, model specification, and provenance;
2. run a checked-in R entry script against a dedicated, pinned R library;
3. call `fastFMM::fui()` with explicit arguments;
4. save the raw fitted object as RDS for audit/reanalysis;
5. export stable, language-neutral tables for fixed-effect curves, pointwise and joint
   bands, AIC/BIC, warnings/failures, and optional random effects;
6. return a machine-readable completion/error record to Python.

Use an `flmm` optional installation path and add `lutaslab-check-install --flmm` (or
equivalent) to report R, package version/commit, and backend readiness. The base GUI
installation must remain usable without R. Pin the scalar backend to a released
`fastFMM` version and the concurrent backend to a reviewed commit of `awqx/fastFMM`;
record both hashes in every result.

### 4. Python result and plotting layer

Add `lutaslab_photometry.functional_mixed` to normalize backend output and create lab
figures without depending on `plot_fui.R`:

- one coefficient panel per term with zero line;
- pointwise and joint 95% bands with visually distinct styling;
- marked time regions where the joint band excludes zero, without converting them into
  multiple uncorrected pointwise claims;
- trial counts over time, exclusions, reference levels, formula, convergence warnings,
  and package versions;
- CSV/NPZ numeric results plus JSON metadata and the raw RDS fit.

The report must label all models as associational. Concurrent plots must say
"same-time association" and link to the model specification.

### 5. CLI

Add a single cross-acquisition entry point, for example:

```text
lutaslab-run-flmm --workflow conventional|fluopulse|iflip3 \
  --manifest ... --data-root ... --output ... \
  --signal ... --event ... --window -5 20 --dt 0.1 \
  --preset condition --condition ...
```

Keep dataset construction and fitting as separate callable functions. Support a
`--prepare-only` mode so a model table can be inspected before R is installed or a fit
is launched. Add `--concurrent` only after the development backend is pinned and
validated.

## GUI plan

Add a fifth **Functional mixed model** tab to each of the three acquisition modes.
Render it through one shared helper so scientific controls do not drift between modes.

The tab should present:

1. **Question and data:** outcome, alignment event, window, time step, group/condition
   subset, normalization, and complete-window policy.
2. **Model:** guided preset, fixed covariates, interactions, reference levels, and
   random-effects preset. Show the generated formula continuously.
3. **Concurrent covariates (Experimental):** collapsed/disabled until the concurrent
   backend passes validation; explain that effects are instantaneous, not lagged.
4. **Inference and compute:** analytic inference initially, spline settings under
   Advanced, core count capped safely, seed, and output directory.
5. **Preflight:** counts of mice/sessions/trials by group/condition, missing/excluded
   trials, outcome coverage, zero-variance timepoints, rank/identifiability checks,
   installed R/backend versions, and estimated model count (`n_timepoints`).
6. Separate **Prepare/preview** and **Run model** actions, following the existing GUI
   safety pattern. Run stays disabled when preflight fails.

Result browsing should show coefficient figures, warnings, exclusions, and links to the
frozen input/model metadata. A failure at one timepoint should be visible and auditable,
not silently smoothed away.

## Validation and tests

### Python-only tests (required in normal CI)

- exact trial identity/order and scalar/functional covariate construction;
- interpolation onto a common grid and complete-window exclusion;
- consistent datasets from conventional, FluoPulse, and iFLIP3 synthetic sessions;
- categorical reference coding, centering, interactions, and formula escaping;
- zero variance, missingness, insufficient biological units, and invalid random-effects
  structures;
- backend command/spec serialization and friendly missing-R errors;
- output parsing/plotting from frozen backend fixtures;
- CLI and Streamlit AppTest coverage for all workflows.

### R integration tests (optional marked job)

- reproduce a small published/tutorial non-concurrent fit within numeric tolerance;
- compare exported estimates/bands with a direct R fit;
- verify Windows path/quoting and parallel behavior;
- save and reload the raw RDS artifact;
- test the exact pinned concurrent fork separately from the CRAN backend.

### Scientific validation

- simulation recovery for known fixed curves and animal heterogeneity;
- empirical coverage checks for joint bands under a small set of realistic sample/noise
  regimes;
- sensitivity to time bin, smoothing knots, outcome preprocessing, and random-effects
  structure;
- compare FLMM conclusions with mouse-level PSTH summaries and held-out GLM results,
  explaining disagreements rather than selecting the most favorable analysis;
- validate one real dataset from each acquisition method before enabling the GUI Run
  button by default.

## Delivery sequence and acceptance gates

### Phase 0: runtime and method spike

- Install R only in an isolated development environment.
- Pin and run CRAN `fastFMM`; separately pin the concurrent fork.
- Prove CSV/spec -> Rscript -> tables/RDS on Windows.
- Decide the exact scalar and functional covariates supported in version 1.

Gate: one reproducible synthetic fit and one tutorial fit, with a documented clean
install and no changes to the base GUI environment.

### Phase 1: common dataset builder

- Implement the core contract and all three acquisition adapters.
- Add prepare-only CLI output and preflight diagnostics.

Gate: modality-independent golden datasets and complete Python test coverage of
exclusions/provenance.

### Phase 2: non-concurrent FLMM MVP

- Implement the pinned scalar R backend, normalized result tables, plots, CLI, and
  metadata.
- Validate condition, continuous scalar, and progress/interactions presets.

Gate: direct-R parity and successful real-data pilot for all three acquisition modes.

### Phase 3: GUI

- Add the shared tab, formula preview, preflight, run/progress display, and result links.
- Extend installer/check documentation without making FLMM mandatory.

Gate: Streamlit tests for all modes plus a clean Windows installation test.

### Phase 4: concurrent FLMM experiment

- Add time-varying licking/running covariates, variance-domain checks, and the pinned
  development backend.
- Explicitly compare same-time cFLMM estimates with the existing lagged/kernel GLM on
  the same data.

Gate: simulation coverage, one justified real use case, and a written interpretation
review. Retain the Experimental label while the method remains tied to the development
fork and the reviewed preprint's evidence is assessed as incomplete.

## Decisions to make before implementation

1. Which first scientific contrast should be the reference acceptance dataset for each
   modality?
2. Should conventional primary analysis use dF/F, raw 465 plus an explicit 405
   covariate, or require both as sensitivity analyses?
3. Which manifest columns represent within-animal versus between-animal conditions in
   current experiments?
4. Is the lab willing to support a separate R installation on GUI workstations, or
   should FLMM runs be prepared locally and executed in a controlled analysis
   environment?
5. Which concurrent question is sufficiently important to justify the development-fork
   dependency: licking, running speed, task phase/location, or cross-sensor signal?

## Sources reviewed

- https://github.com/gloewing/photometry_FLMM
- https://github.com/gloewing/fastFMM
- https://github.com/awqx/fastFMM
- https://github.com/nimh-dsst/fast-fmm-rpy2
- https://elifesciences.org/articles/95802
- https://elifesciences.org/reviewed-preprints/109428
- https://elifesciences.org/reviewed-preprints/109428v1/reviews
