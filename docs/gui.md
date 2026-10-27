# Streamlit interface

The Streamlit app is one browser interface for conventional photometry,
FluoPulse lifetime recordings, and iFLIP3 lifetime recordings. Select the
acquisition system in the sidebar; Streamlit reruns the page with that system's
manifest and controls while preserving each mode's current state. The app calls
the same maintained Python and command-line workflows available to notebooks.

## Windows installer

For a lab computer without an existing project environment:

1. Install 64-bit Python 3.12 and enable **Add python.exe to PATH**.
2. Download or clone this repository.
3. Double-click `install_gui.bat` once.
4. Double-click `launch_gui.bat` to start the interface.

The installer creates `.venv-gui` inside the repository, separate from the
standard developer environment at `.venv`. It installs the version of `uv`
required by the repository and resolves the application from the checked-in
`uv.lock`; it does not modify another Python environment. Internet access is
required for the initial installation. If the repository is moved or renamed,
run `install_gui.bat` again. Do not copy `.venv-gui` between computers.

At the end of installation, the installer imports the required packages and
loads the full Streamlit app in its test harness. A successful installer run
therefore checks more than package availability.

`install_gui.ps1` contains the PowerShell installation logic; the batch file is
the double-clickable wrapper.

## Launch from the uv environment

Users of the standard project environment can install the GUI extra and launch
the app directly:

```powershell
uv sync --frozen --all-packages --extra gui --extra forecasting --extra events
uv run python -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

The interface is bound to the local computer (`127.0.0.1`) rather than exposed
to other devices on the network. Each workflow has separate **Preview** and
**Run** buttons, so previewing a command can never start analysis. Before a Run
action starts, the app validates and atomically saves the current table to the
manifest path; unsaved editor changes are therefore included. Console output
appears while the command runs, with the latest 500 lines retained in the app.

Keep the browser tab open while a workflow runs. The current interface shows
live progress but does not yet provide a process-cancellation control. Existing
processed files remain protected unless **Overwrite existing processed files**
is explicitly enabled.

## Workflow selector and manifests

Each acquisition system retains a separate manifest and output directory:

- **Conventional photometry** uses mouse, date, run, group, condition, and
  photoreceiver channel. Its data root defaults to `Z:\Photometry`.
- **FluoPulse** uses mouse, date, and run to search `Z:\FLIM FLIP` and
  `Z:\Photometry`. The session path assistant presents matching Doric, NI-DAQ,
  and optional running files before adding the row.
- **iFLIP3** uses the same identity fields to find the iFLIP3, NI-DAQ, running,
  and likely background files. A matched background can be selected when one
  exists, but older recordings can leave it blank. Its data root defaults to
  `Z:\`, which contains both `FLIM FLIP` and `Photometry`.

The path assistant accepts descriptive text appended to a standard lifetime
filename. When more than one file matches, it provides a dropdown. Choose
**Enter a different path...** to type or paste an unusual final filename. The
chosen paths are copied into the manifest as paths relative to the data root
when possible, so the saved CSV remains readable and portable. The table below
the assistant remains directly editable.

The PSTH and GLM tabs provide **Group to analyze** and **Condition to analyze**
menus populated from the active manifest. They default to all rows. Selecting a
label filters only that analysis run; it does not alter the master manifest or
reprocess the recordings. These selectors are available for conventional
photometry, FluoPulse, and iFLIP3.

Every GUI PSTH run is saved in a new timestamped folder below the selected
**Figure/output base directory**. The suggested folder label records the selected
group or condition, signal, alignment event, and time window; enter a custom
**Run label** when a more descriptive experimental name is useful. For example,
separate BLA and NAc selections produce separate run folders even when every
other setting is unchanged. Lifetime runs also write `analysis_metadata.json`
with the complete plotting configuration, included sessions, subset labels,
generation time, and code commit. The advanced option to save directly into the
base directory restores the older behavior and can replace matching files, so it
should be selected only for an intentional overwrite.

Every PSTH run also writes portable numeric outputs without requiring the
Summary statistics tab:

- `psth_session_timecourses.csv` contains one trace per session in long format;
- `psth_mouse_timecourses.csv` contains session-balanced mouse traces;
- `psth_group_timecourse.csv` contains mouse-level mean, SEM, and N;
- `psth_prism_timecourses.csv` is a wide XY table with time in the first column
  and one column per group/condition/mouse; and
- `psth_timecourses.npz` and `psth_timecourse_metadata.json` preserve compact
  arrays and the analysis identity for programmatic reuse.

For stratified conventional analyses, the run root contains a combined Prism
timecourse table spanning all conditions, while each stratum keeps its own copy.
These exports use trials within sessions, sessions within mice, and mice as the
biological units.

PSTH controls include hover explanations for the alignment window, time bin,
normalization, and baseline interval. Enable **Use fixed y-axis limits for the
primary PSTH** and enter a minimum and maximum when figures from separate runs
must use an identical vertical scale. The limits apply to the selected
photometry or lifetime signal and to conventional condition-comparison PSTHs.
Optional licking and running outputs retain automatic scaling because they use
different physical units.

The conventional, FluoPulse, and iFLIP3 GLM tabs provide hover explanations for
every response/source choice and numerical model control. These descriptions
distinguish processed and raw photometry responses, define lifetime-derived
responses, and explain kernel durations, time grids, validation folds, and
output locations.

GLM runs stream their current preparation, model, validation-fold, session-kernel,
and output-writing stage to the GUI progress bar. Conventional photometry reports
each model and outer fold. FluoPulse and iFLIP3 report nested session folds and
per-session kernel fits. The lifetime ridge search reuses each training fold's
normal equations across candidate regularization strengths; this preserves the
same nested validation results while avoiding repeated matrix cross-products.

## Saved-PSTH summary statistics

The **Summary statistics** tab sits between PSTH and GLM for conventional,
FluoPulse, and iFLIP3 analyses. Select one completed PSTH run—or its root folder
containing compatible group/condition subfolders—then choose the response
window, peak-smoothing duration, metrics, and mouse-level comparison method.
The preview confirms the saved signal, event, mouse count, and group curves
before analysis. Raw or processed recordings are not reopened.

Mean and AUC use the unsmoothed mouse timecourse. Smoothing is used only to
locate peaks and troughs. Outputs include session and mouse metric tables, group
mean/SEM/N, Holm-adjusted pairwise tests, publication-style figures, one
all-metric Prism table, and a separate Prism-wide CSV for each metric. In the
per-metric files, each row is a mouse and each group/condition is a column, so
paired mice remain aligned for direct import into GraphPad Prism. A new
timestamped output folder is created for each summary run.

Explicit relative paths are resolved from the selected data root. Absolute
paths override discovery. Running files remain optional for lifetime analyses;
the recording inputs are required. NI-DAQ is optional for both lifetime sensors:
when no corresponding file exists, preprocessing keeps the sensor-native clock
and omits NI-DAQ-only behavior. An iFLIP3 matched background is also optional;
when absent, measured-background subtraction is skipped and that choice is
recorded in the processed metadata.

## Conventional photometry coverage

The interface supports session-manifest editing, batch preprocessing,
event-aligned PSTHs, heatmaps, and behavioral GLMs. Heatmaps use the same trial
selection, time window, and normalization as their PSTH and can preserve event
order or sort rows by their mean post-event response. PSTHs can align to
lick-bout onset or offset, use only the first event in each session, and retain
partial recording-boundary windows with unavailable samples left missing.
The **Photometry signal** selector offers IRLS ΔF/F (the default analysis
signal), 465 fluorescence, 405 isosbestic fluorescence, and the IRLS-fitted 405
control. The two 405 choices are labeled as QC controls in the GUI.
Optional checkboxes generate matching licking-rate and running-speed PSTHs in
separate output subdirectories during the same run. Licking remains in licks/s
and running remains in cm/s; neither behavioral signal is normalized. Heatmaps
are independently optional and, when enabled, are saved for every selected
response. Sessions without running data are skipped only for the running output.
A progress bar names the current session and reports progress across every
selected output.
Every run creates:

- one trial heatmap per session;
- a mouse-level heatmap with one row per mouse, after averaging trials within
  sessions and sessions within each mouse; and
- a pooled heatmap containing every trial across the included sessions.

The mouse-level heatmap preserves equal weighting of biological units. The
pooled heatmap is explicitly labeled descriptive because mice with more trials
or sessions contribute more rows and those rows are not independent units.

Heatmap rows can be ordered by recorded event order, response magnitude,
Ensure-delivery latency, first-lick latency, first-bout latency, bout size,
bout duration, post-event lick count, or pre-event lick rate. Lick-bout-aligned
analyses also offer cue-to-bout latency. An Advanced section controls the event
matching window, direction, and whether unmatched trials are retained at the
bottom or excluded. `heatmaps/heatmap_trial_order.csv` records the original and
sorted row, alignment time, matched event, sorting value, window, and unmatched
policy for audit and reuse.

The same implementation is available without the GUI:

```python
from lutaslab_photometry.heatmap_ordering import order_heatmap_trials

ordered = order_heatmap_trials(
    session_result,
    peri_time,
    sort="ensure_latency",
    sort_window=(0, 20),
    unmatched="bottom",
)
```

`save_psth_heatmaps` applies the same options to saved figures, while
`lutaslab-run-psth` exposes them through `--heatmap-sort`,
`--heatmap-sort-window`, `--heatmap-sort-direction`, and
`--heatmap-unmatched`.

## Null diagnostics

Selecting random-onset or circular-shift controls automatically creates a
`null_diagnostics/` directory containing:

- `null_diagnostics_summary.csv` and `null_diagnostics_metadata.json`;
- real-versus-null baseline-SD distributions;
- maximum absolute null-trial z-scores;
- null-center distances from real alignments, cues, Ensure deliveries, licks,
  and lick-bout onsets;
- null-amplitude comparisons for no normalization, subtraction, and z-scoring;
  and
- example normalized null trials.

The summary flags sessions when more than 1% of null baselines are near-zero or
more than 1% of null trials exceed an absolute z-score of 20. “Near-zero” is
defined relative to each session: at most 1% of its median positive real-trial
baseline SD. These flags are diagnostics, not automatic exclusion rules. The
GUI now exposes the null-event exclusion distance; overly large exclusions can
be impossible for densely spaced events.

The Behavioral GLM predicts present photometry from present and past signals
(`horizon = 0`). It compares a fold-local constant baseline, photometry-history,
behavior-only, and combined ridge models. Regularization is selected in nested
forward folds and all reported performance comes from untouched future outer
folds. The present response is never included among its own predictors:
photometry-history features are strictly in the past, while behavioral features
may include the current time bin. Raw 465 fluorescence is the default response;
processed dF/F is available as a sensitivity analysis. Results include
session-, mouse-, and group-level metrics and predictor-kernel tables. The GUI
also saves editable SVG and PNG figures for held-out performance and for the
combined model's photometry-history, running, licking, cue-onset, and
Ensure-onset kernels. Kernel figures show individual-mouse traces together with
the mouse mean and SEM; sessions are first averaged within each mouse. Kernel
coefficients are converted back to native predictor units after fitting. The
saved metadata records the modeling and summarization choices.

Statistics, specialized figure assembly, and result browsing remain
command-line or notebook workflows. The GUI is a focused front end, not a
replacement for every installed command.

## FluoPulse and iFLIP3 coverage

The two lifetime modes share the sensor-independent session representation in
`lutaslab-core`, but retain sensor-specific loading and correction:

- **Preprocess and align** fits the sensor-to-NI-DAQ clock transform when an
  NI-DAQ file is available; otherwise it retains the native sensor clock. It saves
  an analysis-ready compressed `-processed.npz` file beside the primary raw
  lifetime recording. FluoPulse stores vendor tau, amplitude, and fit R-square;
  iFLIP3 stores background-corrected MPET and raw intensity for QC. Source paths,
  processing metadata, and behavior events are retained in the processed file.
  Existing processed files are reused unless overwrite is selected.
  For iFLIP3, the recommended decay workflow is to create and review a saved
  [multi-session calibration](iflip3-multisession-calibration.md). It shares
  lifetimes and IRF width across compatible recordings, estimates one fixed t0
  per recording, fixes residual fit background to zero, and leaves only
  component counts time-point-specific.
- **Event-aligned PSTH** aligns a selected lifetime/QC signal to Ensure, visual
  cue, individual lick, lick-bout onset, or lick-bout offset events. It can use
  every event or only the first event in each session. Long windows can retain
  the available portion of recordings that end before the requested window,
  representing the unavailable tail as missing data rather than excluding the
  event. Lifetime τ and MPET are displayed in picoseconds, either unchanged or
  baseline-subtracted as Δτ/ΔMPET; lifetime PSTHs are not z-scored. It writes the
  mouse-level mean and session, equally
  weighted mouse-level, and descriptive pooled-trial heatmaps. Lifetime
  heatmaps support recorded order, response magnitude, Ensure latency,
  first-lick latency, post-event lick count, and pre-event lick rate. Optional
  checkboxes add licking-rate (licks/s) and running-speed (cm/s) PSTHs in separate
  output subdirectories, with running omitted only for sessions that lack a
  running trace. Heatmaps are controlled separately and apply to every selected
  response. The GUI reports session-level progress across all selected signals.
- **PSTH, heatmap, and GLM analyses** require and load the processed lifetime
  files. They do not reopen or recompute the raw sensor recordings.
- **Lifetime GLM** estimates forward-time lick and Ensure kernels, a separate first
  Ensure response, and adaptation across subsequent deliveries. Ridge strength
  is selected inside each outer leave-one-session-out fold; performance is
  reported only on the untouched session and compared with a train-mean
  baseline. At least three sessions are required. Outputs include held-out
  session and mouse performance, session/mouse/group kernel tables, an editable
  performance figure, and lick/Ensure kernel figures showing individual mice
  with mouse mean and SEM. Lifetime τ and MPET kernels are expressed in
  picoseconds. The exported kernels and figures describe predictive
  associations, not causal effects.

The conventional random-onset/circular-shift null diagnostics currently apply
only to conventional photometry. Lifetime modes do not silently reuse those
controls because their sampling and correction pipelines differ.

Running files that are empty or flat at zero are recoverable quality-control
exclusions. The GUI combines them into one concise warning with an expandable
list of affected files instead of displaying raw Python warning locations.
Other signals continue to process normally, and detailed running QC remains in
each processed session file.
