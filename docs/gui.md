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
  and likely background files. The user must choose the matched background;
  the app presents candidates but does not silently decide which background is
  scientifically appropriate. Its data root defaults to `Z:\`, which contains
  both `FLIM FLIP` and `Photometry`.

The path assistant accepts descriptive text appended to a standard lifetime
filename. When more than one file matches, it provides a dropdown. Choose
**Enter a different path...** to type or paste an unusual final filename. The
chosen paths are copied into the manifest as paths relative to the data root
when possible, so the saved CSV remains readable and portable. The table below
the assistant remains directly editable.

Explicit relative paths are resolved from the selected data root. Absolute
paths override discovery. Running files remain optional for lifetime analyses;
the recording, synchronization, and NI-DAQ behavior inputs are required.

## Conventional photometry coverage

The interface supports session-manifest editing, batch preprocessing,
event-aligned PSTHs, heatmaps, and behavioral GLMs. Heatmaps use the same trial
selection, time window, and normalization as their PSTH and can preserve event
order or sort rows by their mean post-event response. PSTHs can align to
lick-bout onset or offset, use only the first event in each session, and retain
partial recording-boundary windows with unavailable samples left missing.
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
session-, mouse-, and group-level metrics, an editable performance figure, and
analysis metadata.

Statistics, specialized figure assembly, and result browsing remain
command-line or notebook workflows. The GUI is a focused front end, not a
replacement for every installed command.

## FluoPulse and iFLIP3 coverage

The two lifetime modes share the sensor-independent session representation in
`lutaslab-core`, but retain sensor-specific loading and correction:

- **Preprocess and align** fits the sensor-to-NI-DAQ clock transform, then saves
  an analysis-ready compressed `-processed.npz` file beside the primary raw
  lifetime recording. FluoPulse stores vendor tau, amplitude, and fit R-square;
  iFLIP3 stores background-corrected MPET and raw intensity for QC. Source paths,
  processing metadata, and behavior events are retained in the processed file.
  Existing processed files are reused unless overwrite is selected.
- **Event-aligned PSTH** aligns a selected lifetime/QC signal to Ensure, visual
  cue, individual lick, lick-bout onset, or lick-bout offset events. It can use
  every event or only the first event in each session. Long windows can retain
  the available portion of recordings that end before the requested window,
  representing the unavailable tail as missing data rather than excluding the
  event. It writes the mouse-level mean and session, equally
  weighted mouse-level, and descriptive pooled-trial heatmaps. Lifetime
  heatmaps support recorded order, response magnitude, Ensure latency,
  first-lick latency, post-event lick count, and pre-event lick rate.
- **PSTH, heatmap, and GLM analyses** require and load the processed lifetime
  files. They do not reopen or recompute the raw sensor recordings.
- **Lifetime GLM** estimates forward-time lick and Ensure kernels, a separate first
  Ensure response, and adaptation across subsequent deliveries. Ridge strength
  is selected inside each outer leave-one-session-out fold; performance is
  reported only on the untouched session and compared with a train-mean
  baseline. At least three sessions are required. The exported kernels and
  figures describe predictive associations, not causal effects.

The conventional random-onset/circular-shift null diagnostics currently apply
only to conventional photometry. Lifetime modes do not silently reuse those
controls because their sampling and correction pipelines differ.
