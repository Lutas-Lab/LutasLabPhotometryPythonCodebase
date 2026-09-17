# Fiber Photometry Analysis

Python tools for preprocessing, visualizing, and modeling fiber photometry and behavioral data.

This repository is designed to provide a reusable analysis pipeline while keeping the underlying analysis code separate from interactive notebooks and command-line workflows.

## Features

The current pipeline supports:

- 465 nm and 405 nm photometry demultiplexing
- 405-to-465 temporal alignment
- robust IRLS reference fitting
- IRLS-corrected dF/F
- preservation of raw 465 and 405 signals
- locomotion processing
- lick detection
- lick-bout detection
- visual cue detection
- solenoid/reward detection
- cue-lick trial classification
- Pynapple integration
- event-aligned photometry analysis
- trial-level visualization
- NeMoS behavioral GLMs
- temporal behavioral kernels
- causal/predictive models
- two-sided temporal-association models
- blocked cross-validation
- regularized multi-predictor models

---

# Installation

Python 3.12 is recommended for compatibility with the current NeMoS release.

## Quick start: Windows and Anaconda

First download the repository with GitHub Desktop, or clone the current
`main` branch:

```powershell
git clone https://github.com/Lutas-Lab/LutasLabPhotometryPythonCodebase.git
cd LutasLabPhotometryPythonCodebase
```

Open **Anaconda Prompt**, then create and install the analysis environment:

```text
conda create -n photometry python=3.12 pip jupyterlab ipykernel -y
conda activate photometry
python -m pip install -e .
python -m pytest
```

The final command is optional but verifies the installation. Core
preprocessing and plotting do not require JAX or NeMoS. Install those optional
modeling dependencies only when needed:

```text
python -m pip install -e ".[modeling]"
```

### Expected raw-data layout

`--data-root` is the directory containing one folder per mouse. Each session
must follow this layout and naming convention:

```text
Z:\Photometry\
└── DK21\
    └── DK21_230704\
        ├── DK21-230704-001-nidaq.mat
        └── DK21-230704-001-running.mat
```

The NIDAQ MATLAB file must contain `data`, `timestamps`, and `Fs`. With the
default channel map, rows 1–8 of `data` are photoreceiver 1, locomotion TTL,
photoreceiver 2, licking, visual cue, 465-nm TTL, 405-nm TTL, and solenoid TTL.
The running MATLAB file must contain `speed`.

Copy `config/sessions.example.csv` to `analysis/sessions.csv`, then replace the
example rows with the sessions to analyze. Dates use six digits (`YYMMDD`),
`run` is an integer, and `channel` is the photoreceiver containing the signal:

```csv
mouse,date,run,group,condition,channel
DK21,230704,1,control,naive,1
DK21,230705,2,control,trained,1
```

Run commands from the repository root. For example, in PowerShell:

```powershell
python scripts/run_preprocess_batch.py --manifest analysis/sessions.csv --data-root "Z:\Photometry" --continue-on-error
python scripts/run_psth.py --manifest analysis/sessions.csv --data-root "Z:\Photometry" --output-dir analysis/cue_psth_20s --event-key cue_onset --window -5 20 --baseline -5 0 --normalization zscore
```

## Quick start without PowerShell: JupyterLab

The complete batch workflow can also be launched from
[`notebooks/05_batch_workflow.ipynb`](notebooks/05_batch_workflow.ipynb). This
is often the easiest route for Windows users:

1. Download the repository with GitHub Desktop or **Code → Download ZIP** on
   GitHub.
2. In Anaconda Navigator, create an environment named `photometry` with Python
   3.12 and install `pip`, `jupyterlab`, and `ipykernel` in that environment.
3. Launch JupyterLab using the `photometry` environment.
4. Navigate to the repository and open `notebooks/05_batch_workflow.ipynb`.
5. Run the installation cell once, restart the kernel if requested, and then
   edit the configuration and session-list cells.

The notebook uses the active Jupyter kernel to run the same maintained scripts
documented below. It can create the session manifest, batch-preprocess data,
and generate cue-photometry, cue-licking, statistical, and
lick-bout/delivery figures without entering shell commands.

## Prototype browser interface

A Streamlit prototype provides a browser-based session editor and launchers
for batch preprocessing and event-aligned PSTHs. Install the optional GUI
dependency and start it from the repository root:

```text
conda activate photometry
python -m pip install -e ".[gui]"
python -m streamlit run streamlit_app.py
```

The interface opens locally in a browser. It uses the same maintained scripts
as PowerShell and Jupyter rather than reimplementing the analysis. **Preview
analysis commands only** is enabled by default, so preprocessing and PSTH
buttons show the exact command without running it. Manifest validation, saving,
and CSV downloads still work in preview mode. Disable preview mode only when
the manifest, data root, and output directory are correct. Existing processed
files remain protected unless **Overwrite existing processed files** is
explicitly enabled.

## Other environments

```bash
python -m venv .venv
python -m pip install -e .
```

Install modeling and development dependencies when needed:

```bash
python -m pip install -e ".[modeling,dev]"
python -m pytest
```

---

# Repository Structure

```text
photometry-analysis/
│
├── README.md
│
├── src/
│   ├── README.md
│   ├── __init__.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── plotting.py
│   ├── pynapple_utils.py
│   ├── save_sessiondata.py
│   └── nemos_analysis.py
│
├── scripts/
│   ├── README.md
│   └── run_preprocess.py
│
└── notebooks/
    ├── README.md
    └── 01_explore_session.ipynb
```

The three main directories serve different purposes:

```text
src/        reusable analysis implementation

scripts/    automated workflows

notebooks/  interactive exploration and examples
```

---

# Typical Workflow

The general analysis pipeline is:

```text
Raw photometry + behavior
            |
            v
       preprocessing
            |
            v
     processed session
            |
            v
       saved .npz
            |
      +-----+------+
      |            |
      v            v
 visualization   modeling
      |            |
      v            v
  notebooks      NeMoS
```

Raw data are processed once and saved as a standardized processed session.

The processed `.npz` files can then be reused for plotting, trial analysis, group analysis, and behavioral modeling without rerunning the raw-data preprocessing.

---

# Preprocessing a Session

Routine preprocessing is performed using the command-line wrapper in `scripts/`.

For example:

```bash
python scripts/run_preprocess.py --mouse DK21 --date 230704 --run 2 --data-root "Z:\Photometry"
```

The script:

```text
mouse / date / run
        |
        v
locate raw session files
        |
        v
load raw data
        |
        v
run preprocessing
        |
        v
save processed session
```

A processed session is saved using a standardized filename such as:

```text
DK21-230704-002-processed.npz
```

## Batch Processing and Group PSTHs

The same CSV session manifest can drive raw-data preprocessing and subsequent
mouse/group PSTH figures. Start by copying `config/sessions.example.csv` to a
local file under the Git-ignored `analysis/` directory:

```csv
mouse,date,run,group,condition,channel
DK21,230704,1,control,rewarded,1
DK21,230704,2,control,unrewarded,1
DK40,231005,1,experimental,rewarded,2
```

`group` and `condition` are optional for preprocessing and plotting, but enable
mouse-level statistical comparisons and Prism-ready exports. `channel` selects
photoreceiver 1 or 2 independently for each session during PSTH, statistics,
and forecasting analyses. Older manifests without this column default to
channel 1. Passing `--channel 1` or `--channel 2` explicitly overrides the
manifest for every session.

Batch preprocessing saves each processed file beside its original raw files:

```bash
python scripts/run_preprocess_batch.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry"
```

Existing processed files are skipped by default so that an old analysis is not
silently overwritten. Use `--overwrite` only after backing up results that must
be retained. Use `--continue-on-error` to attempt later sessions and report all
failures in one run.

Generate cue-aligned per-mouse figures and a group mean with SEM across mice:

```bash
python scripts/run_psth.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir "analysis/figures/cue" \
    --event-key cue_onset \
    --normalization zscore \
    --baseline -5 0
```

Cue trials can be filtered at analysis time without rerunning preprocessing.
For example, plot only trials with no licking during the cue or the following
four seconds:

```bash
python scripts/run_psth.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir "analysis/figures/cue_miss_4s" \
    --event-key cue_onset \
    --trial-class cue_miss \
    --post-cue-window 4
```

Available classes are `all`, `cue_lick`, `post_cue_lick`, `cue_only`,
`post_only`, `cue_and_post`, and `cue_miss`. The selected class and post-cue
window are recorded in numeric/statistical outputs. These masks are recomputed
from saved `cue_onset`, `cue_offset`, and `lick_times` arrays; the masks stored
during preprocessing remain available for provenance and older workflows.

Add a reproducible random-alignment control with:

```bash
python scripts/run_psth.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir "analysis/figures/cue_random" \
    --event-key cue_onset \
    --null-method random_onsets \
    --n-shuffles 500 \
    --seed 123
```

`random_onsets` samples the same number of valid onsets within each recording.
`circular_shift` moves each session's event train as a block and therefore
preserves its relative event spacing. Set `--null-exclusion` to require null
onsets to remain a chosen number of seconds away from real events. Figures show
the observed PSTH together with the shuffled mean and 95% null envelope.

Other timestamp arrays in a processed session can be selected with
`--event-key`, including `solenoid_onset`, `lick_bout_onset`, and `lick_times`.
Use `--normalization none` to plot processed dF/F without trial-local baseline
normalization.

Generate cue-aligned licking-rate PSTHs using the same group and condition
comparisons:

```bash
python scripts/run_psth.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir "analysis/licking_psth_20s" \
    --event-key cue_onset \
    --signal licking \
    --window -5 20 \
    --dt 0.1 \
    --normalization none
```

For licking, each trial is a histogram of lick timestamps expressed as licks
per second. Trials are averaged within sessions, sessions within mice, and mice
within groups. A moderate bin width such as 0.1 seconds is recommended.

To test whether Astrocyte photometry follows Ensure delivery timing rather than
licking itself, generate lick-bout-aligned PSTHs and delivery-sorted heatmaps:

```bash
python scripts/run_lickbout_delivery_analysis.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir analysis/astrocyte_lickbout_delivery_20s \
    --group Astrocyte \
    --window -5 20 \
    --baseline -5 0 \
    --minimum-delivery-latency 0 \
    --normalization zscore
```

Each behavioral trial extends from one cue onset to the next. The analysis
pairs the first lick bout and first solenoid onset in that interval, excludes
trials in which delivery preceded lick-bout onset, then uses the remaining
paired trials for both the PSTH and heatmap. Heatmap rows are sorted by
`solenoid_onset - lick_bout_onset`. A white overlay marks the solenoid/Ensure
delivery time on each row. The sorted trial matrix and matching metadata are
also exported as NPZ and CSV files.

The averaging hierarchy is deliberately:

```text
events -> session mean -> mouse mean -> group mean +/- SEM across mice
```

Thus, a mouse with more sessions or trials does not receive more weight in the
group-level result. The figure workflow also saves the numeric mouse matrix,
group mean, and group SEM to `psth_results.npz`, plus counts to
`psth_summary.csv`. When randomization is enabled, the shuffle matrices, null
mean, percentile bounds, method, seed, and shuffle count are also saved.

When `group` and `condition` are present, `run_psth.py` separates them by
default rather than averaging conditions together. For example, outputs are
written under `Astrocyte/Naive`, `Astrocyte/Trained`, `D1/Naive`, and
`D1/Trained`. The `comparisons` folder contains one figure per group with the
condition PSTHs and a second panel showing each mouse's paired Trained-minus-
Naive trace plus the group mean and SEM. Pass `--no-stratify` only when a
deliberately condition-combined PSTH is desired.

### PSTH response statistics

Extract predefined response metrics and run statistics with mice, rather than
trials, as the independent biological units:

```bash
python scripts/run_psth_statistics.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir analysis/statistics/cue \
    --event-key cue_onset \
    --baseline -5 0 \
    --response-window 0 2 \
    --metrics mean auc peak peak_latency \
    --test auto
```

The workflow calculates metrics for every trial, summarizes session PSTHs,
averages sessions within each mouse, and only then performs group comparisons.
Mean, signed/positive/negative AUC, peak, trough, and peak/trough latency are
available. Peak and trough measurements use configurable light smoothing;
mean and AUC use the unsmoothed response.

With `--test auto`, conditions measured in the same mice use paired t-tests and
disjoint groups use Welch tests. Wilcoxon and Mann-Whitney alternatives can be
requested explicitly. Results include effect sizes, 95% confidence intervals,
raw p-values, and Holm-adjusted p-values. Add `--null-method random_onsets` or
`circular_shift` for two-sided empirical tests against shuffled alignments.

Outputs include long-format trial, session, mouse, and group tables; statistical
and shuffle-test tables; and `psth_prism_wide.csv`, which has one row per mouse
and one column per group/condition/metric. The baseline and response windows and
primary metric should be selected before comparing experimental conditions.

### Publication figures and Adobe Illustrator

PSTH, response-metric, and forecasting commands save both editable SVG figures
and 300-DPI PNG previews by default. SVG text remains text rather than being
converted to paths, which makes labels and fonts editable in Adobe Illustrator.
Use `--formats svg png pdf` to request all supported outputs and
`--font-family Arial` to select an installed font.

Response-metric figures show individual mice, connect repeated measurements,
overlay the group mean with a 95% confidence interval, and display
Holm-adjusted statistical comparisons. Use `--no-mouse-points`, `--no-pairs`,
or `--no-statistics` when preparing a different presentation. Python determines
the data and statistics; Illustrator can then be used for final panel layout and
cosmetic editing without changing the underlying analysis.

The actual preprocessing implementation is contained in:

```text
src/load_data.py
src/preprocess.py
src/save_sessiondata.py
```

See `scripts/README.md` for additional information about command-line workflows.

---

# Exploring a Session

Interactive exploration is performed using the example notebooks.

The primary starting point is:

```text
notebooks/01_explore_session.ipynb
```

A session can be selected using:

```python
MOUSE = "DK21"
DATE = "230704"
RUN = 2
```

The notebook demonstrates how to:

- load a processed session
- inspect recording duration and sampling rates
- inspect raw 465 and 405 photometry
- inspect processed dF/F
- inspect locomotion
- inspect lick events
- inspect visual cue events
- inspect solenoid/reward events
- compare raw 465 with processed dF/F
- optionally prepare photometry for NeMoS modeling

See `notebooks/README.md` for additional information.

---

# Working Analysis Notebooks

The notebooks committed under:

```text
notebooks/
```

are intended to serve as clean, reusable templates and examples.

For actual mouse- or session-specific analyses, create a local directory called:

```text
analysis/
```

at the repository root.

For example:

```text
photometry-analysis/
│
├── README.md
├── src/
├── scripts/
├── notebooks/
│   ├── 01_explore_session.ipynb
│   └── 02_event_aligned_photometry.ipynb
│
└── analysis/
    ├── DK21_230704_002.ipynb
    ├── DK21_230704_002_events.ipynb
    └── DK40_231005_001.ipynb
```

The `analysis/` directory is intentionally excluded from Git using `.gitignore`.

This allows the notebooks in `notebooks/` to remain clean templates while local working notebooks can contain:

- mouse-specific paths and settings
- exploratory analyses
- generated figures
- notebook outputs
- temporary code
- analysis notes
- session-specific results

A typical workflow is to copy a template notebook:

```text
notebooks/01_explore_session.ipynb
```

into:

```text
analysis/DK21_230704_002.ipynb
```

and perform the session-specific analysis in the copied notebook.

The repository `.gitignore` should contain:

```gitignore
analysis/
```

so these working notebooks are not committed to GitHub.

Reusable improvements discovered while working in `analysis/` should be moved into the appropriate `src/` module or incorporated into the clean template notebook under `notebooks/`.

# Photometry Processing

The preprocessing pipeline separates the interleaved 465 nm and 405 nm photometry measurements using their LED TTL signals.

The 405 nm signal is then interpolated onto the 465 nm timebase.

Robust iteratively reweighted least squares (IRLS) regression is used to fit the aligned 405 nm signal to the 465 nm signal.

The IRLS-corrected signal is calculated as:

```text
dF/F = (465 - fitted 405) / fitted 405
```

Multiple representations of the photometry signal are retained, including:

```text
raw 465
raw 405
aligned 405
IRLS-fitted 405
IRLS dF/F
```

IRLS-corrected dF/F should be considered **one available photometry representation rather than automatically assumed to be optimal for every sensor or recording**.

For some sensors or sessions, the 405 nm channel may not provide an ideal nuisance reference.

Retaining raw 465 fluorescence allows alternative approaches to be evaluated, including:

- trial-local normalization
- slow-trend modeling
- alternative detrending strategies
- raw-465 behavioral GLMs

---

# Behavioral Processing

The preprocessing pipeline also extracts behavioral and task variables.

These currently include:

```text
locomotion
individual licks
lick bouts
visual cue onset and offset
solenoid/reward onset and offset
cue-related licking classifications
```

Cue trials can be classified according to whether licking occurs:

- during the cue
- after cue offset
- during both periods
- during neither period

Actual cue onset and offset timestamps are used so that analyses can accommodate experiments with different cue durations.

For cue-aligned PSTHs and statistics, classification is performed at analysis
time from the saved cue and lick timestamps. This allows the post-cue response
window to be changed without reprocessing raw photometry.

---

# Forecasting Future Photometry and Behavior

Forecasting is available without NeMoS or JAX through the optional
`forecasting` dependency:

```bash
python -m pip install -e ".[forecasting]"
```

The manifest-driven forecasting script supports future `photometry`,
`locomotion`, `lick_binary`, and `lick_count` targets. For example:

```bash
python scripts/run_forecasting.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir analysis/forecasts/licks \
    --target lick_binary \
    --horizons 0.5 1 2 5 \
    --history 5 \
    --target-window 1
```

Every target and feature row has an explicit prediction time. Only signals at
or before that time enter the design matrix. Evaluation uses expanding-window
cross-validation: training data always precede testing data, and a temporal gap
separates them. Unless explicitly overridden, that gap covers predictor history,
forecast horizon, and the future lick-counting window.

Each forecast compares:

```text
history_only  target's own past
cross_modal   photometry/behavior signals other than the target's own past
combined      target history plus cross-modal signals
```

Continuous targets use ridge regression, future lick occurrence uses logistic
regression, and future lick counts use Poisson regression. Raw 465 is the
default photometry representation so forecasting does not depend on a
whole-session 405 fit; `--photometry-source dff` is available as a secondary
comparison.

Outputs include session-level metrics, mouse-level means, group mean and SEM
across mice, a performance-versus-horizon figure, and a JSON record of all
forecasting settings. Forecasting indicates predictive information and should
not automatically be interpreted as biological causality.

---

# NeMoS Modeling

Behavior-photometry relationships can be modeled using NeMoS.

Current behavioral predictors include:

```text
locomotion
licking
visual cue
solenoid/reward
```

The current modeling framework can use raw 465 fractional fluorescence while estimating a broad session-scale fluorescence component separately.

Conceptually:

```text
raw 465 fluorescence
        |
        +---- broad session-scale component
        |
        +---- residual fluorescence
                    |
                    v
              behavioral GLM
```

The broad component is retained and should not automatically be interpreted as pure photobleaching because genuine biological signals may also occur at long timescales.

---

# Temporal Models

Behavioral predictors can be represented using temporal basis functions.

This allows the model to estimate relationships across time rather than assuming that behavior and photometry are instantaneously related.

Two complementary analyses are being developed.

## Causal / predictive models

Only behavioral information from the present and past is used to predict photometry.

For example:

```text
past licking ----------------> photometry now
-15 s                              0 s
```

These models ask how much information about current photometry is contained in behavior that has already occurred.

## Two-sided temporal-association models

Behavior both before and after the photometry measurement can be examined.

For example:

```text
-7.5 s ------------ 0 ------------ +7.5 s
```

These models characterize the temporal relationship between photometry and behavior.

Because future behavioral information is included, two-sided models should not be interpreted as causal models.

The reverse modeling direction is also of interest:

```text
past photometry -> future behavior
```

---

# Multi-Predictor Models

Behavioral variables are often correlated.

For example:

```text
cue
 |
 v
reward
 |
 v
licking
```

Therefore, a strong single-predictor relationship does not necessarily imply that the predictor uniquely explains the photometry signal.

The modeling framework supports full models such as:

```text
photometry
    ~
locomotion
+ licking
+ cue
+ solenoid
```

and reduced models in which one predictor is removed.

Comparing full and reduced models can help estimate the unique predictive contribution of each behavioral variable.

---

# Cross-Validation and Regularization

Photometry samples close together in time are highly autocorrelated.

Model evaluation therefore uses temporally blocked cross-validation rather than randomly shuffling individual samples.

Temporal exclusion gaps can also be placed around held-out test blocks to reduce leakage from temporal predictor windows.

Multi-predictor models may contain strongly correlated variables, so ridge regularization can be used to stabilize model fitting.

Regularization and cross-validation procedures are currently being evaluated for computational efficiency and robustness across sessions.

All learned preprocessing used by the behavioral GLM is fold-local. The broad
fluorescence component and predictor scaling are fitted on training samples and
then applied to held-out samples. Temporal exclusion gaps are measured from the
original timestamps rather than from compressed array positions.

Temporal lag convention:

```text
negative lag = predictor before the response
zero lag     = simultaneous predictor and response
positive lag = predictor after the response
```

Only non-positive lag windows should be interpreted as causal or predictive.

---

# Processed-Session Provenance

New processed files use schema version `1.0` and record the preprocessing
parameters, package versions, code commit, processing timestamp, event counts,
and IRLS quality-control summaries. Legacy files can still be loaded, but emit a
warning because their exact processing configuration may be unavailable.

---

# Group Analysis

Processed `.npz` files provide the foundation for group-level analyses across mice.

The intended hierarchy is:

```text
samples
   |
   v
trials
   |
   v
session mean
   |
   v
mouse mean
   |
   v
group mean
```

Group analyses should generally preserve the mouse as the biological unit rather than simply pooling every trial from every mouse.

The current group-level tools support:

- mean event-aligned timecourses across mice
- SEM across mice
- cue hit versus miss comparisons
- reward-aligned responses
- lick-bout-aligned responses
- condition comparisons and mouse-level statistical exports

---

# Platforms

Raw-data preprocessing is currently designed primarily for a Windows workstation where photometry data are available through a mapped drive such as:

```text
Z:\Photometry
```

Computational NeMoS analyses are being developed and tested on NIH Biowulf/Linux.

Processed `.npz` sessions provide a portable interface between these environments:

```text
Windows
raw data
   |
   v
preprocessing
   |
   v
processed .npz
   |
   | copy selected sessions
   v
Biowulf
NeMoS / HPC analysis
```

The Windows and Linux directory structures do not need to be identical.

---

# Documentation

More detailed documentation is available within each major directory:

```text
src/README.md
```

describes the reusable analysis modules.

```text
scripts/README.md
```

describes command-line workflows.

```text
notebooks/README.md
```

describes interactive and example notebooks.

---

# Development Status

This repository is under active development.

Current areas of development and validation include:

- photometry quality-control procedures
- alternative handling of poor 405 reference signals
- slow fluorescence decomposition
- expanded real-data validation of temporal behavioral GLMs
- interpretation of causal versus two-sided models
- real-data validation of photometry-to-behavior forecasts
- regularization and cross-validation parameter selection
- computational efficiency on HPC systems

Analysis parameters should therefore be treated as configurable modeling choices rather than fixed biological assumptions.
