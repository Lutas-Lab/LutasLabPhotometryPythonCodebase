# Example and Exploratory Notebooks

This directory contains Jupyter notebooks for interactive exploration, quality control, visualization, trial-level analysis, and group-level analysis of fiber photometry data.

The notebooks provide a high-level interface to the reusable functions implemented in `src`.

The general distinction is:

```text
src/        = reusable analysis implementation
scripts/    = automated workflows
notebooks/  = clean interactive examples and templates
analysis/   = local mouse/session-specific working notebooks
```

The notebooks committed to GitHub should remain clean, reusable examples. Mouse- or experiment-specific working copies should generally be placed in the local `analysis/` directory, which is excluded from Git.

---

## `01_explore_session.ipynb`

High-level example for loading and exploring one processed photometry session.

This notebook is intended as the primary starting point for examining a new recording.

The notebook demonstrates how to:

- select a session using mouse, date, and run
- automatically construct the processed-session path
- load a processed `.npz` session
- convert the session to Pynapple objects
- inspect recording duration and sampling rates
- inspect raw 465 and 405 photometry
- inspect processed dF/F
- inspect locomotion
- inspect lick events
- inspect visual cue events
- inspect solenoid/reward events
- compare raw 465 with processed dF/F
- optionally prepare raw 465 for NeMoS modeling

### Session Selection

A notebook should ideally require only high-level session identifiers:

```python
MOUSE = "DK21"
DATE = "230704"
RUN = 2
```

The corresponding processed-session path is then assembled automatically according to the repository's file-naming convention.

This is preferable to repeatedly entering complete file paths.

---

## `02_event_aligned_photometry.ipynb`

Event-aligned analysis of photometry and behavioral events within one session.

This notebook demonstrates how to align photometry to:

- visual cue onset
- solenoid/reward onset
- individual licks
- lick-bout onset

For each event type, the notebook can generate:

- event-aligned trial matrices
- mean event-aligned timecourses
- SEM across events within the session
- trial-by-trial heatmaps

The notebook also compares:

```text
IRLS-corrected dF/F
versus
raw 465 fractional fluorescence
```

This comparison can be useful when evaluating whether 405-based reference correction materially changes the apparent event-related photometry response.

Event-aligned averages describe temporal relationships around events but should not by themselves be interpreted as evidence of causality.

---

## `03_trial_analysis.ipynb`

Trial-level analysis of cue-related photometry responses.

This notebook uses the behavioral trial classifications generated during preprocessing rather than recalculating them in the notebook.

Analyses include:

- cue-lick versus cue-miss trials
- cue-only trials
- post-cue-only trials
- cue + post-cue licking trials
- miss trials
- trial-by-trial heatmaps
- trial-local baseline normalization
- trial-local z-scoring
- comparison of IRLS-corrected dF/F with raw 465 fluorescence
- raw 465 trial-local z-scoring

Cue-related licking classifications use the actual cue onset and cue offset for each trial.

Post-cue licking is defined relative to cue offset.

The notebook preserves trial-level variability within a session but does not treat individual trials as independent animals for group-level statistical analysis.

---

## `04_group_analysis.ipynb`

Group-level analysis across multiple sessions and mice.

This notebook is designed to preserve the hierarchy of the experimental data:

```text
individual trials
      |
      v
session mean
      |
      v
mouse mean
      |
      v
group mean ± SEM
```

This is important because mice may contribute different numbers of trials or sessions.

Individual trials are therefore **not simply pooled across all mice**.

If a mouse contributes multiple sessions, those sessions are first summarized within that mouse. Each mouse then contributes one mouse-level timecourse to the group analysis.

### Session List

Sessions are specified near the top of the notebook using a simple list:

```python
SESSIONS = [
    {"mouse": "DK21", "date": "230704", "run": 1},
    {"mouse": "DK21", "date": "230704", "run": 2},
    {"mouse": "DK40", "date": "231005", "run": 1},
]
```

The corresponding processed-session paths are constructed automatically.

### Current Group Analyses

The notebook currently demonstrates:

- loading multiple processed sessions
- cue-aligned trial extraction
- cue-lick trial summaries
- cue-miss trial summaries
- trial-local z-scoring
- session-level averaging
- averaging multiple sessions within mouse
- individual mouse timecourses
- group mean timecourses
- SEM calculated across mice
- retention of mouse-level traces for later statistical analysis

For example:

```text
DK21 run 1 ─┐
             ├─→ DK21 mean ─┐
DK21 run 2 ─┘               │
                             ├─→ group mean ± SEM
DK40 run 1 ───→ DK40 mean ──┤
                             │
DK41 run 1 ───→ DK41 mean ──┘
```

The mouse, rather than the individual trial, is therefore treated as the biological unit for the final group summary.

The group-analysis framework can later be extended to:

- solenoid/reward-aligned responses
- lick-bout-aligned responses
- additional trial classifications
- raw 465 analyses
- group condition comparisons
- formal mouse-level statistical testing

---

# Notebook Philosophy

Notebooks should contain the high-level analysis workflow rather than duplicate the underlying implementation.

For example, a notebook should ideally do something like:

```python
session = load_session(...)
data = session_to_pynapple(session)
plot_something(...)
```

rather than containing hundreds of lines implementing the loading, preprocessing, or plotting algorithms themselves.

If code becomes reusable across notebooks, it should generally be moved into `src`.

This keeps the analysis logic centralized and reduces the chance that different notebooks accidentally use different versions of the same analysis.

---

# Working Analysis Notebooks

The notebooks in this directory are intended to remain clean, reusable templates.

Do not use these template notebooks directly for long-term mouse-specific analyses.

Instead, create a local directory at the repository root:

```text
analysis/
```

and copy the desired notebook template there.

For example:

```text
notebooks/02_event_aligned_photometry.ipynb
                    |
                    v
analysis/DK21_230704_002_events.ipynb
```

or:

```text
notebooks/03_trial_analysis.ipynb
                    |
                    v
analysis/DK21_230704_002_trials.ipynb
```

The repository should therefore look something like:

```text
photometry-analysis/
│
├── src/
├── scripts/
│
├── notebooks/
│   ├── 01_explore_session.ipynb
│   ├── 02_event_aligned_photometry.ipynb
│   ├── 03_trial_analysis.ipynb
│   └── 04_group_analysis.ipynb
│
└── analysis/
    ├── DK21_230704_002.ipynb
    ├── DK21_230704_002_events.ipynb
    └── group_experiment_1.ipynb
```

The `analysis/` directory is intentionally excluded from Git using:

```gitignore
analysis/
```

This allows working notebooks to contain:

- mouse-specific settings
- experiment-specific session lists
- generated figures
- notebook outputs
- exploratory code
- analysis notes
- temporary analyses

without cluttering the GitHub repository.

Reusable improvements discovered while working in `analysis/` should be moved into the appropriate `src/` module or incorporated into one of the clean notebook templates.

---

# Data

Raw and processed photometry data should not normally be committed to this directory or to the Git repository.

Notebooks should load data from external storage using configurable paths and high-level session identifiers.

For example:

```python
PHOTOMETRY_ROOT = Path(
    r"Z:\Photometry"
)

MOUSE = "DK21"
DATE = "230704"
RUN = 2
```

Processed-session filenames can then be assembled automatically.

---

# GitHub

Example notebooks committed to GitHub should preferably:

- use placeholder paths rather than personal user directories
- contain clear Markdown explanations
- avoid unnecessary exploratory cells
- avoid storing large outputs
- demonstrate the current recommended analysis workflow
- keep mouse-specific analyses in the Git-ignored `analysis/` directory

---

# Planned Notebook

The next planned notebook is:

```text
05_nemos_glm.ipynb
```

This notebook will provide the high-level workflow for behavioral modeling using NeMoS, including:

- raw 465 response preparation
- broad session-scale fluorescence modeling
- locomotion predictors
- licking predictors
- cue predictors
- solenoid/reward predictors
- causal/predictive temporal models
- two-sided temporal-association models
- temporal kernels
- blocked cross-validation
- regularization
- full-versus-reduced model comparisons
