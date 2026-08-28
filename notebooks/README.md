# Example and Exploratory Notebooks

This directory contains Jupyter notebooks for interactive exploration, quality control, visualization, and worked examples of the analysis pipeline.

The notebooks provide a high-level interface to the reusable functions implemented in `src`.

The general distinction is:

```text
src/        = reusable analysis implementation
scripts/    = automated workflows
notebooks/  = interactive exploration and examples
```

## `01_explore_session.ipynb`

High-level example for loading and exploring one processed photometry session.

The notebook demonstrates how to:

- select a session using mouse, date, and run
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

The notebook is intended as the primary starting point for examining a new session.

## Session Selection

A notebook should ideally require only high-level session identifiers:

```python
MOUSE = "DK21"
DATE = "230704"
RUN = 2
```

The corresponding processed-session path can then be assembled automatically according to the repository's file-naming convention.

This is preferable to repeatedly entering complete file paths.

## Notebook Philosophy

Notebooks should contain the high-level analysis workflow rather than duplicate the underlying implementation.

For example, a notebook should do something like:

```python
session = load_session(...)
data = session_to_pynapple(session)
plot_something(...)
```

rather than containing hundreds of lines implementing the loading, preprocessing, or plotting algorithms themselves.

If code becomes reusable across notebooks, it should generally be moved into `src`.

## Future Notebooks

Additional notebooks may include:

```text
02_event_aligned_photometry.ipynb
03_trial_analysis.ipynb
04_nemos_glm.ipynb
05_group_analysis.ipynb
```

Possible topics include:

### Event-aligned photometry

- cue-aligned responses
- solenoid-aligned responses
- lick-aligned responses
- lick-bout-aligned responses

### Trial analysis

- cue hits and misses
- post-cue licking
- trial heatmaps
- trial-baseline normalization
- trial-level z-scoring

### NeMoS modeling

- behavior-to-photometry prediction
- temporal-association models
- locomotion kernels
- licking kernels
- cue and reward kernels
- full-versus-reduced models
- cross-validation

### Group analysis

- session averages
- mouse-level averages
- group mean timecourses
- SEM across mice
- condition comparisons

## Data

Raw and processed photometry data should not normally be committed to this directory.

Notebooks should load data from external data storage using configurable paths or session identifiers.

## GitHub

Example notebooks committed to GitHub should preferably:

- use placeholder paths rather than personal user directories
- have clear Markdown explanations
- avoid unnecessary exploratory cells
- avoid storing large outputs
- demonstrate the current recommended analysis workflow
