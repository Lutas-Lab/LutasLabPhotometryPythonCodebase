# Photometry Analysis Source Code

This `src` directory contains the reusable Python code for loading, preprocessing, analyzing, and plotting fiber photometry and behavioral data.

The purpose of keeping these functions in `src` is to separate the underlying analysis code from Jupyter notebooks. Notebooks can then focus on individual experiments and analyses without repeatedly containing hundreds of lines of preprocessing, plotting, or modeling code.

`src` is short for **source code**.

---

# Directory Contents

## `load_data.py`

Functions for locating and loading the raw photometry and behavioral data associated with a recording session.

This module handles the initial conversion of files on disk into Python objects that can be passed into the preprocessing pipeline.

---

## `preprocess.py`

Core fiber photometry and behavioral preprocessing functions.

This module contains the main preprocessing workflow used to construct a processed session.

The preprocessing pipeline includes operations such as:

- photometry signal processing
- 465 nm signal handling
- 405 nm reference handling
- IRLS reference fitting
- dF/F calculation
- locomotion processing
- lick detection
- lick-bout detection
- cue detection
- cue offset detection
- solenoid/reward event detection
- trial classification

The processed data are organized into a Python `session` dictionary.

Example:

```python
session = preprocess_session(...)
```

Important photometry variables may include things such as:

```python
session["photometry_465_ch1"]
session["photometry_405_aligned_ch1"]
session["dff_ch1"]
```

Behavioral variables may include:

```python
session["lick_times"]
session["cue_onset"]
session["cue_offset"]
session["solenoid_onset"]
session["processed_locomotion"]
```

### IRLS correction

The original photometry correction uses robust regression (IRLS) to fit the 405 nm reference signal to the 465 nm signal.

IRLS QC should be inspected because the quality of the 405-to-465 relationship can differ substantially between recordings.

A poor IRLS fit does **not necessarily mean that the recording is unusable**. In particular, the 405 and 465 channels may have different slow bleaching trajectories.

For this reason, raw 465 fluorescence is retained and can also be used directly for downstream GLM analyses.

---

## `save_sessiondata.py`

Functions for saving and loading processed sessions.

Processed sessions are stored as compressed NumPy `.npz` files so that the raw data do not need to be reprocessed every time an analysis is run.

Example filename:

```text
DK21-230704-003-processed.npz
```

### Loading by path

The most portable way to load a session is by specifying the processed file directly:

```python
import src.save_sessiondata

session = src.save_sessiondata.load_session(
    "/path/to/DK21-230704-003-processed.npz"
)
```

This works across Windows, Linux, and NIH Biowulf.

For example, on Biowulf:

```python
session = src.save_sessiondata.load_session(
    "/data/USERNAME/processed_photometry/DK21-230704-003-processed.npz"
)
```

The original Windows photometry path may still be stored inside:

```python
session["photometry_path"]
```

as provenance.

The actual location from which the processed session was loaded is stored separately as:

```python
session["processed_session_path"]
```

This allows processed sessions generated on Windows to be copied to and analyzed on Linux without changing the original provenance information.

---

## `pynapple_utils.py`

Functions for converting processed session data into Pynapple objects.

Pynapple provides convenient time-series and event-based data structures that make it easier to work with signals recorded at different sampling rates.

A processed session can be converted with:

```python
import src.pynapple_utils

data = src.pynapple_utils.session_to_pynapple(
    session
)
```

The resulting dictionary may contain:

```python
data["dff_ch1"]
data["dff_ch2"]
data["locomotion"]
data["licks"]
data["visual_cues"]
data["solenoid"]
data["lick_bouts"]
```

This makes it straightforward to interpolate signals onto common timebases and perform event-aligned analyses.

---

## `plotting.py`

Reusable plotting functions for photometry and behavioral analyses.

The purpose of this module is to avoid repeatedly writing long plotting scripts in notebooks.

Plotting functions can be used for analyses such as:

- event-aligned photometry traces
- peri-event time histograms
- trial averages
- trial heatmaps
- cue-aligned responses
- lick-aligned responses
- lick-bout-aligned responses
- solenoid/reward-aligned responses
- hit-versus-miss trial comparisons
- z-scored trial responses
- IRLS quality-control plots

Keeping these functions in `plotting.py` makes figures more consistent across mice and sessions.

---

# Trial Classification

Cue trials can be classified based on licking behavior.

The current logic distinguishes several trial types.

For each cue:

### Cue hit

The mouse licked during the cue.

```text
cue onset ---------------- cue offset
              lick
```

### Post-cue hit

The mouse licked during the specified post-cue period.

The post-cue period is defined relative to **cue offset**, rather than assuming a fixed cue duration.

For example:

```text
cue onset ------- cue offset |---- 2 s ----|
                                  lick
```

This is important because cue durations may differ between experiments, for example 5-second versus 8-second cues.

Possible classifications include:

```text
cue only
post-cue only
cue + post-cue
miss
```

These classifications can be used to restrict event-aligned photometry analyses to behaviorally relevant trials.

---

# Photometry Normalization and Z-Scoring

Several photometry representations may be useful depending on the analysis.

These include:

```text
raw 465 fluorescence
IRLS-corrected dF/F
trial-baseline z-score
raw-465 trial z-score
slow-component residual fluorescence
```

These representations should not automatically be assumed to be equivalent.

For example, z-scoring the raw 465 signal without 405 correction can be useful as a comparison when evaluating whether reference correction materially changes an event-related response.

---

# Slow Fluorescence Changes

Photometry recordings may contain large slow changes across a session.

These could reflect:

- photobleaching
- optical changes
- movement-related changes
- sensor drift
- genuine biological changes
- combinations of these processes

A fixed high-pass or smoothing operation cannot inherently determine whether a slow fluctuation is biological or artifactual.

Therefore, slow fluorescence should not automatically be labeled "bleaching."

For GLM analyses, the current strategy is to retain the raw 465 signal and estimate a broad session-scale component separately.

Conceptually:

```text
raw 465 fluorescence
        |
        +---- broad session-scale component
        |
        +---- residual fluorescence
                    |
                    +---- behavioral GLMs
```

The broad component remains available for inspection and analysis.

The residual is used to investigate faster behavior-related fluorescence fluctuations.

---

# `nemos_analysis.py`

Functions for modeling relationships between photometry and behavior using NeMoS.

The NeMoS analysis is currently designed to work with raw demultiplexed 465 nm fluorescence rather than requiring IRLS-corrected dF/F.

This is useful for recordings in which the 405-to-465 IRLS correction performs poorly.

---

# Common GLM Timebase

Photometry and locomotion can have different sampling rates.

For example:

```text
photometry   ~50 Hz
locomotion   ~30 Hz
```

The current GLM workflow uses the locomotion timestamps as the common timebase.

The 465 nm photometry signal is interpolated onto this timebase.

This avoids artificially upsampling the lower-frequency locomotion signal.

---

# Raw 465 Fractional Fluorescence

Raw 465 fluorescence is converted to session-level fractional fluorescence:

```text
(F - median(F)) / median(F)
```

This changes the numerical scale without high-pass filtering the signal.

Importantly, this operation does **not** remove slow fluorescence changes.

---

# Slow Session-Scale Component

A broad B-spline basis can be used to estimate gradual changes across the recording.

For example:

```python
behavioral = src.nemos_analysis.prepare_behavioral_response(
    session=session,
    data=data,
    channel=1,
    n_slow_basis=10
)
```

This returns information including:

```python
behavioral["y"]
behavioral["slow_prediction"]
behavioral["residual"]
behavioral["glm_time"]
```

where:

```text
y
    = raw fractional 465 fluorescence

slow_prediction
    = broad session-scale component

residual
    = y - slow_prediction
```

The slow component is **not assumed to be pure photobleaching**.

---

# Behavioral Predictors

Standard behavioral predictors currently include:

```text
locomotion
licking
cue onset
solenoid/reward onset
```

These can be created on the common GLM timebase with:

```python
predictors = src.nemos_analysis.prepare_standard_predictors(
    session=session,
    data=data,
    glm_time=behavioral["glm_time"]
)
```

The predictors are returned as:

```python
predictors["locomotion"]
predictors["licking"]
predictors["cue"]
predictors["solenoid"]
```

Licks, cues, and solenoid events are converted from timestamps into signals on the common GLM timebase.

---

# Temporal Basis Functions

Behavioral predictors are represented using temporal basis functions.

This allows the model to estimate how behavior and photometry are related across time instead of assuming an instantaneous relationship.

The lag convention is:

```text
negative lag                      positive lag

predictor occurs                  predictor occurs
BEFORE response                   AFTER response

<---------------- 0 ---------------->
```

---

# Causal / Predictive Models

A causal or predictive model only uses predictor information from the present or past.

For example:

```python
window=(-15.0, 0.0)
```

means:

```text
-15 s ------------------------- 0 s
past behavior             photometry now
```

This asks:

> How well does behavior that has already occurred predict the current photometry signal?

Examples include:

```text
past locomotion -> current photometry
past licking -> current photometry
past cue information -> current photometry
past solenoid information -> current photometry
```

These models provide temporal prediction but should still not automatically be interpreted as proof of biological causality.

---

# Two-Sided Temporal-Association Models

Two-sided models include predictor information both before and after the response.

For example:

```python
window=(-7.5, 7.5)
```

represents:

```text
-7.5 s ------------ 0 ------------ +7.5 s
```

These models ask:

> At what temporal offset are behavior and photometry most strongly associated?

They are useful for determining whether fluorescence changes tend to precede or follow behavior.

They should **not** be described as causal models because future behavioral information is included.

Default association windows currently include approximately:

```text
locomotion   -7.5 to +7.5 s
licking      -7.5 to +7.5 s
cue          -5 to +10 s
solenoid     -5 to +10 s
```

These windows are analysis parameters and can be changed.

---

# Photometry Predicting Future Behavior

The reverse modeling direction is also scientifically interesting:

```text
past photometry -> future behavior
```

Examples include:

```text
photometry -> future licking
photometry -> future locomotion
```

This is distinct from asking whether behavior predicts photometry.

Different response types may require different GLMs.

For example:

```text
continuous photometry
    -> Gaussian GLM

lick counts
    -> Poisson GLM

binary lick/no-lick state
    -> Bernoulli GLM
```

---

# Single-Predictor Models

Each behavioral variable can first be modeled independently.

For example:

```text
locomotion -> photometry
licking -> photometry
cue -> photometry
solenoid -> photometry
```

These models answer:

> How much photometry variance is associated with this predictor by itself?

However, single-predictor models should be interpreted cautiously because behavioral variables are often correlated.

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

Therefore, licking, cue, and solenoid may explain overlapping portions of the photometry signal.

---

# Full Behavioral Model

A full model includes multiple behavioral predictors simultaneously:

```text
residual 465
    ~
locomotion
+ licking
+ cue
+ solenoid
```

This allows the model to account for correlations among behavioral variables.

---

# Reduced Models and Unique Contributions

To estimate the unique predictive contribution of a variable, the full model can be compared with a model in which that predictor is removed.

For example:

```text
FULL
locomotion + licking + cue + solenoid

versus

NO LICKING
locomotion + cue + solenoid
```

The difference in held-out performance estimates how much unique predictive information licking contributes after accounting for the other variables.

The same comparison can be performed for:

```text
locomotion
licking
cue
solenoid
```

This is generally more informative than comparing single-predictor R² values alone.

---

# Cross-Validation

Photometry samples close together in time are highly autocorrelated.

Therefore, individual time samples should not simply be randomly shuffled into training and testing sets.

The analysis uses blocked cross-validation in which contiguous sections of the recording are held out.

Conceptually:

```text
Fold 1:
TEST | TRAIN | TRAIN | TRAIN | TRAIN

Fold 2:
TRAIN | TEST | TRAIN | TRAIN | TRAIN

Fold 3:
TRAIN | TRAIN | TEST | TRAIN | TRAIN

...
```

---

# Temporal CV Gap

Temporal predictors can include several seconds of history or future information.

Therefore, training samples immediately adjacent to a held-out test block may contain information overlapping the test period.

To reduce this leakage, a temporal exclusion gap can be placed around each held-out block:

```text
TRAIN -------- GAP | TEST TEST TEST | GAP -------- TRAIN
```

The gap should be at least as large as the largest temporal lag represented in the model.

---

# Regularization

Multi-predictor behavioral models can contain strongly correlated variables.

For example:

```text
cue
solenoid
licking
```

may occur at similar times.

An unregularized GLM can therefore produce unstable coefficients and poor held-out predictions.

Ridge (L2) regularization can be used to stabilize the model.

Conceptually:

```text
correlated behavioral predictors
             |
             v
      Ridge-regularized GLM
             |
             v
      more stable prediction
```

Regularization strength is an analysis parameter and should be selected without using held-out test data.

---

# Nested Cross-Validation

When tuning regularization strength, the test data used to evaluate the final model should not also be used to select the regularization parameter.

Nested cross-validation separates these steps.

Conceptually:

```text
OUTER TRAINING DATA
        |
        +---- inner CV
        |       |
        |       +---- test candidate lambda values
        |
        +---- choose lambda
        |
        +---- fit model
        |
        v
OUTER TEST DATA
```

This provides a more defensible estimate of held-out model performance.

Nested CV can be computationally expensive with JAX/NeMoS, so a lighter routine workflow may eventually be used once reasonable model parameters have been established.

---

# Temporal Kernels

Temporal basis coefficients can be reconstructed into a continuous temporal kernel.

The kernel shows how strongly a predictor is associated with the response at different temporal lags.

For example:

```text
negative lag                positive lag

behavior before             behavior after
photometry                  photometry

<------------- 0 ---------------->
```

A kernel can therefore help determine whether behavior tends to precede or follow a fluorescence change.

Kernel sign indicates whether the predictor is associated with an increase or decrease in fluorescence, although the magnitude also depends on the scaling of the predictor.

---

# Model Performance

For continuous photometry models, useful metrics include:

```text
R²
mean squared error (MSE)
```

Cross-validated performance should be emphasized over training performance.

A negative held-out R² means that the model predicted the held-out data worse than simply predicting the mean of that held-out segment.

For multi-predictor analyses, changes in held-out performance between full and reduced models are particularly informative.

---

# Typical Analysis Workflow

A typical session analysis follows:

```text
Raw data
   |
   v
load_data.py
   |
   v
preprocess.py
   |
   v
processed session dictionary
   |
   v
save_sessiondata.py
   |
   v
processed .npz
   |
   v
pynapple_utils.py
   |
   v
common time-series/event representation
   |
   +----------------------+
   |                      |
   v                      v
plotting.py        nemos_analysis.py
   |                      |
event plots          behavioral GLMs
heatmaps             temporal kernels
trial averages       model comparison
```

---

# Example Notebook Setup

A typical notebook can begin with:

```python
import sys
import numpy as np
import matplotlib.pyplot as plt

project_dir = "/path/to/project"

if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

import src.save_sessiondata
import src.pynapple_utils
import src.nemos_analysis
```

Load a processed session:

```python
session = src.save_sessiondata.load_session(
    "/path/to/processed-session.npz"
)
```

Convert it to Pynapple:

```python
data = src.pynapple_utils.session_to_pynapple(
    session
)
```

Prepare the photometry response:

```python
behavioral = src.nemos_analysis.prepare_behavioral_response(
    session=session,
    data=data,
    channel=1,
    n_slow_basis=10
)
```

Prepare behavioral predictors:

```python
predictors = src.nemos_analysis.prepare_standard_predictors(
    session=session,
    data=data,
    glm_time=behavioral["glm_time"]
)
```

---

# Python Environment

The general preprocessing and plotting code uses packages including:

```text
NumPy
SciPy
Matplotlib
Pynapple
```

The NeMoS modeling environment additionally uses:

```text
NeMoS
JAX
```

The current NeMoS environment has been developed using Python 3.12 on NIH Biowulf/Linux.

---

# `__init__.py`

The `__init__.py` file marks `src` as a Python package.

This allows imports such as:

```python
import src.preprocess
import src.plotting
import src.pynapple_utils
import src.nemos_analysis
```

The file itself may be empty.

---

# `__pycache__`

Python may automatically create:

```text
src/__pycache__/
```

This folder contains compiled Python bytecode files ending in:

```text
.pyc
```

For example:

```text
nemos_analysis.cpython-312.pyc
```

These files are automatically generated when Python imports modules.

They are **not source code**.

Do not manually edit them.

They can safely be deleted, and Python will recreate them when necessary.

They generally do not need to be uploaded to a Git repository.

---

# Repository Notes

When manually uploading files to a repository, upload the actual source files such as:

```text
src/
    __init__.py
    load_data.py
    preprocess.py
    plotting.py
    pynapple_utils.py
    save_sessiondata.py
    nemos_analysis.py
    README.md
```

Do not upload:

```text
__pycache__/
*.pyc
```

If Git is later used directly from the command line, these generated files can be excluded with a `.gitignore`.

---

# Development Status

This analysis pipeline is under active development.

In particular, the NeMoS modeling framework is still being evaluated and refined with respect to:

- temporal window selection
- causal versus two-sided temporal models
- number of temporal basis functions
- ridge regularization strength
- blocked cross-validation
- temporal exclusion gaps
- computational efficiency on Biowulf
- slow fluorescence decomposition
- behavior-to-photometry prediction
- photometry-to-behavior prediction
- unique contributions of correlated behavioral predictors
- interpretation of temporal kernels

These settings should therefore be treated as **analysis parameters**, not fixed biological assumptions.

When analysis choices are finalized, this README should be updated to distinguish established defaults from exploratory options.
