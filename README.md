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
python scripts/run_preprocess.py --mouse DK21 --date 230704 --run 2
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

---

# Group Analysis

Processed `.npz` files provide the foundation for future group-level analyses across mice.

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

Future group-level tools will support analyses such as:

- mean event-aligned timecourses across mice
- SEM across mice
- cue hit versus miss comparisons
- reward-aligned responses
- lick-bout-aligned responses
- group-level model summaries

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

Current areas of development include:

- photometry quality-control procedures
- alternative handling of poor 405 reference signals
- slow fluorescence decomposition
- event-aligned analysis
- group-level analysis across mice
- temporal behavioral GLMs
- causal versus two-sided models
- photometry-to-behavior prediction
- regularization
- blocked cross-validation
- temporal exclusion gaps
- computational efficiency on HPC systems

Analysis parameters should therefore be treated as configurable modeling choices rather than fixed biological assumptions.
