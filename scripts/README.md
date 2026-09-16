# Analysis Scripts

This directory contains command-line programs that run routine analysis workflows using the reusable functions in `src/`.

The distinction between the two directories is:

```text
src/      = how an analysis works
scripts/  = run the analysis
```

Scripts should generally remain relatively short. The underlying analysis logic should live in `src` rather than being duplicated here.

## `run_preprocess.py`

Runs the complete preprocessing pipeline for one photometry session.

Example:

```bash
python scripts/run_preprocess.py --mouse DK21 --date 230704 --run 2 --data-root "Z:\Photometry"
```

Use `--output-dir` to save processed data outside the raw session directory.
Run `python scripts/run_preprocess.py --help` for configurable detection and
classification parameters. The script:

```text
mouse/date/run
      |
      v
locate raw data
      |
      v
load session
      |
      v
preprocess session
      |
      v
save processed .npz
```

The actual preprocessing functions are implemented in:

```text
src/load_data.py
src/preprocess.py
src/save_sessiondata.py
```

On the current Windows setup, raw photometry data are stored under a mapped photometry drive such as:

```text
Z:\Photometry
```

A processed session is saved using a standardized filename such as:

```text
DK21-230704-002-processed.npz
```

## `run_preprocess_batch.py`

Uses a CSV session manifest with `mouse,date,run` columns to preprocess many
sessions. By default, every processed `.npz` is saved in the same directory as
its source `.mat` files and existing processed files are skipped. Pass
`--overwrite` to replace existing processed files, or `--continue-on-error` to
continue after a failed session and print a complete summary.

```bash
python scripts/run_preprocess_batch.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry"
```

## `run_psth.py`

Uses the same manifest to load processed sessions and generate event-aligned
photometry figures. It saves one figure per mouse, a group mean with SEM across
mice, a numeric `.npz`, and a CSV containing the session/event counts per mouse.

```bash
python scripts/run_psth.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir analysis/figures/cue \
    --event-key cue_onset
```

The group error band uses mice, not trials, as independent biological units.
Within each mouse, event trials are averaged within a session and session means
are then averaged across that mouse's sessions.

Optional null comparisons are available with `--null-method random_onsets` or
`--null-method circular_shift`. Both operate independently within every session,
match the real event count, and are reproducible with `--seed`. The output plots
overlay the observed PSTH with the shuffled mean and 95% null envelope. Use
`--null-exclusion` when null onsets should remain away from real events.

Run either script with `--help` to see all preprocessing, event, channel,
normalization, time-window, and figure options.

## `run_psth_statistics.py`

Calculates predefined mean, AUC, peak/trough, and latency measurements from
event-aligned responses. It saves trial-, session-, mouse-, and group-level CSV
tables, performs mouse-level paired or independent tests, and writes a
Prism-ready wide table. Optional random-onset or circular-shift controls produce
empirical shuffle tests.

```bash
python scripts/run_psth_statistics.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir analysis/statistics/cue \
    --event-key cue_onset \
    --response-window 0 2 \
    --metrics mean auc peak peak_latency \
    --null-method random_onsets \
    --n-shuffles 1000
```

Optional `group` and `condition` columns in the manifest determine the
comparisons. Mice are the independent units; trials and sessions are retained
in the exported tables but are not counted as separate animals. One figure per
metric is saved as editable SVG and PNG by default, including mouse points,
paired lines, 95% confidence intervals, and adjusted statistical annotations.

## `run_forecasting.py`

Uses the session manifest to forecast future photometry, locomotion, lick
occurrence, or lick counts from past-only photometry and behavioral features.
It compares target-history, cross-modal, and combined models over one or more
forecast horizons.

```bash
python scripts/run_forecasting.py \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir analysis/forecasts/photometry \
    --target photometry \
    --horizons 0.5 1 2 5
```

Forecasting uses expanding-window evaluation with train-only standardization and
a temporal exclusion gap. The default estimator backend is scikit-learn and
does not require NeMoS/JAX. Its performance figure is saved as editable SVG and
PNG by default.

All figure-producing commands accept `--formats svg png pdf`, `--font-family`,
and `--dpi`. SVG output retains editable text for Adobe Illustrator.

## Future Scripts

Additional command-line workflows may be added here, for example:

```text
run_nemos.py
run_nemos_batch.py
```

### NeMoS analysis

Computationally intensive NeMoS analyses can eventually be run as command-line or NIH Biowulf batch jobs rather than requiring an interactive Jupyter session.

## Running Scripts

Run scripts from the repository root.

For example:

```bash
python scripts/run_preprocess.py --mouse DK21 --date 230704 --run 2 --data-root "Z:\Photometry"
```

Scripts import the reusable analysis functions from `src`.

## What Does Not Belong Here

Avoid placing large analysis implementations in this directory.

If a script starts accumulating substantial preprocessing, modeling, or plotting logic, that logic should generally be moved into an appropriate module under `src`.
