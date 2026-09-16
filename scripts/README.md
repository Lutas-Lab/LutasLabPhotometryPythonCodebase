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

Run either script with `--help` to see all preprocessing, event, channel,
normalization, time-window, and figure options.

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
