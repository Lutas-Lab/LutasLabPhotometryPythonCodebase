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
python scripts/run_preprocess.py --mouse DK21 --date 230704 --run 2
```

The script:

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

## Future Scripts

Additional command-line workflows may be added here, for example:

```text
run_preprocess_batch.py
run_nemos.py
run_group_analysis.py
```

### Batch preprocessing

A future batch preprocessing script can run the same preprocessing pipeline across many mouse/date/run combinations.

For example:

```text
mouse,date,run
DK21,230704,1
DK21,230704,2
DK21,230704,3
DK40,231005,1
```

### NeMoS analysis

Computationally intensive NeMoS analyses can eventually be run as command-line or NIH Biowulf batch jobs rather than requiring an interactive Jupyter session.

## Running Scripts

Run scripts from the repository root.

For example:

```bash
python scripts/run_preprocess.py --mouse DK21 --date 230704 --run 2
```

Scripts import the reusable analysis functions from `src`.

## What Does Not Belong Here

Avoid placing large analysis implementations in this directory.

If a script starts accumulating substantial preprocessing, modeling, or plotting logic, that logic should generally be moved into an appropriate module under `src`.
