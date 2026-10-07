# Command-Line Workflows

The installed `lutaslab-*` commands run routine analysis workflows using the
reusable functions in `lutaslab_photometry/`. Their implementations live in
`lutaslab_photometry/cli/`.

The distinction between the two directories is:

```text
lutaslab_photometry/      = how an analysis works
lutaslab_photometry/cli/  = run the analysis
```

Scripts should generally remain relatively short. The underlying analysis logic should live in `lutaslab_photometry` rather than being duplicated here.

## `lutaslab-run-preprocess`

Runs the complete preprocessing pipeline for one photometry session.

Example:

```bash
lutaslab-run-preprocess --mouse DK21 --date 230704 --run 2 --data-root "Z:\Photometry"
```

Use `--output-dir` to save processed data outside the raw session directory.
Run `lutaslab-run-preprocess --help` for configurable detection and
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
lutaslab_photometry/load_data.py
lutaslab_photometry/preprocess.py
lutaslab_photometry/save_sessiondata.py
```

On the current Windows setup, raw photometry data are stored under a mapped photometry drive such as:

```text
Z:\Photometry
```

A processed session is saved using a standardized filename such as:

```text
DK21-230704-002-processed.npz
```

## `lutaslab-run-preprocess-batch`

Uses a CSV session manifest with `mouse,date,run` columns to preprocess many
sessions. By default, every processed `.npz` is saved in the same directory as
its source `.mat` files and existing processed files are skipped. Pass
`--overwrite` to replace existing processed files, or `--continue-on-error` to
continue after a failed session and print a complete summary.

```bash
lutaslab-run-preprocess-batch \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry"
```

## `lutaslab-run-psth`

Uses the same manifest to load processed sessions and generate event-aligned
photometry figures. It saves one figure per mouse, a group mean with SEM across
mice, a numeric `.npz`, and a CSV containing the session/event counts per mouse.

```bash
lutaslab-run-psth \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir analysis/figures/cue \
    --event-key cue_onset
```

The group error band uses mice, not trials, as independent biological units.
Within each mouse, event trials are averaged within a session and session means
are then averaged across that mouse's sessions.

When the manifest includes a `channel` column, each session automatically uses
its listed photoreceiver channel. The default `--channel manifest` supports
mixed-channel cohorts; `--channel 1` or `--channel 2` overrides all rows.

For cue-onset analyses, use `--trial-class cue_miss` (or `cue_lick`,
`post_cue_lick`, `cue_only`, `post_only`, or `cue_and_post`) to filter trials.
`--post-cue-window 4` sets the classification period after cue offset. The
classification is recalculated from timestamps in the processed session, so
changing this value does not require preprocessing again.

Manifest `group` and `condition` columns are also honored automatically. Each
stratum receives its own numeric results and individual/group figures. A
comparison figure for each experimental group overlays condition PSTHs and
shows paired within-mouse difference traces in a second panel. Use
`--no-stratify` to intentionally combine conditions.

Optional null comparisons are available with `--null-method random_onsets` or
`--null-method circular_shift`. Both operate independently within every session,
match the real event count, and are reproducible with `--seed`. The output plots
overlay the observed PSTH with the shuffled mean and 95% null envelope. Use
`--null-exclusion` when null onsets should remain away from real events.

Run either script with `--help` to see all preprocessing, event, channel,
normalization, time-window, and figure options.

## `lutaslab-run-psth-statistics`

Calculates predefined mean, AUC, peak/trough, and latency measurements from
event-aligned responses. It saves trial-, session-, mouse-, and group-level CSV
tables, performs mouse-level paired or independent tests, and writes a
Prism-ready wide table. Optional random-onset or circular-shift controls produce
empirical shuffle tests.

```bash
lutaslab-run-psth-statistics \
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
`--test auto` selects paired t-tests for conditions measured in the same mice
and Welch tests for independent groups; it does not select a test from the
observed p-values. The statistical table includes effect sizes, analytical
mean-difference intervals, and reproducible mouse-level percentile-bootstrap
intervals for both the mean difference and effect size. Configure the latter
with `--bootstrap-resamples` and `--bootstrap-seed`.

Pairwise Holm adjustment treats every pairwise comparison produced by one
command invocation as one family, across metrics, groups, conditions, and both
comparison directions. Shuffle tests are adjusted as a separate family across
all shuffle-test rows from that invocation. The CSV records the family label
and size in `holm_family` and `holm_family_size`. The same `--trial-class` and
`--post-cue-window` options are available here.

Empirical two-sided shuffle p-values use `(k + 1) / (n + 1)`, count ties as
equally extreme, and exclude non-finite null draws from both `k` and `n`.
Consequently they cannot report zero; for 500 valid shuffles the minimum is
`1 / 501`.

## `lutaslab-run-forecasting`

Uses the session manifest to forecast future photometry, locomotion, lick
occurrence, or lick counts from past-only photometry and behavioral features.
It compares a fold-local constant baseline, target-history, cross-modal, and
combined models over one or more forecast horizons.

```bash
lutaslab-run-forecasting \
    --manifest analysis/sessions.csv \
    --data-root "Z:\Photometry" \
    --output-dir analysis/forecasts/photometry \
    --target photometry \
    --horizons 0.5 1 2 5
```

Forecasting uses nested expanding-window evaluation with train-only
standardization and temporal exclusion gaps. Each outer training prefix selects
regularization strength with its own forward inner folds, then the selected
model is refit before the future outer block is scored. Change the candidate
grid with `--alphas`; use `--alpha` only for a strength prespecified independently
of the reported results. Binary lick forecasts report average precision as the
primary metric, AUROC, log loss, prevalence, and average-precision gain over
prevalence. The default estimator backend is scikit-learn and does not require
NeMoS/JAX. Performance figures are saved as editable SVG and PNG by default.

All figure-producing commands accept `--formats svg png pdf`, `--font-family`,
and `--dpi`. SVG output retains editable text for Adobe Illustrator.

## `lutaslab-validate-notebooks`

Checks maintained root, FluoPulse, and iFLiP3 notebooks without executing
experimental analyses. Validation requires nbformat 4 JSON, Python-syntax code
cells, cleared outputs and execution counts, and no personal or former-checkout
paths.

```bash
lutaslab-validate-notebooks
```

This check also runs in CI. Pass explicit notebook files or directories to
validate a smaller set.

## Running commands

Installed commands can run from any directory. Relative manifest, data, and
output paths are resolved from the current working directory.

For example:

```bash
lutaslab-run-preprocess --mouse DK21 --date 230704 --run 2 --data-root "Z:\Photometry"
```

Commands import the reusable analysis functions from `lutaslab_photometry`.

## What Does Not Belong Here

Avoid placing large analysis implementations in `lutaslab_photometry/cli/`.

If a script starts accumulating substantial preprocessing, modeling, or plotting logic, that logic should generally be moved into an appropriate module under `lutaslab_photometry`.
