# Conventional photometry analysis guide

This guide describes the data layout, preprocessing outputs, manifest-driven
analysis, notebooks, and platform assumptions for the conventional photometry
package. Command options and additional examples are documented in
[`cli.md`](cli.md).

## Raw-data layout

`--data-root` is the directory containing one folder per mouse. A session is
expected to follow this naming convention:

```text
Z:\Photometry\
└── DK21\
    └── DK21_230704\
        ├── DK21-230704-001-nidaq.mat
        └── DK21-230704-001-running.mat
```

The NIDAQ MATLAB file must contain `data`, `timestamps`, and `Fs`. With the
default channel map, rows 1–8 are photoreceiver 1, locomotion TTL,
photoreceiver 2, licking, visual cue, 465-nm TTL, 405-nm TTL, and solenoid TTL.
The running MATLAB file must contain `speed`.

## Processing pipeline

The pipeline demultiplexes 465-nm and 405-nm samples using the LED TTLs,
interpolates 405 nm onto the 465-nm timebase, and uses robust iteratively
reweighted least squares to fit the aligned reference:

```text
dF/F = (465 - fitted 405) / fitted 405
```

Processed sessions retain raw 465, raw 405, aligned 405, fitted 405, and IRLS
dF/F. The corrected signal is one available representation, not an assumption
that 405 nm is the ideal reference for every sensor or recording. Keeping raw
signals permits trial-local normalization, alternative detrending, and
raw-fluorescence models.

The same preprocessing pass extracts locomotion, individual licks, lick bouts,
cue onset/offset, solenoid onset/offset, and cue-related licking classes.
Analysis commands can recompute cue classes from saved timestamps with a
different post-cue window, so this does not require raw-data reprocessing.

New processed files use schema version `1.0` and record preprocessing
parameters, package versions, the code commit, processing time, event counts,
and IRLS quality-control summaries. Legacy files remain readable but may lack
the exact processing configuration.

## Session manifests and batch analysis

Copy `config/sessions.example.csv` to the Git-ignored `analysis/` directory and
edit it for the cohort:

```csv
mouse,date,run,group,condition,channel
DK21,230704,1,control,rewarded,1
DK21,230705,2,control,unrewarded,1
DK40,231005,1,experimental,rewarded,2
```

`mouse`, `date`, and `run` identify a session. `group` and `condition` enable
stratified plots and mouse-level comparisons. `channel` selects photoreceiver 1
or 2 per session; older manifests without it default to channel 1.

Batch preprocessing saves a standardized file such as
`DK21-230704-001-processed.npz` beside the raw session by default. Existing
processed files are skipped unless `--overwrite` is supplied.

```powershell
uv run lutaslab-run-preprocess-batch `
  --manifest analysis/sessions.csv `
  --data-root "Z:\Photometry" `
  --continue-on-error
```

The same manifest drives event-aligned photometry or licking analyses:

```powershell
uv run lutaslab-run-psth `
  --manifest analysis/sessions.csv `
  --data-root "Z:\Photometry" `
  --output-dir analysis/figures/cue `
  --event-key cue_onset `
  --normalization zscore `
  --baseline -5 0
```

Other event arrays include `solenoid_onset`, `lick_bout_onset`, and
`lick_times`. Cue analyses can select `cue_lick`, `post_cue_lick`, `cue_only`,
`post_only`, `cue_and_post`, or `cue_miss` trials. Random-onset and
circular-shift controls are available through `--null-method`.

The averaging hierarchy is deliberately:

```text
events → session mean → mouse mean → group mean ± SEM across mice
```

Thus, a mouse with more sessions or trials does not receive greater weight in
the group result. PSTH outputs include editable figures, numeric arrays, and
event/session counts. Statistical workflows export trial-, session-, mouse-,
and group-level tables plus Prism-ready wide tables. See
[`statistical-analysis-policy.md`](statistical-analysis-policy.md) for the
reporting policy.

Equal-weight mouse summaries are the primary inferential analysis. A
prespecified mixed-effects model may be reported as a sensitivity analysis
when the number of mice and sessions supports its random-effects structure;
trials must remain nested within sessions and mice. More trials improve the
precision of a mouse's estimate but do not increase the number of independent
biological replicates.

## Figures and Illustrator

Figure-producing commands save editable SVG and 300-DPI PNG by default. SVG
text remains editable in Adobe Illustrator. Use `--formats svg png pdf`,
`--font-family`, and `--dpi` to change export settings. Python should determine
the data and statistics; Illustrator is appropriate for final panel layout and
cosmetic editing.

## Notebooks and local work

Maintained notebooks under `notebooks/` are clean templates:

- `01_explore_session.ipynb`: inspect one processed session;
- `02_event_aligned_photometry.ipynb`: event-aligned visualization;
- `03_trial_analysis.ipynb`: trial-level analysis;
- `04_group_analysis.ipynb`: group summaries;
- `05_batch_workflow.ipynb`: manifest creation, preprocessing, and PSTHs.

Copy templates into the Git-ignored `analysis/` directory for mouse-specific
work. Keep reusable code in
`packages/lutaslab-photometry/src/lutaslab_photometry/` and keep maintained notebook
outputs cleared. See [`../notebooks/README.md`](../notebooks/README.md).

## Paper-specific workflows

The Figure 4 dopamine-input builder and delayed PKA transfer fit are exposed as
`lutaslab-build-figure4-dopamine-input` and
`lutaslab-fit-dopamine-pka-transfer`. Figure 5 and multitastant workflows have
dedicated guides:

- [`figure5-reanalysis.md`](figure5-reanalysis.md)
- [`figure5-bout-analysis.md`](figure5-bout-analysis.md)
- [`multitastant-analysis.md`](multitastant-analysis.md)

## Platforms

Raw-data preprocessing is primarily designed for a Windows workstation with a
local or mapped data drive. Processed `.npz` sessions are the portable boundary
for optional Linux or NIH Biowulf work; the Windows and Linux directory layouts
do not need to match.
