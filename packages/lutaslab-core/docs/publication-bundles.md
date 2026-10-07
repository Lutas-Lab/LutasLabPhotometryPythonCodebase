# Slim publication bundles

Publication bundles are the small, portable output of an analysis. They contain
the numerical values needed to reproduce paper figures without carrying all raw
acquisition channels or intermediate processing arrays.

They do **not** replace either of the other two data levels:

1. Raw acquisition files support complete reprocessing.
2. Processed session files support exploratory analysis and method development.
3. Publication bundles support exact figure and statistical-result reconstruction.

## Format

`lutaslab_core.publication.PublicationBundle` writes a directory containing:

- `tables/*.csv.gz`: UTF-8, gzip-compressed CSV source tables;
- `metadata/*.json`: model specifications, analysis parameters, and provenance;
- `figure_manifest.json`: versioned file schemas and panel-to-source mappings;
- `checksums.sha256`: integrity checks for every deposited file; and
- `README.md`: a human-readable inventory and reconstruction commands.

Compressed CSV is the default because it is open, durable, and readable from
Python, R, MATLAB, spreadsheet software, or a command line. A shared time column
should appear once in a wide trace table rather than once per mouse and signal.

## Minimum information to retain

Keep the lowest-level observations used for inference or plotting:

- stable anonymous subject, session, and trial identifiers;
- experimental group, condition, date/run identifiers when appropriate;
- exclusions and their reasons;
- per-subject traces, not only population mean and SEM;
- per-trial values when trials are the analysis unit or the figure requires them;
- every point underlying scatter, bar, violin, or paired-line plots;
- GLM kernels or coefficients for each subject and model;
- held-out predictions when a panel plots observed versus predicted traces;
- model metrics for each cross-validation fold or subject;
- event times when regrouping events is necessary for figure reconstruction; and
- units, normalization, windows, basis parameters, regularization, folds, and seeds.

The inferential unit must remain recoverable. Do not replace animal-level values
with a population summary, even when the final panel displays only mean and SEM.

## Data to omit

Do not include information that can be deterministically regenerated or is not
used by a deposited figure:

- full GLM design matrices;
- repeated copies of a common time axis;
- unused photoreceiver or NI-DAQ channels;
- intermediate filtered traces;
- redundant raw, z-scored, and dF/F forms when the transformation is documented;
- bootstrap samples when intervals can be regenerated from a recorded seed; or
- local absolute paths and personally identifying metadata.

## Example

```python
from lutaslab_core import PublicationBundle

bundle = PublicationBundle(
    "paper_figure_data",
    title="Paper title: source data",
    creators=["Lutas Lab"],
    code_url="https://github.com/Lutas-Lab/LutasLabPhotometryPythonCodebase",
    code_version="git-commit-or-release-tag",
)

trace_path = bundle.add_table(
    "figure2_mouse_traces",
    trace_rows,
    columns=["time_s", "mouse_1", "mouse_2"],
    description="Cue-aligned CeA dopamine, one column per mouse.",
    units={"time_s": "s", "mouse_1": "z-score", "mouse_2": "z-score"},
    primary_key=["time_s"],
    panel_ids=["2A"],
)
spec_path = bundle.add_json(
    "figure2_analysis",
    analysis_parameters,
    description="Alignment, normalization, uncertainty, and plotting parameters.",
    panel_ids=["2A"],
)
bundle.add_figure(
    "2A",
    title="Consumption-evoked CeA dopamine",
    sources=[trace_path.relative_to(bundle.root), spec_path.relative_to(bundle.root)],
    command="python reproduce_figures.py --panel 2A",
)
bundle.finalize()
```

Validate a completed deposit before release:

```powershell
python -m lutaslab_core.publication paper_figure_data --validate
```

Validation checks the schema, paths, table columns and row counts, panel source
references, and all SHA-256 hashes. The reconstruction script should also be run
in a clean environment as the final release check.
