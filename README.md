# Lutas Lab Photometry

Python tools for preprocessing, visualizing, and modeling fiber-photometry and
behavioral data. This monorepo contains the conventional photometry pipeline,
FluoPulse and iFLiP3 analyses, and the shared `lutaslab-core` package.

The conventional pipeline supports 465/405 demultiplexing, robust reference
fitting, locomotion and lick detection, task-event extraction, processed-session
provenance, event-aligned analysis, mouse-level statistics, and temporal GLMs.

## Quickstart

The base packages support Python 3.10–3.12. Python 3.12 is recommended because
it can also run the optional NeMoS/JAX backend. From PowerShell in the
repository root:

```powershell
python -m pip install "uv==0.12.21"
uv python install 3.12
uv sync --frozen --all-packages
uv run lutaslab-check-install
```

Copy the example session manifest and edit it for the cohort:

```powershell
New-Item -ItemType Directory -Force analysis
Copy-Item config\sessions.example.csv analysis\sessions.csv
```

The manifest identifies each session and may include grouping and channel
information:

```csv
mouse,date,run,group,condition,channel
DK21,230704,1,control,naive,1
DK21,230705,2,control,trained,1
```

Run one end-to-end batch workflow—preprocess the listed sessions, then create
cue-aligned PSTHs:

```powershell
uv run lutaslab-run-preprocess-batch `
  --manifest analysis\sessions.csv `
  --data-root "Z:\Photometry" `
  --continue-on-error

uv run lutaslab-run-psth `
  --manifest analysis\sessions.csv `
  --data-root "Z:\Photometry" `
  --output-dir analysis\cue_psth `
  --event-key cue_onset `
  --window -5 20 `
  --baseline -5 0 `
  --normalization zscore
```

Processed `.npz` sessions are saved beside their raw data by default. Figures,
numeric PSTH results, and session/event counts are written to
`analysis\cue_psth`. Existing processed files are skipped unless overwrite is
explicitly requested.

Raw sessions are expected below the data root as
`MOUSE\MOUSE_YYMMDD\MOUSE-YYMMDD-RUN-nidaq.mat`, with a matching
`-running.mat` file. See the
[conventional analysis guide](docs/analysis-guide.md) for the complete input
contract, channel map, processing details, trial classes, null controls,
statistics, and notebook workflow.

## Other ways to run the pipeline

- Jupyter: install the notebook tools with
  `uv sync --frozen --all-packages --extra notebooks`, then run
  `uv run jupyter lab` and open
  [`notebooks/05_batch_workflow.ipynb`](notebooks/05_batch_workflow.ipynb).
- Browser GUI: use `install_gui.bat` and `launch_gui.bat`, or install the `gui`
  extra and launch `streamlit_app.py`. See the [GUI guide](docs/gui.md).
- Individual commands: see the [command-line reference](docs/cli.md) and run
  any `lutaslab-*` command with `--help`.

## Repository structure

```text
LutasLabPhotometryPythonCodebase/
├── packages/
│   ├── lutaslab-core/         shared events, synchronization, GLM, provenance
│   ├── lutaslab-photometry/   conventional package, commands, and tests
│   │   └── src/lutaslab_photometry/
│   │       ├── cli/           installed workflow implementations
│   │       └── data/          packaged default analysis configuration
│   ├── fluopulse-analysis/    FluoPulse package, tests, and examples
│   └── iflip3-analysis/       iFLiP3 package, tests, and examples
├── config/                    session-manifest example and reanalysis paths
├── docs/                      focused user, analysis, and maintainer guides
├── examples/                  standalone API examples
├── notebooks/                 maintained interactive workflow templates
├── analysis/                  local ignored notebooks and results
├── outputs/                   generated ignored outputs
├── streamlit_app.py           browser interface
├── install_gui.bat            double-clickable Windows installer
├── install_gui.ps1            GUI installation implementation
├── launch_gui.bat             double-clickable GUI launcher
├── pyproject.toml             workspace and repository-wide tool configuration
└── uv.lock                    reproducible workspace dependency lock
```

Command implementations live inside
`packages/lutaslab-photometry/src/lutaslab_photometry/cli/` and are exposed
as installed `[project.scripts]` entry points such as
`lutaslab-run-preprocess-batch`, `lutaslab-run-psth`, and
`lutaslab-run-psth-statistics`. They do not require the repository root to be
the current working directory.

## Documentation

| Guide | Contents |
| --- | --- |
| [Environment](docs/environment.md) | Installation, optional dependencies, Conda fallback, and dependency updates |
| [Conventional analysis](docs/analysis-guide.md) | Raw-data layout, preprocessing, manifests, PSTHs, figures, notebooks, and platforms |
| [Command-line workflows](docs/cli.md) | Commands, options, outputs, trial filters, and null controls |
| [GUI](docs/gui.md) | Windows installer, Streamlit launch, and preview safeguards |
| [Modeling and forecasting](docs/modeling.md) | Ridge GLMs, temporal models, validation, regularization, and forecasting |
| [Architecture](docs/architecture.md) | Package ownership, API policy, and compatibility |
| [Statistical policy](docs/statistical-analysis-policy.md) | Biological units, deposited-data tests, and interpretation |
| [Publication bundles](docs/publication-bundles.md) | Compact source-data export format |
| [Figure 5 reanalysis](docs/figure5-reanalysis.md) | Smooth-basis deposited-data reanalysis |
| [Figure 5 bout analysis](docs/figure5-bout-analysis.md) | Lick-bout structure follow-up |
| [Multitastant analysis](docs/multitastant-analysis.md) | Matched delivery-kernel analysis |
| [Legacy repository audit](docs/legacy-repository-audit.md) | FluoPulse and iFLiP3 migration record |
| [Release hardening](docs/release-hardening.md) | Remaining validation and release checklist |

Package-specific documentation is also available in
[`packages/lutaslab-photometry/README.md`](packages/lutaslab-photometry/README.md),
[`notebooks/README.md`](notebooks/README.md), and each workspace package.

## Development

Run the conventional test and lint checks with:

```powershell
uv run python -m pytest
uv run ruff check packages/lutaslab-photometry/src `
  packages/lutaslab-photometry/tests streamlit_app.py
uv run ruff check packages/lutaslab-core/src packages/lutaslab-core/tests
uv run mypy
uv run lutaslab-validate-notebooks
```

GitHub Actions tests the conventional package, shared core, FluoPulse, iFLiP3,
and optional NeMoS compatibility in separate jobs. The codebase is under active
scientific development; analysis parameters are configurable modeling choices,
not fixed biological assumptions.
