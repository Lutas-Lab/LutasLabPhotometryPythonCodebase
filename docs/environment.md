# Reproducible Python environment

All workspace packages use Python 3.12. A single supported interpreter keeps
the locked analysis environment, GUI, and CI in sync. `uv.lock` contains the
compatible resolution, source artifacts, and
hashes; do not hand-edit it.

## Recommended lab-user environment

Conda is not required. Open PowerShell in the repository root, where
`pyproject.toml` and `uv.lock` are located, then run:

```powershell
python -m pip install "uv==0.12.21"
uv python install 3.12
uv sync --frozen --all-packages
uv run lutaslab-check-install
```

This installs conventional photometry, FluoPulse, and iFLiP3. `uv` creates the
project environment at `.venv`. It is a
regular isolated Python environment, but it is owned by this repository rather
than Conda, so it does not appear in `conda info --envs`.

There is no need to activate `.venv`. For later sessions, open PowerShell in
the repository and run commands through `uv`:

```powershell
uv run lutaslab-run-preprocess-batch --help
```

`uv run` reuses the existing environment and checks that it remains consistent
with the project. `uv sync` updates the environment rather than creating a new
Python installation on every run. Do not copy `.venv` to another computer;
recreate it there from `uv.lock`.

The checked-in `pyproject.toml` enables operating-system certificates for
`uv`. This is useful on institution-managed Windows computers and replaces the
deprecated `--native-tls` command-line option.

Optional features are installed only when required:

```powershell
uv sync --frozen --all-packages --extra forecasting
```

Running a later sync without an extra removes that extra from the exact
project environment. To add multiple capabilities, specify them together.

## Notebook environment

JupyterLab and its kernel support are kept in the explicitly named `notebooks`
extra rather than the developer-tool extra:

```powershell
uv sync --frozen --all-packages --extra notebooks
uv run jupyter lab
```

## Optional Conda fallback

Use Conda only when a lab computer or another workflow specifically requires a
Conda-managed environment. Conda must create the environment before this
repository can be installed:

```powershell
conda create -n photometry python=3.12 pip jupyterlab ipykernel -y
conda activate photometry
python -m pip install -e packages/lutaslab-core -e packages/lutaslab-photometry `
  -e packages/fluopulse-analysis -e packages/iflip3-analysis
lutaslab-check-install
```

If `conda create` stops while gathering or reviewing channels, the repository
has not yet been read. That indicates a Conda channel, proxy, certificate, or
Terms-of-Service problem; using the recommended `uv` workflow avoids that
Conda-specific setup step.

## Maintainer and developer environment

The `dev` extra is for maintainers and contributors; normal analysis does not
require it. Maintainers can add forecasting support and run the complete test
suite with:

```powershell
uv sync --frozen --all-packages --extra dev --extra forecasting
uv run python -m pytest
uv run ruff check packages/lutaslab-photometry/src `
  packages/lutaslab-photometry/tests streamlit_app.py
uv run ruff check packages/lutaslab-core/src packages/lutaslab-core/tests
uv run mypy
```

GitHub Actions remains the authoritative full validation matrix. It tests the
shared core, conventional photometry, FluoPulse, and iFLiP3 in separate jobs.

## Updating dependencies

Dependency updates are deliberate source changes:

1. edit a dependency range in the appropriate `pyproject.toml`;
2. run `uv lock --upgrade-package PACKAGE` for a targeted update, or
   `uv lock --upgrade` for a full refresh;
3. rebuild and test the default environment;
4. review and commit `pyproject.toml` and `uv.lock` together.

Use `uv lock --check` in read-only validation to confirm that project metadata
and the lockfile agree.

## Clean-environment verification

On 2026-09-30, an isolated Python 3.12.14 environment was created from the
lockfile on Windows. The package tests, notebook validation, and Ruff checks
passed. The temporary verification environment is not part of the repository.
