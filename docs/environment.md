# Reproducible Python environment

The base workspace packages support Python 3.10–3.12. The recommended complete
environment uses Python 3.12 so it can also install the optional NeMoS/JAX
backend. `uv.lock` contains the compatible resolution, source artifacts, and
hashes; do not hand-edit it.

## Recommended lab-user environment

Conda is not required. Open PowerShell in the repository root, where
`pyproject.toml` and `uv.lock` are located, then run:

```powershell
python -m pip install "uv==0.12.21"
uv python install 3.12
uv sync --frozen --all-packages --extra dev
uv run lutaslab-check-install
```

This installs conventional photometry, FluoPulse, iFLiP3, JupyterLab, and the
developer checks. `uv` creates the project environment at `.venv`. It is a
regular isolated Python environment, but it is owned by this repository rather
than Conda, so it does not appear in `conda info --envs`.

There is no need to activate `.venv`. For later sessions, open PowerShell in
the repository and run commands through `uv`:

```powershell
uv run jupyter lab
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
uv sync --frozen --all-packages --extra dev --extra forecasting
```

Running a later sync without an extra removes that extra from the exact
project environment. To add multiple capabilities, specify them together.

## Optional Conda fallback

Use Conda only when a lab computer or another workflow specifically requires a
Conda-managed environment. Conda must create the environment before this
repository can be installed:

```powershell
conda create -n photometry python=3.12 pip jupyterlab ipykernel -y
conda activate photometry
python -m pip install -e packages/lutaslab-core -e . `
  -e packages/fluopulse-analysis -e packages/iflip3-analysis
lutaslab-check-install
```

If `conda create` stops while gathering or reviewing channels, the repository
has not yet been read. That indicates a Conda channel, proxy, certificate, or
Terms-of-Service problem; using the recommended `uv` workflow avoids that
Conda-specific setup step.

## Optional NeMoS compatibility environment

Add the optional backend only when an older notebook or a NeMoS-specific
analysis requires it:

```powershell
uv sync --frozen --all-packages --extra dev --extra nemos
```

The `nemos` extra requires Python 3.12 and constrains JAX to the tested 0.11
release series. Running a later sync without `--extra nemos` removes the
optional backend from the environment.

## Maintainer and developer environment

The recommended lab setup includes the lightweight developer tools so one
command also provides JupyterLab and package tests across the workspace.
Maintainers can add forecasting support and run the complete non-NeMoS test
suite with:

```powershell
uv sync --frozen --all-packages --extra dev --extra forecasting
uv run python -m pytest
uv run ruff check lutaslab_photometry tests streamlit_app.py
uv run ruff check packages/lutaslab-core/src packages/lutaslab-core/tests
uv run mypy
```

Install and test NeMoS separately so an optional JAX dependency problem cannot
hide failures in the standard analysis stack:

```powershell
uv sync --frozen --all-packages --extra dev --extra nemos
uv run python -m pytest tests/test_nemos_analysis.py
```

GitHub Actions remains the authoritative full validation matrix. It tests the
shared core, conventional photometry, FluoPulse, iFLiP3, and NeMoS compatibility
in separate jobs.

## Updating dependencies

Dependency updates are deliberate source changes:

1. edit a dependency range in the appropriate `pyproject.toml`;
2. run `uv lock --upgrade-package PACKAGE` for a targeted update, or
   `uv lock --upgrade` for a full refresh;
3. rebuild and test both default and NeMoS environments;
4. review and commit `pyproject.toml` and `uv.lock` together.

Use `uv lock --check` in read-only validation to confirm that project metadata
and the lockfile agree.

## Clean-environment verification

On 2026-09-30, two isolated Python 3.12.14 environments were created from the
lockfile on Windows:

- default: NeMoS absent, `h5py 3.16.0`, all 128 package tests passed;
- optional backend: NeMoS 0.2.9 and JAX/JAXLIB 0.11.2 installed, all 128
  package tests passed, and an actual Gaussian ridge fit produced finite
  predictions.

Both environments also passed notebook validation and Ruff. The temporary
verification environments are not part of the repository.
