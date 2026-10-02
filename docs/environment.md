# Reproducible Python environment

The authoritative monorepo environment uses Python 3.12 and is resolved in
`uv.lock`. The lock includes every workspace package and optional extra, with
exact versions, source artifacts, and hashes. Do not hand-edit the lockfile.

## Lab-user environment

Install `uv` once, then create the default environment from the repository
root:

```powershell
python -m pip install "uv==0.12.21"
uv sync --frozen
```

This installs conventional photometry and `lutaslab-core`. The environment is
created at `.venv`. Confirm the main imports and a small synthetic GLM fit with
the lightweight installation check:

```powershell
uv run python -m scripts.check_install
```

The checked-in `pyproject.toml` enables operating-system certificates for
`uv`. This is useful on institution-managed Windows computers and replaces the
deprecated `--native-tls` command-line option.

Optional features are installed only when required:

```powershell
uv sync --frozen --extra forecasting   # scikit-learn forecasting
uv sync --frozen --all-packages        # conventional, FluoPulse, and iFLiP3
```

Running a later sync without an extra removes that extra from the exact
project environment. To add multiple capabilities, specify them together.

## Optional NeMoS compatibility environment

Add the optional backend only when an older notebook or a NeMoS-specific
analysis requires it:

```powershell
uv sync --frozen --extra nemos
```

The historical extra name remains accepted:

```powershell
uv sync --frozen --extra modeling
```

Running a later sync without `--extra nemos` or `--extra modeling` removes the
optional backend from the environment.

## Maintainer and developer environment

Ordinary users do not need pytest, Ruff, every workspace package, or every
optional analysis backend. Maintainers can install the complete non-NeMoS test
environment and run all package tests with:

```powershell
uv sync --frozen --all-packages --extra dev --extra forecasting
uv run python -m pytest
uv run ruff check src scripts tests packages
```

Install and test NeMoS separately so an optional JAX dependency problem cannot
hide failures in the standard analysis stack:

```powershell
uv sync --frozen --extra dev --extra nemos
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
