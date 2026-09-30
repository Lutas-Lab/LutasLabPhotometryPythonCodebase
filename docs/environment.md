# Reproducible Python environment

The authoritative monorepo environment uses Python 3.12 and is resolved in
`uv.lock`. The lock includes every workspace package and optional extra, with
exact versions, source artifacts, and hashes. Do not hand-edit the lockfile.

## Default analysis environment

Install `uv` once, then create the default environment from the repository
root:

```powershell
python -m pip install "uv==0.12.21"
uv sync --frozen --all-packages --extra dev --extra forecasting
```

The environment is created at `.venv`. Run commands through it directly or
activate it:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest
```

This default environment intentionally does not install NeMoS or JAX.

## Optional NeMoS compatibility environment

Add the optional backend only when an older notebook or a NeMoS-specific
analysis requires it:

```powershell
uv sync --frozen --all-packages --extra dev --extra forecasting --extra nemos
```

The historical extra name remains accepted:

```powershell
uv sync --frozen --all-packages --extra dev --extra modeling
```

Running a later sync without `--extra nemos` or `--extra modeling` removes the
optional backend from the environment.

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
