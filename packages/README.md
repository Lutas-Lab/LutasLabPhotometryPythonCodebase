# Lutas Lab analysis packages

This repository is the authoritative shared-package monorepo. The packages
were migrated in small, tested steps so existing analysis notebooks remain
reproducible throughout the transition.

The sensor package source and tests live here alongside `lutaslab-core`. The
former standalone repositories remain intact as read-only references while
their examples, notebooks, and Git histories are audited. Make reusable source
changes in this monorepo.

Install the complete workspace from the monorepo root with the recommended
`uv` environment:

```powershell
python -m pip install "uv==0.12.21"
uv python install 3.12
uv sync --frozen --all-packages
uv run lutaslab-check-install
```

This creates `.venv` beside the root `pyproject.toml`; Conda is not required.
Add `--extra dev` only when running repository tests and linters. For Jupyter,
sync with `--extra notebooks`, then run `uv run jupyter lab`.

If an existing environment must be used, install all local packages together
with ordinary pip from the monorepo root:

```powershell
python -m pip install -e packages/lutaslab-core -e . `
  -e packages/fluopulse-analysis -e packages/iflip3-analysis
```

## Package boundaries

- `lutaslab-core`: acquisition-independent timestamps, events, NI-DAQ input,
  clock synchronization, aligned-session containers, and Pynapple adapters.
- `lutaslab-photometry`: interleaved 405/465 preprocessing and conventional
  fiber-photometry workflows.
- `fluopulse-analysis`: Doric FluoPulse waveform and lifetime preprocessing.
- `iflip3-analysis`: iFLiP3 parsing, correction, and lifetime preprocessing.

Plotting, batch execution, temporal GLM utilities, and exact signed-lag design
matrices now live in `lutaslab-core`. Sensor packages retain only
acquisition-specific preprocessing and experiment-specific model assembly.
The NumPy/SciPy ridge GLM in `lutaslab-core` is the default backend for
continuous photometry; NeMoS is an optional compatibility backend.

See [`../docs/architecture.md`](../docs/architecture.md) for the ownership and
support contract and [`../docs/release-hardening.md`](../docs/release-hardening.md)
for the remaining release work.

Curated, output-free example notebooks are retained within each sensor package
under `examples/notebooks`. `lutaslab-validate-notebooks` checks their JSON,
Python syntax, cleared execution state, and absence of machine-specific paths.

## Migration order

1. **Complete:** Establish and test `lutaslab-core` without changing existing callers.
2. **Complete:** Replace duplicated TTL, NI-DAQ, and clock-alignment implementations with
   compatibility imports or adapters.
3. **Complete:** Make each sensor-specific preprocessor produce a common `AlignedSession`.
4. **Complete:** Extract peri-event analysis, plotting, manifests, and batch execution.
5. **Complete:** Consolidate temporal-basis and cross-validated GLM machinery.
6. **Complete:** Reconstruct representative published Figure 5 predictions from
   processed data and compare them numerically with the archived MATLAB outputs.

Processed experimental data remain outside the source repository.
