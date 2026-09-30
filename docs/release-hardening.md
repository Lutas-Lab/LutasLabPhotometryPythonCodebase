# Release-hardening checklist

This checklist tracks the work needed to move from an analysis-ready `0.1`
codebase to a reproducible internal release.

## 1. Ownership and package boundaries

- [x] Designate this monorepo as the authoritative codebase.
- [x] Define shared versus acquisition-specific package ownership.
- [x] Make `lutaslab_core.glm` the default continuous-photometry GLM backend.
- [x] Retain NeMoS as an optional compatibility backend.
- [x] Audit the former standalone repositories for notebooks or uncommitted
  work not represented in this monorepo.
- [x] Curate the retained FluoPulse and iFLiP3 notebooks into the monorepo,
  remove saved outputs, and validate syntax, paths, and package imports.
- [ ] Archive the former repositories only after that audit and explicit
  approval.

## 2. Data configuration and provenance

- [x] Replace experiment-specific path assumptions with versioned config files.
- [x] Add a reusable run record containing input identity, parameters, package
  versions, code revision, start time, and output inventory.
- [x] Adopt the run record in the Figure 5, bout, and multitastant workflows.
- [x] Document which statistical tests are primary and which are exploratory.

## 3. Environments and dependencies

- [x] Keep NeMoS/JAX out of the base installation.
- [x] Preserve the historical `modeling` dependency extra.
- [x] Create a reproducible Python 3.12 `uv.lock` with package hashes.
- [x] Verify a clean base install and a clean optional-NeMoS install.
- [x] Verify FluoPulse `h5py` import and package tests in the current Windows
  analysis environment (`h5py 3.16.0`; 15 tests passed).

## 4. API and quality gates

- [x] Add CI jobs for `lutaslab-core`, conventional photometry,
  `fluopulse-analysis`, and `iflip3-analysis` independently.
- [x] Add a separate optional-NeMoS CI job so its dependency failures cannot
  hide failures in the default analysis path.
- [x] Define public API compatibility expectations and deprecation policy.
- [x] Run all package tests from clean default and optional-NeMoS environments.
- [ ] Tag and document the first internal release.
