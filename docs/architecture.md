# Codebase architecture and support policy

This repository is the authoritative source for the Lutas Lab photometry
analysis codebase. It is a monorepo containing one shared package and three
acquisition-specific packages. New reusable code should be added here rather
than copied between the former standalone repositories.

## Package ownership

| Package | Owns | Does not own |
| --- | --- | --- |
| `lutaslab-core` | time-series containers, TTL/event handling, NI-DAQ loading, synchronization, peri-event analysis, plotting helpers, manifests, batch execution, temporal design matrices, ridge GLMs, and cross-validation | sensor-specific signal extraction or experiment-specific model definitions |
| `lutaslab-photometry` | conventional interleaved 405/465 preprocessing and conventional-photometry workflows | FluoPulse or iFLiP3 waveform extraction |
| `fluopulse-analysis` | Doric FluoPulse waveform/lifetime preprocessing and FluoPulse-specific model assembly | generic GLM solvers and temporal bases |
| `iflip3-analysis` | iFLiP3 parsing, correction, and lifetime preprocessing | generic event, plotting, or GLM infrastructure |

The common interchange type is `lutaslab_core.AlignedSession`. Public times
and durations use seconds.

## GLM policy

The NumPy/SciPy ridge implementation in `lutaslab_core.glm` is the supported
default for continuous photometry models. It provides the temporal-basis,
signed-lag, blocked/grouped cross-validation, ridge fitting, prediction, and
kernel-reconstruction operations used by the Figure 5 reanalyses.

NeMoS is optional. `lutaslab_photometry.nemos_analysis` remains available for older notebooks
and for analyses that specifically need NeMoS/JAX functionality, but it is not
required by preprocessing, plotting, Pynapple integration, or the supported
continuous-photometry GLM workflow. New analyses should use
`lutaslab_core.glm` unless a documented NeMoS-only capability is needed.

All workspace packages support Python 3.12. The single `nemos` extra constrains
JAX to the tested 0.11 release series. Base preprocessing and analysis do not
require NeMoS or JAX.

## Repository lifecycle

The copies under `packages/fluopulse-analysis` and `packages/iflip3-analysis`
are the maintained package sources. The former standalone repositories should
remain read-only references until their notebooks and uncommitted work have
been audited. Archiving or deleting them is a separate, explicit decision; it
is not part of the package migration itself.

## Stability levels

- Shared dataclasses, event utilities, synchronization, peri-event functions,
  and GLM primitives exported by `lutaslab_core` are the intended public API.
- Acquisition-specific preprocessing APIs are supported within their packages.
- Modules under `packages/lutaslab-photometry/src/lutaslab_photometry/cli/` are
  reproducible workflows exposed as
  console commands and may provide additional
  experiment-specific options.
- Notebook code and `lutaslab_photometry.nemos_analysis` are compatibility interfaces and are
  not the preferred foundation for new reusable code.

The current release series is `0.1.x`: scientifically usable and tested, but
still open to small API changes before a stable `1.0` release.

## Compatibility and deprecation

During `0.1.x`, public APIs receive best-effort compatibility. A planned
removal should first be documented and, when practical, emit a
`DeprecationWarning` for at least one minor release. Breaking changes must be
listed in the release notes. Beginning with `1.0`, package versions will follow
semantic versioning: incompatible public-API changes require a major release.

Compatibility wrappers may live in acquisition packages or the root
`lutaslab_photometry` package, but new implementations belong in the owning package identified
above. Compatibility code should be thin and separately tested.
