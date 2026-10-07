# FluoPulse analysis

Python tools for Doric FluoPulse waveform-sampling lifetime photometry. The
primary analysis uses Doric's saved `Tau01` output. The package also exposes
amplitude and fit-quality traces, raw fluorescence waveforms, the saved IRF,
digital inputs, independent exploratory waveform fits, and alignment to the
laboratory NI-DAQ behavior files.

> This repository is private and pre-release. See `NOTICE.md` before sharing.

## Analysis policy

- Use the vendor-provided `Tau01` as the primary lifetime measurement.
- Retain amplitude, chi-square, and R-square as quality-control channels.
- Treat independent waveform fitting and moment estimates as exploratory until
  they are validated with lifetime standards and compared against Doric output.
- Keep the IRF acquired with the same laser power and detector gain as the
  animal recording. The FluoPulse manual requires recalibration when either
  setting changes.
- Use Doric DIO02 (`CAMpulse`) to align the Doric and NI-DAQ clocks.
- Use Doric DIO04 (`lick_event`) as the primary lick timestamps.
- Use NI-DAQ behavior rows for Ensure/solenoid, visual cue, and other behavior.
- Compare Doric DIO05 (`ensure_delivery`) with NI-DAQ Ensure as a wiring/QC
  check when Doric Ensure pulses are available.

## Validated Doric layout

The local example files are HDF5 containers with the following datasets:

```text
/DataAcquisition/FluoPulse/Signals/Series0001/
  Calculation01/{Time,Tau01,Amplitude,Chisquare,Rsquare}
  DigitalIO/{Time,DIO02,DIO04,DIO05,DIO09}
  Fluorescence01/{Time,Values}
  IRF01/{Time,Raw,Values}
```

The calculation series is approximately 10 Hz, raw waveforms contain 4,096
points per lifetime sample, and digital inputs are sampled at 10 kHz. The
reader discovers semantic DIO names from the configuration instead of assuming
that every Doric file uses identical port numbers.

## Installation

Install FluoPulse as part of the authoritative monorepo workspace. Open
PowerShell in the monorepo root, where `uv.lock` is located, then run:

```powershell
python -m pip install "uv==0.12.21"
uv python install 3.12
uv sync --frozen --all-packages --extra dev
uv run python -m pytest packages/fluopulse-analysis/tests
```

Conda is not required. `uv` creates a repository-local `.venv` containing the
locked workspace dependencies. Launch Jupyter from the monorepo root with:

```powershell
uv run jupyter lab
```

## Inspect a Doric recording

Open `packages/fluopulse-analysis/examples/notebooks/01_load_and_qc.ipynb` for
vendor-lifetime QC and
`packages/fluopulse-analysis/examples/notebooks/02_waveform_reanalysis.ipynb`
for independent IRF-convolved fits. The latter is not intended to reproduce
Doric's proprietary deconvolution exactly. Fit-window selection is important:
on the local example, a 0-40 ns window produced values near `Tau01`, whereas
shorter windows were substantially biased. This observation is a
method-development result, not a completed validation.

## Align with NI-DAQ

The standard NI-DAQ hierarchy is:

```text
Z:\Photometry\<mouse>\<mouse>_<date>\<mouse>-<date>-<run>-nidaq.mat
Z:\Photometry\<mouse>\<mouse>_<date>\<mouse>-<date>-<run>-running.mat
```

A Doric filename such as `260923_SC81_run1_0000.doric` supplies mouse, date,
and run automatically. Use `nidaq_paths_for_doric` to infer companion paths or
pass explicit paths to `process_aligned_session`, as shown below. The aligned
event-analysis workflow is demonstrated in
`packages/fluopulse-analysis/examples/notebooks/03_aligned_event_analysis.ipynb`.

Running timestamps are taken directly from the running MAT file when present.
Otherwise, speed is attached to CAM pulses when the lengths match; if they do
not, a uniform running time base is inferred from the NI-DAQ recording duration
and should be treated as a QC assumption.

Because a uniform CAM pulse train does not encode pulse identity, the default
alignment assumes that the first saved Doric and NI-DAQ CAM pulses correspond.
If one system missed known leading pulses, pass `doric_sync_start_index` or
`nidaq_sync_start_index` to the alignment API.

## Python and Pynapple

```python
from fluopulse_analysis import (
    nidaq_paths_for_doric,
    process_aligned_session,
    read_doric,
)

doric_file = r"C:\path\to\260923_SC81_run1_0000.doric"
recording = read_doric(doric_file)

# Raw data are lazy: this loads only three 4,096-point waveforms.
batch = recording.load_waveforms([0, 100, 1000])

paths = nidaq_paths_for_doric(doric_file)
session = process_aligned_session(
    doric_file,
    paths.nidaq,
    running_path=paths.running,
)
data = session.to_pynapple()
```

Pynapple objects include vendor tau, amplitude, fit R-square, Doric licks,
NI-DAQ licks, Ensure delivery, visual cue, and optional running speed. Standard
Pynapple `compute_perievent` calls can then align tau to Ensure, cues, licks, or
other behavioral events.

Doric calculation timestamps contain small acquisition-timing variations.
`to_pynapple()` therefore interpolates lifetime, amplitude, fit quality, and
running speed onto uniform grids, as required by Pynapple's continuous-data
peri-event functions. The original unmodified timestamps and values remain in
the `AlignedSession` and `FluoPulseRecording` objects.

## Jupyter notebooks

- `examples/notebooks/01_load_and_qc.ipynb`: vendor tau and acquisition QC.
- `examples/notebooks/02_waveform_reanalysis.ipynb`: lazy raw-waveform loading
  and explicitly exploratory independent single- or double-exponential fits.
- `examples/notebooks/03_aligned_event_analysis.ipynb`: CAM alignment and
  Pynapple peri-event analysis.
- `examples/notebooks/04_batch_session_qc.ipynb`: scan all runs in one session,
  tabulate alignment/coverage diagnostics, and flag incomplete NI-DAQ files.
- `examples/notebooks/05_lick_bout_analysis.ipynb`: define lick bouts from a
  fixed inter-lick interval, select the major bout near a target time, and plot
  lifetime aligned to bout onset.
- `examples/notebooks/06_batch_mice_lick_bouts.ipynb`: read a mouse/date Excel
  sheet, process all matching Doric and NI-DAQ runs, select bouts with or
  without nearby Ensure delivery, and export aligned lifetime results.
- `examples/notebooks/07_batch_doric_only_lick_bouts.ipynb`: process descriptive
  FluoPulse filenames directly from `.doric` files when lifetime, licking, and
  Ensure TTLs are all embedded and NI-DAQ alignment is unnecessary. The slow
  pass caches every aligned bout in `all_bout_cache.npz`; later Ensure-status,
  mouse, date, recording-text, lick-count, and preceding lick-free interval
  filters reuse that cache. Its default clean-onset view requires 10 seconds
  without a detected lick before the bout. Cached lick histograms can be shown
  as a trial raster, mean lick-rate trace, both, or neither alongside lifetime.
- `examples/notebooks/08_first_ensure_transition.ipynb`: align each recording to
  its first Ensure TTL over a 60-second pre/post window, cache lifetime and
  behavior, and visualize licking plus subsequent scheduled Ensure deliveries.
- `examples/notebooks/09_full_ensure_session.ipynb`: cache from 60 seconds before
  the first Ensure through the end of every recording, retain every later
  delivery, and generate arbitrary shorter views without rereading `.doric` files.
- `examples/notebooks/10_adaptive_ensure_glm.ipynb`: use complete baseline,
  post-Ensure, and reward recordings to estimate lick and delivery-dependent
  Ensure response kernels with a cross-validated convolutional GLM.
- `examples/notebooks/11_mechanistic_pka_cascade.ipynb`: fit a minimal
  consumption-to-signaling-to-PKA-to-sensor cascade with recoverable adaptation,
  using the full-recording cache and explicit identifiability diagnostics.

Run them in order. The first two require only a Doric file; the third also
requires the corresponding files under `Z:\Photometry`.

The alignment notebooks also support folders copied into `Documents`. Always
inspect `nidaq_coverage_fraction`: a low value means that the clocks can align
over their shared interval but the behavior recording does not cover the
complete lifetime recording.

## NI-DAQ channel map

Embedded NI-DAQ channel-name metadata are ignored because the laboratory files
use this established physical row map:

| Hardware row | Python index | Signal |
|---:|---:|---|
| 1 | `data[0]` | photoreceiver 1 |
| 2 | `data[1]` | CAM synchronization TTL |
| 3 | `data[2]` | photoreceiver 2 |
| 4 | `data[3]` | licking TTL |
| 5 | `data[4]` | visual cue TTL |
| 6 | `data[5]` | 465-nm TTL |
| 7 | `data[6]` | 405-nm TTL |
| 8 | `data[7]` | Ensure/solenoid TTL |

## Data safety

- Never commit `.doric`, `.mat`, animal, or identifying data.
- Do not copy Doric manuals or vendor software into this repository.
- Record the Doric filename, device settings, IRF, and software version with
  every analysis.
- Do not interpret an exploratory raw-waveform fit biologically until its bias,
  precision, and calibration dependence have been characterized.
