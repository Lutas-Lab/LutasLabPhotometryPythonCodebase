# Validation record

Validated on 2026-09-22 against a local six-file example dataset. The dataset
and animal/session identifier are intentionally excluded from version control.

## File loading

All six files loaded successfully with the safe Python parser.

| Recording | Loaded data shape | Mean photons/sample |
|---|---:|---:|
| Example recording 001 | 126 x 3000 x 1 | 411443.682 |
| Example recording 002 | 126 x 3000 x 1 | 400419.181 |
| Example recording 003 | 126 x 3000 x 1 | 622222.095 |
| Example recording 004 | 126 x 3000 x 1 | 600562.931 |
| Example background 001 | 126 x 100 x 1 | background input |
| Example background 002 | 126 x 100 x 1 | background input |

Background 001 contained 3049.2 counts/sample and background 002 contained
4991.1 counts/sample. They were acquired at different laser powers and were
therefore not pooled. Background 001 was paired with recordings 001–002 and
background 002 with recordings 003–004, consistent with their timestamps and
the two measurement-intensity groups. Parsed metadata matched the payload
dimensions: 10 Hz, 300 seconds for each mouse recording, one channel, 126
lifetime bins, 0.1 ns/bin, and an approximately 12.5 ns excitation period.

## Background correction and MPET

Each measurement was corrected with its laser-power-matched background.
Dead-time-aware background correction completed without invalid count
efficiencies. Using SPC range 0.4–12.3 ns, each file's header `t0`, and the
configured afterpulse ratio of `0.03`:

| Recording | Mean MPET (ns) |
|---|---:|
| Example recording 001 | 1.321408 |
| Example recording 002 | 1.283499 |
| Example recording 003 | 1.273911 |
| Example recording 004 | 1.258768 |

These values are computational validation outputs, not biological conclusions.
Approximately 0.794% of corrected bins were negative after background and
afterpulse subtraction. They were retained rather than clipped, because clipping
background-subtracted noise would bias MPET and fitted component contributions.
The `0.03` ratio is configurable and can be replaced by a later calibration.

## Model validation

The periodic Gaussian-convolved exponential was numerically compared with the
formula in `h_fitLifetimeBySingleExp.m` at the acquisition's representative
parameters (`tau=2.5 ns`, `t0=1.0086569 ns`, `sigma=0.1482341 ns`). Values agreed
to relative tolerance `2e-9`. Area-normalized bases summed to one.

Source, examples, and tests compiled successfully with Python 3.12. Reader,
header-parser, correction, MPET, binning, and model checks passed locally.

## Fitting and workflow validation

The project-local Conda environment supplied SciPy, Matplotlib, and pytest.
All 16 unit tests passed, including recovery of known single-decay and shared
double-decay lifetimes, fixed-basis component counts, lifetime-window selection,
MPET, correction, binary decoding, explicit NI-DAQ channel mapping, TTL event
detection, clock conversion, and construction of the optional Pynapple objects.

The recommended command-line workflow completed on example recording 004 with
its matching background 002, afterpulse ratio `0.03`, one-second bins, and the
0.4–12.3 ns window. It estimated aggregate lifetimes of approximately 0.488 ns
and 1.722 ns. The fixed-basis long photon fraction and MPET time series had a
correlation of approximately 0.991. These are computational validation results,
not biological conclusions or a calibration of molecular state fraction.

MATLAB R2026a was present, but its license service returned error 5201 before a
vendor-reference run could complete. Direct MATLAB parity remains an optional
future check if that license service becomes available.

## NI-DAQ synchronization and behavior

Validated on 2026-09-24 against four local paired iFLiP/NI-DAQ recordings. The
incorrect NI-DAQ channel-name metadata were ignored; row 2 (`data[1]`) was used
for the 5 Hz synchronization train, row 4 (`data[3]`) for licking, and row 8
(`data[7]`) for the Ensure/solenoid TTL.

All four iFLiP Marker 1 trains and NI-DAQ synchronization trains had complete
internal pulse sequences. An affine iFLiP-to-NI-DAQ clock fit gave approximately
9.06–9.25 ppm drift across recordings. Synchronization residual RMS was
0.120–0.126 ms and the largest residual was approximately 0.255 ms. Matched
pulse counts were 1498, 1497, 1500, and 1494. Running-speed vector lengths
matched the NI-DAQ synchronization pulse counts in all four recordings.

The optional Pynapple adapter is covered with an API test double. A live
Pynapple installation was not available in the validation environment and must
be installed with the project's `events` extra before running the example
peri-event script.

These residuals validate the clock conversion, not the temporal resolution of
the lifetime readout. The iFLiP lifetime series is sampled at 10 Hz and therefore
has 0.1-second sample spacing after alignment.
