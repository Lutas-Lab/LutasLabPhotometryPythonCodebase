# Validation record

Validated on 2026-09-24 against six local SC81 Doric FluoPulse recordings. The
recordings and animal identifiers are excluded from version control.

All six files opened as HDF5 and contained vendor lifetime, amplitude,
chi-square, R-square, raw waveform, IRF, and digital-input datasets. Calculation
sampling was approximately 10 Hz, each raw waveform contained 4,096 points,
and digital inputs were sampled at 10 kHz.

Configuration metadata identified DIO02 as `CAMpulse`, DIO04 as `lick_event`,
and DIO05 as `ensure_delivery`. DIO02 contained approximately 30 Hz, 2 ms sync
pulses in all recordings. DIO04 contained approximately 10-11 ms lick pulses.
DIO05 was configured but remained continuously low in all six recordings, so
the matching NI-DAQ Ensure channel should be checked before diagnosing wiring.

Run 1 contained 3,343 vendor lifetime samples over approximately 334.6 seconds,
10,037 CAM pulses, and 416 lick pulses. Mean `Tau01` was approximately 2.658 ns,
mean amplitude approximately 83.9, and mean fit R-square approximately 0.99984.

The independent waveform-analysis functions are synthetic-test targets only at
this stage. They are not validated replacements for Doric's lifetime output.

## Paired Doric and NI-DAQ validation

The six recordings were compared with the matching local NI-DAQ files. Runs 1
and 3--6 had complete behavior coverage, identical Doric/NI lick counts,
0.123--0.145 ms CAM residual RMS, and 3.59--4.01 ppm fitted clock drift. Run 2's
NI-DAQ file contains only 8.7 seconds for a 243.9-second Doric recording. Its
shared section aligns, but it is a truncated behavior file and is unsuitable
for full-session behavioral analysis.

## Legacy descriptive FluoPulse files

Three SC43 files using the older `Calculation`, `AnalogIn`, and configuration
IRF layout were compared with their vendor-exported tau and behavior CSVs. The
embedded `Tau01` values and timestamps matched the tau CSVs to floating-point
precision. All CAM, lick, and Ensure rising edges matched the behavior CSVs
exactly. These `.doric` files can therefore be used alone for lifetime analysis
aligned to embedded lick and Ensure events. The food-reward recording contained
23,069 CAM pulses, 797 licks, and 40 Ensure events; the two 10-minute recordings
contained licks but no saved CAM or Ensure pulses.
