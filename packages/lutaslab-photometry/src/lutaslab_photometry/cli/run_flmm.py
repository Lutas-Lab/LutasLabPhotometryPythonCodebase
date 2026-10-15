"""Prepare or fit a non-concurrent functional linear mixed model."""

from __future__ import annotations

import argparse
from pathlib import Path

from lutaslab_core.manifest import load_session_manifest

from lutaslab_photometry.functional_mixed import (
    CONVENTIONAL_FLMM_EVENTS,
    CONVENTIONAL_FLMM_SIGNALS,
    FLMM_EFFECTS,
    build_flmm_dataset,
    prepare_flmm_analysis,
    run_prepared_flmm,
)
from lutaslab_photometry.lifetime_workflows import (
    LIFETIME_EVENTS,
    LIFETIME_SIGNALS,
    read_lifetime_manifest,
)
from lutaslab_photometry.session_manifest import filter_manifest_sessions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow", required=True, choices=("conventional", "fluopulse", "iflip3")
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--effect", choices=FLMM_EFFECTS, default="condition")
    parser.add_argument("--channel", choices=("manifest", "1", "2"), default="manifest")
    parser.add_argument("--window", nargs=2, type=float, default=(-5.0, 20.0))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--normalization", choices=("zscore", "subtract", "none"), default="zscore"
    )
    parser.add_argument("--baseline", nargs=2, type=float, default=(-5.0, 0.0))
    parser.add_argument("--group", help="Analyze only this manifest group.")
    parser.add_argument("--condition", help="Analyze only this manifest condition.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rscript", type=Path)
    parser.add_argument("--cores", type=int, default=1)
    return parser


def _validate_choices(args) -> None:
    if args.workflow == "conventional":
        valid_signals = CONVENTIONAL_FLMM_SIGNALS
        valid_events = CONVENTIONAL_FLMM_EVENTS
    else:
        valid_signals = LIFETIME_SIGNALS[args.workflow]
        valid_events = LIFETIME_EVENTS
    if args.signal not in valid_signals:
        raise ValueError(
            f"Signal {args.signal!r} is not available for {args.workflow}; "
            f"choose one of {tuple(valid_signals)}"
        )
    if args.event not in valid_events:
        raise ValueError(
            f"Event {args.event!r} is not available for {args.workflow}; "
            f"choose one of {tuple(valid_events)}"
        )
    if args.cores < 1:
        raise ValueError("--cores must be positive")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _validate_choices(args)
    if args.workflow == "conventional":
        rows = load_session_manifest(args.manifest)
    else:
        rows = read_lifetime_manifest(args.workflow, args.manifest)
    rows = filter_manifest_sessions(rows, group=args.group, condition=args.condition)
    dataset = build_flmm_dataset(
        args.workflow,
        rows,
        args.data_root,
        signal=args.signal,
        event=args.event,
        window=tuple(args.window),
        dt=args.dt,
        normalization=args.normalization,
        baseline=tuple(args.baseline),
        channel=args.channel,
    )
    outputs = prepare_flmm_analysis(dataset, args.output_dir, effect=args.effect)
    if not args.prepare_only:
        outputs.extend(
            run_prepared_flmm(
                args.output_dir,
                rscript=args.rscript,
                cores=args.cores,
            )
        )
    for output in outputs:
        print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
