"""CLI used by the GUI for FluoPulse and iFLiP3 analyses."""

from __future__ import annotations

import argparse

from lutaslab_photometry.lifetime_workflows import (
    preprocess_lifetime_sessions,
    read_lifetime_manifest,
    run_lifetime_glm,
    run_lifetime_psth,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preprocess", "psth", "glm"))
    parser.add_argument("--workflow", required=True, choices=("fluopulse", "iflip3"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--signal")
    parser.add_argument("--event", choices=(
        "ensure", "visual_cue", "licks", "lick_bout_onset", "lick_bout_offset"
    ))
    parser.add_argument("--first-event-only", action="store_true")
    parser.add_argument("--allow-partial-windows", action="store_true")
    parser.add_argument("--window", nargs=2, type=float, default=(-5.0, 20.0))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--normalization", choices=("zscore", "subtract", "none"), default="zscore"
    )
    parser.add_argument("--baseline", nargs=2, type=float, default=(-5.0, 0.0))
    parser.add_argument("--heatmaps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heatmap-sort", default="event_order")
    parser.add_argument("--heatmap-sort-window", nargs=2, type=float)
    parser.add_argument("--heatmap-sort-direction", default="auto")
    parser.add_argument("--heatmap-unmatched", default="bottom")
    parser.add_argument("--heatmap-cmap", default="coolwarm")
    parser.add_argument("--lick-kernel-seconds", type=float, default=10.0)
    parser.add_argument("--ensure-kernel-seconds", type=float, default=20.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_lifetime_manifest(args.workflow, args.manifest)
    if args.action == "preprocess":
        outputs = preprocess_lifetime_sessions(
            args.workflow,
            rows,
            args.data_root,
            overwrite=args.overwrite,
        )
    elif args.action == "psth":
        if not args.signal or not args.event or not args.output_dir:
            raise ValueError("PSTH requires --signal, --event, and --output-dir")
        outputs = run_lifetime_psth(
            args.workflow,
            rows,
            args.data_root,
            args.output_dir,
            signal=args.signal,
            event=args.event,
            window=tuple(args.window),
            dt=args.dt,
            normalization=args.normalization,
            baseline=tuple(args.baseline),
            heatmaps=args.heatmaps,
            heatmap_sort=args.heatmap_sort,
            heatmap_sort_window=(
                None
                if args.heatmap_sort_window is None
                else tuple(args.heatmap_sort_window)
            ),
            heatmap_sort_direction=args.heatmap_sort_direction,
            heatmap_unmatched=args.heatmap_unmatched,
            heatmap_cmap=args.heatmap_cmap,
            first_event_only=args.first_event_only,
            allow_partial_windows=args.allow_partial_windows,
        )
    else:
        if not args.signal or not args.output_dir:
            raise ValueError("GLM requires --signal and --output-dir")
        outputs = run_lifetime_glm(
            args.workflow,
            rows,
            args.data_root,
            args.output_dir,
            signal=args.signal,
            lick_kernel_seconds=args.lick_kernel_seconds,
            ensure_kernel_seconds=args.ensure_kernel_seconds,
        )
    for output in outputs:
        print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
