"""CLI used by the GUI for FluoPulse and iFLiP3 analyses."""

from __future__ import annotations

import argparse
import json

from lutaslab_photometry.lifetime_workflows import (
    preprocess_lifetime_sessions,
    read_lifetime_manifest,
    run_lifetime_glm,
    run_lifetime_psth,
)
from lutaslab_photometry.qc_reporting import concise_qc_warnings
from lutaslab_photometry.session_manifest import filter_manifest_sessions
from lutaslab_photometry.trial_classification import TRIAL_CLASS_KEYS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preprocess", "psth", "glm"))
    parser.add_argument("--workflow", required=True, choices=("fluopulse", "iflip3"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--group", help="Analyze only this manifest group.")
    parser.add_argument("--condition", help="Analyze only this manifest condition.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--iflip3-fit-settings",
        help="JSON fit controls produced by the GUI for iFLIP3 preprocessing.",
    )
    parser.add_argument(
        "--iflip3-calibration",
        help="Saved multi-session iFLIP3 calibration JSON to apply during preprocessing.",
    )
    parser.add_argument(
        "--iflip3-temporal-bin-seconds",
        type=float,
        help="Pool corrected iFLIP3 photon histograms to this duration before analysis.",
    )
    parser.add_argument("--signal")
    parser.add_argument(
        "--event",
        choices=(
            "ensure",
            "visual_cue",
            "licks",
            "lick_bout_onset",
            "lick_bout_offset",
            "first_lick_after_ensure",
            "first_lick_bout_after_ensure",
        ),
    )
    parser.add_argument("--first-event-only", action="store_true")
    parser.add_argument("--allow-partial-windows", action="store_true")
    parser.add_argument("--window", nargs=2, type=float, default=(-5.0, 20.0))
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--normalization", choices=("subtract", "none"), default="subtract"
    )
    parser.add_argument("--baseline", nargs=2, type=float, default=(-5.0, 0.0))
    parser.add_argument("--y-limits", nargs=2, type=float)
    parser.add_argument(
        "--heatmaps", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--heatmap-sort", default="event_order")
    parser.add_argument("--heatmap-sort-window", nargs=2, type=float)
    parser.add_argument("--heatmap-sort-direction", default="auto")
    parser.add_argument("--heatmap-unmatched", default="bottom")
    parser.add_argument("--heatmap-cmap", default="coolwarm")
    parser.add_argument(
        "--null-method",
        choices=("none", "random_onsets", "circular_shift"),
        default="none",
    )
    parser.add_argument("--n-shuffles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null-exclusion", type=float, default=0.0)
    parser.add_argument("--trial-class", choices=TRIAL_CLASS_KEYS, default="all")
    parser.add_argument("--post-cue-window", type=float, default=2.0)
    parser.add_argument(
        "--derived-event-window", nargs=2, type=float, default=(0.0, 20.0)
    )
    parser.add_argument("--lick-kernel-seconds", type=float, default=10.0)
    parser.add_argument("--ensure-kernel-seconds", type=float, default=20.0)
    parser.add_argument("--pre-event-seconds", type=float, default=5.0)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rows = read_lifetime_manifest(args.workflow, args.manifest)
    rows = filter_manifest_sessions(rows, group=args.group, condition=args.condition)
    if args.action == "preprocess":
        if args.iflip3_fit_settings is not None and args.iflip3_calibration is not None:
            raise ValueError(
                "Use either --iflip3-fit-settings or --iflip3-calibration, not both"
            )
        with concise_qc_warnings():
            outputs = preprocess_lifetime_sessions(
                args.workflow,
                rows,
                args.data_root,
                overwrite=args.overwrite,
                iflip3_fit_settings=(
                    json.loads(args.iflip3_fit_settings)
                    if args.iflip3_fit_settings is not None
                    else None
                ),
                iflip3_calibration=args.iflip3_calibration,
                iflip3_temporal_bin_seconds=args.iflip3_temporal_bin_seconds,
            )
    elif args.action == "psth":
        if not args.signal or not args.event or not args.output_dir:
            raise ValueError("PSTH requires --signal, --event, and --output-dir")
        if args.y_limits is not None and args.y_limits[0] >= args.y_limits[1]:
            raise ValueError("--y-limits MIN must be less than MAX")
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
            y_limits=None if args.y_limits is None else tuple(args.y_limits),
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
            null_method=args.null_method,
            n_shuffles=args.n_shuffles,
            random_seed=args.seed,
            null_exclusion=args.null_exclusion,
            trial_class=args.trial_class,
            post_cue_window=args.post_cue_window,
            derived_event_window=tuple(args.derived_event_window),
            first_event_only=args.first_event_only,
            allow_partial_windows=args.allow_partial_windows,
            progress_callback=lambda current, total, label: print(
                f"PROGRESS\t{current}\t{total}\t{label}",
                flush=True,
            ),
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
            pre_event_seconds=args.pre_event_seconds,
            lick_kernel_seconds=args.lick_kernel_seconds,
            ensure_kernel_seconds=args.ensure_kernel_seconds,
            progress_callback=lambda current, total, label: print(
                f"PROGRESS\t{current}\t{total}\t{label}",
                flush=True,
            ),
        )
    for output in outputs:
        print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
