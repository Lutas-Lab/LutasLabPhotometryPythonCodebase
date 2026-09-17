import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_processing import preprocess_manifest_sessions
from src.session_manifest import load_session_manifest


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess every session in a CSV manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace processed files that already exist. Default: skip them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later sessions after a session fails.",
    )
    parser.add_argument(
        "--channel-map",
        type=json.loads,
        default=None,
        help="Optional JSON object mapping signal names to zero-based data rows.",
    )
    parser.add_argument("--ttl-threshold", type=float, default=1.5)
    parser.add_argument("--photometry-edge", type=int, default=3)
    parser.add_argument("--irls-constant", type=float, default=1.4)
    parser.add_argument("--cue-max-pulse-gap", type=float, default=0.5)
    parser.add_argument("--locomotion-min-width", type=int, default=4)
    parser.add_argument("--locomotion-max-width", type=int, default=6)
    parser.add_argument(
        "--locomotion-invert",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--post-cue-window", type=float, default=2.0)
    parser.add_argument("--lick-bout-interval", type=float, default=1.0)
    parser.add_argument("--minimum-bout-licks", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_arguments()
    sessions = load_session_manifest(args.manifest)
    options = {
        "ttl_threshold": args.ttl_threshold,
        "photometry_edge": args.photometry_edge,
        "irls_constant": args.irls_constant,
        "cue_max_pulse_gap": args.cue_max_pulse_gap,
        "locomotion_min_width": args.locomotion_min_width,
        "locomotion_max_width": args.locomotion_max_width,
        "locomotion_invert": args.locomotion_invert,
        "post_cue_window": args.post_cue_window,
        "lick_bout_interval": args.lick_bout_interval,
        "minimum_bout_licks": args.minimum_bout_licks,
    }
    print(f"Loaded {len(sessions)} sessions from {args.manifest}")
    results = preprocess_manifest_sessions(
        sessions,
        args.data_root,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
        channel_map=args.channel_map,
        preprocessing_options=options,
    )

    print("\nBatch summary")
    for result in results:
        label = f"{result['mouse']} {result['date']} run {result['run']}"
        print(f"  {result['status']:9s} {label}: {result['path']}")
        if result["error"]:
            print(f"             {result['error']}")

    failures = [result for result in results if result["status"] == "failed"]
    print(
        "\n"
        f"Processed: {sum(result['status'] == 'processed' for result in results)}; "
        f"skipped: {sum(result['status'] == 'skipped' for result in results)}; "
        f"failed: {len(failures)}"
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
