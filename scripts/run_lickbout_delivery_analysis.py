import argparse
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.group_analysis import (
    save_condition_comparison_figures,
    save_psth_figures,
)
from src.lickbout_delivery import (
    compute_lickbout_delivery_strata,
    save_delivery_sorted_heatmap,
    save_delivery_trial_data,
)
from src.session_manifest import load_session_manifest


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create lick-bout-aligned photometry PSTHs and delivery-sorted "
            "trial heatmaps."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", default="Astrocyte")
    parser.add_argument(
        "--channel",
        choices=("manifest", "1", "2"),
        default="manifest",
    )
    parser.add_argument("--window", type=float, nargs=2, default=(-5.0, 20.0))
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument(
        "--normalization",
        choices=("none", "subtract", "zscore"),
        default="zscore",
    )
    parser.add_argument("--baseline", type=float, nargs=2, default=(-5.0, 0.0))
    parser.add_argument(
        "--minimum-delivery-latency",
        type=float,
        default=0.0,
        help=(
            "Exclude trials whose solenoid onset minus lick-bout onset is below "
            "this value (default: 0 seconds)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("svg", "png", "pdf"),
        default=("svg", "png"),
    )
    parser.add_argument("--font-family", default="Arial")
    return parser.parse_args()


def _safe_path_component(value):
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(value)
    )
    return safe or "all"


def main():
    args = parse_arguments()
    sessions = load_session_manifest(args.manifest)
    selected = [
        session
        for session in sessions
        if str(session.get("group", "")).casefold() == args.group.casefold()
    ]
    if not selected:
        raise ValueError(f"No manifest sessions matched group {args.group!r}.")

    results_by_stratum = compute_lickbout_delivery_strata(
        selected,
        args.data_root,
        window=args.window,
        dt=args.dt,
        normalization=args.normalization,
        baseline=args.baseline,
        channel=args.channel,
        minimum_delivery_latency=args.minimum_delivery_latency,
    )
    saved = []
    for (group, condition), results in results_by_stratum.items():
        output_dir = (
            args.output_dir
            / _safe_path_component(group)
            / _safe_path_component(condition)
        )
        saved.extend(
            save_psth_figures(
                results,
                output_dir,
                figure_level="both",
                formats=args.formats,
                dpi=args.dpi,
                font_family=args.font_family,
            )
        )
        saved.extend(
            save_delivery_sorted_heatmap(
                results,
                output_dir,
                formats=args.formats,
                dpi=args.dpi,
                font_family=args.font_family,
            )
        )
        npz_path, csv_path = save_delivery_trial_data(results, output_dir)
        print(
            f"{group} / {condition}: {len(results['trial_rows'])} paired trials, "
            f"{results['n_mice']} mice."
        )
        print(f"Saved trial data: {npz_path}")
        print(f"Saved trial table: {csv_path}")

    saved.extend(
        save_condition_comparison_figures(
            results_by_stratum,
            args.output_dir / "comparisons",
            formats=args.formats,
            dpi=args.dpi,
            font_family=args.font_family,
        )
    )
    for path in saved:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
