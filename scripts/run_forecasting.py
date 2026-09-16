import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting import (
    MODEL_ORDER,
    VALID_TARGETS,
    forecast_manifest,
    primary_metric,
    summarize_forecasts,
)
from src.session_manifest import load_session_manifest


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Forecast future photometry or behavior from past-only signals."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(VALID_TARGETS), required=True)
    parser.add_argument("--horizons", type=float, nargs="+", default=(0.5, 1.0, 2.0, 5.0))
    parser.add_argument("--history", type=float, default=5.0)
    parser.add_argument("--lag-step", type=float, default=0.5)
    parser.add_argument("--target-window", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--channel", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--photometry-source", choices=("raw465", "dff"), default="raw465"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--initial-train-fraction", type=float, default=0.5)
    parser.add_argument("--gap-seconds", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _write_rows(path, rows):
    preferred = [
        "mouse",
        "date",
        "run",
        "target",
        "horizon",
        "model",
        "metric",
        "value",
        "mean",
        "sem",
        "n_sessions",
        "n_mice",
        "n_samples",
        "n_test",
        "n_folds",
        "gap_seconds",
        "r2",
        "mse",
        "mae",
        "average_precision",
        "roc_auc",
        "log_loss",
        "poisson_deviance",
        "negative_poisson_deviance",
        "path",
    ]
    available = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in available]
    fieldnames.extend(sorted(available.difference(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_group_performance(group_rows, target, metric, path, dpi):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODEL_ORDER:
        selected = sorted(
            (row for row in group_rows if row["model"] == model),
            key=lambda row: row["horizon"],
        )
        if not selected:
            continue
        horizons = np.asarray([row["horizon"] for row in selected], dtype=float)
        means = np.asarray([row["mean"] for row in selected], dtype=float)
        sems = np.asarray([row["sem"] for row in selected], dtype=float)
        ax.plot(horizons, means, marker="o", linewidth=2, label=model.replace("_", " "))
        ax.fill_between(horizons, means - sems, means + sems, alpha=0.2)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Forecast horizon (s)",
        ylabel=metric.replace("_", " "),
        title=f"Future {target.replace('_', ' ')} forecast",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_arguments()
    sessions = load_session_manifest(args.manifest)
    rows = forecast_manifest(
        sessions,
        args.data_root,
        target=args.target,
        horizons=args.horizons,
        dt=args.dt,
        history=args.history,
        lag_step=args.lag_step,
        target_window=args.target_window,
        channel=args.channel,
        photometry_source=args.photometry_source,
        n_folds=args.folds,
        alpha=args.alpha,
        gap_seconds=args.gap_seconds,
        initial_train_fraction=args.initial_train_fraction,
    )
    metric = primary_metric(args.target)
    mouse_rows, group_rows = summarize_forecasts(rows, metric)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_path = args.output_dir / "forecast_session_metrics.csv"
    mouse_path = args.output_dir / "forecast_mouse_metrics.csv"
    group_path = args.output_dir / "forecast_group_metrics.csv"
    figure_path = args.output_dir / "forecast_performance.png"
    metadata_path = args.output_dir / "forecast_metadata.json"
    _write_rows(session_path, rows)
    _write_rows(mouse_path, mouse_rows)
    _write_rows(group_path, group_rows)
    _plot_group_performance(group_rows, args.target, metric, figure_path, args.dpi)

    metadata = {
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "target": args.target,
        "horizons": args.horizons,
        "history": args.history,
        "lag_step": args.lag_step,
        "target_window": args.target_window,
        "dt": args.dt,
        "channel": args.channel,
        "photometry_source": args.photometry_source,
        "folds": args.folds,
        "initial_train_fraction": args.initial_train_fraction,
        "gap_seconds": args.gap_seconds,
        "alpha": args.alpha,
        "primary_metric": metric,
        "models": MODEL_ORDER,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Forecasted {args.target} for {len(sessions)} sessions.")
    print(f"Primary group metric: {metric}")
    print(f"Saved session metrics: {session_path}")
    print(f"Saved mouse metrics: {mouse_path}")
    print(f"Saved group metrics: {group_path}")
    print(f"Saved figure: {figure_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
