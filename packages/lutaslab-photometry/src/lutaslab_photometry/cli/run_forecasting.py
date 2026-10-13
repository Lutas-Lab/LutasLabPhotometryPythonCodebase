import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from lutaslab_photometry.forecasting import (
    DEFAULT_ALPHAS,
    MODEL_ORDER,
    VALID_TARGETS,
    forecast_manifest,
    primary_metric,
    summarize_forecast_kernels,
    summarize_forecasts,
)
from lutaslab_photometry.publication_figures import configure_publication_style, save_figure_formats
from lutaslab_photometry.session_manifest import (
    filter_manifest_sessions,
    load_session_manifest,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Forecast future photometry or behavior from past-only signals."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group", help="Analyze only this manifest group.")
    parser.add_argument("--condition", help="Analyze only this manifest condition.")
    parser.add_argument("--target", choices=sorted(VALID_TARGETS), required=True)
    parser.add_argument("--horizons", type=float, nargs="+", default=(0.5, 1.0, 2.0, 5.0))
    parser.add_argument("--history", type=float, default=5.0)
    parser.add_argument("--lag-step", type=float, default=0.5)
    parser.add_argument("--target-window", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument(
        "--channel",
        choices=("manifest", "1", "2"),
        default="manifest",
        help="Use each manifest row's channel, or override every session.",
    )
    parser.add_argument(
        "--photometry-source", choices=("raw465", "dff"), default="raw465"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.5)
    parser.add_argument("--gap-seconds", type=float, default=None)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Candidate strengths selected within each outer training fold.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Use one prespecified strength instead of nested selection.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("svg", "png", "pdf"),
        default=("svg", "png"),
    )
    parser.add_argument("--font-family", default="Arial")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _write_rows(path, rows):
    preferred = [
        "mouse",
        "date",
        "run",
        "channel",
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
        "alpha_selection",
        "selected_alphas",
        "alpha_min",
        "alpha_median",
        "alpha_max",
        "r2",
        "mse",
        "mae",
        "average_precision",
        "positive_prevalence",
        "average_precision_gain_over_prevalence",
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


def _plot_group_performance(
    mouse_rows, group_rows, target, metric, output_base, formats, dpi
):
    fig, ax = plt.subplots(figsize=(8, 5))
    horizons_present = sorted({float(row["horizon"]) for row in group_rows})
    if len(horizons_present) == 1:
        horizon = horizons_present[0]
        x = np.arange(len(MODEL_ORDER), dtype=float)
        for mouse in sorted({row["mouse"] for row in mouse_rows}):
            lookup = {
                row["model"]: row["value"]
                for row in mouse_rows
                if row["mouse"] == mouse and row["horizon"] == horizon
            }
            values = np.asarray([lookup.get(model, np.nan) for model in MODEL_ORDER])
            ax.plot(x, values, color="0.65", marker="o", linewidth=0.8, alpha=0.55)
        group_lookup = {
            row["model"]: row
            for row in group_rows
            if row["horizon"] == horizon
        }
        means = np.asarray(
            [group_lookup.get(model, {}).get("mean", np.nan) for model in MODEL_ORDER]
        )
        sems = np.asarray(
            [group_lookup.get(model, {}).get("sem", np.nan) for model in MODEL_ORDER]
        )
        ax.errorbar(x, means, yerr=sems, color="C0", marker="o", linewidth=2.2)
        ax.set(
            xticks=x,
            xticklabels=[model.replace("_", " ") for model in MODEL_ORDER],
            xlabel="Model",
            title=f"Present {target.replace('_', ' ')} prediction",
        )
        ax.tick_params(axis="x", rotation=15)
    else:
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
            (line,) = ax.plot(
                horizons,
                means,
                marker="o",
                linewidth=2,
                label=model.replace("_", " "),
            )
            ax.fill_between(horizons, means - sems, means + sems, alpha=0.2)
            mouse_names = sorted(
                {row["mouse"] for row in mouse_rows if row["model"] == model}
            )
            for mouse in mouse_names:
                mouse_selected = sorted(
                    (
                        row
                        for row in mouse_rows
                        if row["model"] == model and row["mouse"] == mouse
                    ),
                    key=lambda row: row["horizon"],
                )
                ax.plot(
                    [row["horizon"] for row in mouse_selected],
                    [row["value"] for row in mouse_selected],
                    marker=".",
                    linewidth=0.7,
                    alpha=0.25,
                    color=line.get_color(),
                )
        ax.set(
            xlabel="Forecast horizon (s)",
            title=f"Future {target.replace('_', ' ')} forecast",
        )
        ax.legend()
    if metric != "average_precision":
        ax.axhline(0, color="black", linestyle=":", linewidth=1)
    else:
        ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " "))
    fig.tight_layout()
    paths = save_figure_formats(fig, output_base, formats=formats, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_predictor_kernels(
    mouse_rows,
    group_rows,
    *,
    target,
    photometry_source,
    output_base,
    formats,
    dpi,
):
    """Plot full-model predictor kernels with mice and group mean ± SEM."""

    predictor_order = ("photometry", "locomotion", "licking", "cue", "solenoid")
    labels = {
        "photometry": "Photometry history",
        "locomotion": "Running speed",
        "licking": "Licking",
        "cue": "Cue onset",
        "solenoid": "Ensure onset",
    }
    horizons = sorted(
        {
            float(row["horizon"])
            for row in group_rows
            if row["model"] == "combined"
        }
    )
    output_paths = []
    for horizon in horizons:
        predictors = [
            predictor
            for predictor in predictor_order
            if any(
                row["model"] == "combined"
                and row["horizon"] == horizon
                and row["predictor"] == predictor
                for row in group_rows
            )
        ]
        if not predictors:
            continue
        ncols = min(3, len(predictors))
        nrows = int(np.ceil(len(predictors) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.4 * nrows),
            squeeze=False,
            sharex=True,
        )
        for ax, predictor in zip(axes.flat, predictors, strict=False):
            selected = sorted(
                (
                    row
                    for row in group_rows
                    if row["model"] == "combined"
                    and row["horizon"] == horizon
                    and row["predictor"] == predictor
                ),
                key=lambda row: row["lag_s"],
            )
            lag = np.asarray([row["lag_s"] for row in selected], dtype=float)
            mean = np.asarray([row["mean"] for row in selected], dtype=float)
            sem = np.asarray([row["sem"] for row in selected], dtype=float)
            for mouse in sorted(
                {
                    row["mouse"]
                    for row in mouse_rows
                    if row["model"] == "combined"
                    and row["horizon"] == horizon
                    and row["predictor"] == predictor
                }
            ):
                mouse_selected = sorted(
                    (
                        row
                        for row in mouse_rows
                        if row["model"] == "combined"
                        and row["horizon"] == horizon
                        and row["predictor"] == predictor
                        and row["mouse"] == mouse
                    ),
                    key=lambda row: row["lag_s"],
                )
                ax.plot(
                    [row["lag_s"] for row in mouse_selected],
                    [row["coefficient"] for row in mouse_selected],
                    color="0.65",
                    linewidth=0.8,
                    alpha=0.45,
                )
            ax.plot(lag, mean, color="C0", linewidth=2.2)
            ax.fill_between(lag, mean - sem, mean + sem, color="C0", alpha=0.25)
            ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
            ax.set_title(labels.get(predictor, predictor))
            ax.set_xlabel("Time after predictor (s)")
            ax.set_ylabel("GLM coefficient")
        for ax in axes.flat[len(predictors) :]:
            ax.set_visible(False)
        response = (
            "processed dF/F" if photometry_source == "dff" else "raw 465 fluorescence"
        )
        fig.suptitle(
            f"Combined GLM kernels for {response}\n"
            f"individual mice and mean ± SEM; horizon = {horizon:g} s"
        )
        fig.tight_layout()
        base = (
            output_base
            if len(horizons) == 1
            else output_base.with_name(f"{output_base.name}_horizon_{horizon:g}s")
        )
        output_paths.extend(
            save_figure_formats(fig, base, formats=formats, dpi=dpi)
        )
        plt.close(fig)
    return output_paths


def main():
    args = parse_arguments()
    sessions = load_session_manifest(args.manifest)
    sessions = filter_manifest_sessions(
        sessions,
        group=args.group,
        condition=args.condition,
    )
    rows, kernel_rows = forecast_manifest(
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
        candidate_alphas=args.alphas,
        n_inner_folds=args.inner_folds,
        gap_seconds=args.gap_seconds,
        initial_train_fraction=args.initial_train_fraction,
        return_kernels=True,
    )
    metric = primary_metric(args.target)
    mouse_rows, group_rows = summarize_forecasts(rows, metric)
    mouse_kernel_rows, group_kernel_rows = summarize_forecast_kernels(kernel_rows)
    configure_publication_style(font_family=args.font_family)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_path = args.output_dir / "forecast_session_metrics.csv"
    mouse_path = args.output_dir / "forecast_mouse_metrics.csv"
    group_path = args.output_dir / "forecast_group_metrics.csv"
    session_kernel_path = args.output_dir / "forecast_session_kernels.csv"
    mouse_kernel_path = args.output_dir / "forecast_mouse_kernels.csv"
    group_kernel_path = args.output_dir / "forecast_group_kernels.csv"
    figure_base = args.output_dir / "forecast_performance"
    kernel_figure_base = args.output_dir / "forecast_predictor_kernels"
    metadata_path = args.output_dir / "forecast_metadata.json"
    _write_rows(session_path, rows)
    _write_rows(mouse_path, mouse_rows)
    _write_rows(group_path, group_rows)
    _write_rows(session_kernel_path, kernel_rows)
    _write_rows(mouse_kernel_path, mouse_kernel_rows)
    _write_rows(group_kernel_path, group_kernel_rows)
    figure_paths = _plot_group_performance(
        mouse_rows,
        group_rows,
        args.target,
        metric,
        figure_base,
        args.formats,
        args.dpi,
    )
    kernel_figure_paths = _plot_predictor_kernels(
        mouse_kernel_rows,
        group_kernel_rows,
        target=args.target,
        photometry_source=args.photometry_source,
        output_base=kernel_figure_base,
        formats=args.formats,
        dpi=args.dpi,
    )

    metadata = {
        "manifest": str(args.manifest),
        "group_filter": args.group,
        "condition_filter": args.condition,
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
        "inner_folds": args.inner_folds,
        "initial_train_fraction": args.initial_train_fraction,
        "gap_seconds": args.gap_seconds,
        "alpha": args.alpha,
        "candidate_alphas": args.alphas,
        "alpha_selection": "fixed" if args.alpha is not None else "nested_forward_cv",
        "primary_metric": metric,
        "models": MODEL_ORDER,
        "kernel_model": "combined",
        "kernel_summary": "sessions averaged within mouse; mean and SEM across mice",
        "kernel_coefficient_scale": "native predictor units",
        "binary_baseline": "fold-local outer-training prevalence",
        "figure_formats": args.formats,
        "font_family": args.font_family,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Forecasted {args.target} for {len(sessions)} sessions.")
    print(f"Primary group metric: {metric}")
    print(f"Saved session metrics: {session_path}")
    print(f"Saved mouse metrics: {mouse_path}")
    print(f"Saved group metrics: {group_path}")
    print(f"Saved session kernels: {session_kernel_path}")
    print(f"Saved mouse kernels: {mouse_kernel_path}")
    print(f"Saved group kernels: {group_kernel_path}")
    for figure_path in (*figure_paths, *kernel_figure_paths):
        print(f"Saved figure: {figure_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
