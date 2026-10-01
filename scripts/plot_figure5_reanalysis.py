"""Create Figure 5-style B, F, I, and J panels from the reanalysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.figure5_reanalysis import (
    build_figure5_design,
    fit_nested_blocked_ridge,
    load_figure5_trials,
    load_raw_figure5_trials,
    model_columns,
    reconstruct_figure5_kernels,
)


COLORS = {
    "kernel": "#2F73D5",
    "actual": "#222222",
    "prediction": "#E31A1C",
    "residual": "#EC168C",
    "licking": "#228B22",
    "cue": "#CFE8F7",
}


def _sem(values: np.ndarray) -> np.ndarray:
    return np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])


def _smooth_rate(events: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    width = max(1, int(round(0.5 * sample_rate_hz)))
    kernel = np.ones(width) / width
    return np.convolve(events * sample_rate_hz, kernel, mode="same")


def _predict_full_trials_from_dry_fit(design, dry_fit, dry_columns) -> np.ndarray:
    prediction = np.full(design.response.size, np.nan)
    for block, coefficients in zip(
        np.unique(design.block_groups), dry_fit.outer_coefficients, strict=True
    ):
        test = design.block_groups == block
        prediction[test] = design.matrix[test][:, dry_columns] @ coefficients
    return prediction


def _collect_panel_data(trials, mouse_ids: np.ndarray, example_mouse: int) -> dict:
    alphas = np.logspace(-3, 5, 9)
    samples_per_trial = trials.photometry.shape[1]
    dry_stop = 654
    dry_kernels = []
    full_kernels = []
    actual_means = []
    dry_residual_means = []
    full_residual_means = []
    lick_rate_means = []
    example = {}

    for mouse_id in mouse_ids:
        design = build_figure5_design(trials, int(mouse_id), lick_basis_count=12)
        trial_count = design.trial_numbers.size
        lick_columns = model_columns(design, ("lick",))
        full_columns = model_columns(design, ("lick", "cue", "ensure"))

        full_fit = fit_nested_blocked_ridge(
            design.matrix[:, full_columns],
            design.response,
            design.block_groups,
            alphas,
            penalty_weights=design.penalty_weights[full_columns],
        )
        full_coefficients = np.zeros(design.matrix.shape[1])
        full_coefficients[full_columns] = full_fit.final_fit.coefficients
        full_kernel = reconstruct_figure5_kernels(design, full_coefficients)["lick"]

        matrix_trials = design.matrix[:, lick_columns].reshape(
            trial_count, samples_per_trial, -1
        )
        response_trials = design.response.reshape(trial_count, samples_per_trial)
        group_trials = design.block_groups.reshape(trial_count, samples_per_trial)
        dry_matrix = matrix_trials[:, :dry_stop].reshape(-1, lick_columns.size)
        dry_response = response_trials[:, :dry_stop].reshape(-1)
        dry_groups = group_trials[:, :dry_stop].reshape(-1)
        dry_fit = fit_nested_blocked_ridge(
            dry_matrix,
            dry_response,
            dry_groups,
            alphas,
            penalty_weights=design.penalty_weights[lick_columns],
        )
        dry_coefficients = np.zeros(design.matrix.shape[1])
        dry_coefficients[lick_columns] = dry_fit.final_fit.coefficients
        dry_kernel = reconstruct_figure5_kernels(design, dry_coefficients)["lick"]
        dry_full_prediction = _predict_full_trials_from_dry_fit(
            design, dry_fit, lick_columns
        )

        rows = np.flatnonzero(trials.mouse_ids == mouse_id)
        actual = response_trials.mean(axis=0)
        dry_residual = (
            design.response - dry_full_prediction
        ).reshape(trial_count, samples_per_trial).mean(axis=0)
        full_residual = (
            design.response - full_fit.prediction
        ).reshape(trial_count, samples_per_trial).mean(axis=0)
        lick_rate = np.mean(
            [_smooth_rate(row, trials.sample_rate_hz) for row in trials.lick_events[rows]],
            axis=0,
        )

        dry_kernels.append(dry_kernel[1])
        full_kernels.append(full_kernel[1])
        actual_means.append(actual)
        dry_residual_means.append(dry_residual)
        full_residual_means.append(full_residual)
        lick_rate_means.append(lick_rate)
        if int(mouse_id) == example_mouse:
            example = {"dry": dry_kernel[1], "full": full_kernel[1]}
        lag_seconds = dry_kernel[0]

    if not example:
        raise ValueError(f"example mouse {example_mouse} was not analyzed")
    return {
        "lag_seconds": lag_seconds,
        "dry_kernels": np.vstack(dry_kernels),
        "full_kernels": np.vstack(full_kernels),
        "actual": np.vstack(actual_means),
        "dry_residual": np.vstack(dry_residual_means),
        "full_residual": np.vstack(full_residual_means),
        "lick_rate": np.vstack(lick_rate_means),
        "example": example,
        "sample_rate_hz": trials.sample_rate_hz,
    }


def _style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.8)


def _kernel_panel(fig, spec, data: dict, *, mode: str, letter: str, title: str):
    grid = spec.subgridspec(2, 1, hspace=0.55)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1], sharex=top)
    time = data["lag_seconds"]
    kernels = data[f"{mode}_kernels"]
    mean = kernels.mean(axis=0)
    sem = _sem(kernels)
    top.plot(time, data["example"][mode], color=COLORS["kernel"], linewidth=1.2)
    bottom.plot(time, mean, color=COLORS["kernel"], linewidth=1.4)
    bottom.fill_between(time, mean - sem, mean + sem, color=COLORS["kernel"], alpha=0.2)
    for axis in (top, bottom):
        axis.axhline(0, color="0.65", linewidth=0.7, linestyle="--")
        axis.axvline(0, color="0.55", linewidth=0.7, linestyle=":")
        axis.set_xlim(-5, 8)
        axis.set_ylabel("Photometry\n(Z-score)")
        _style_axis(axis)
    top.set_title(f"{title} - example kernel", fontsize=9, pad=5)
    bottom.set_title(f"Mean kernel (n={kernels.shape[0]})", fontsize=9)
    bottom.set_xlabel("Lags (s)")
    top.text(-0.28, 1.25, letter, transform=top.transAxes, fontsize=14)


def _residual_panel(fig, spec, data: dict, *, mode: str, letter: str, title: str):
    grid = spec.subgridspec(2, 1, height_ratios=(2, 1), hspace=0.08)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1], sharex=top)
    sample_count = data["actual"].shape[1]
    time = np.arange(sample_count) / data["sample_rate_hz"] - 5.0
    actual_mean = data["actual"].mean(axis=0)
    actual_sem = _sem(data["actual"])
    residuals = data[f"{mode}_residual"]
    residual_mean = residuals.mean(axis=0)
    residual_sem = _sem(residuals)
    lick_mean = data["lick_rate"].mean(axis=0)
    lick_sem = _sem(data["lick_rate"])

    for axis in (top, bottom):
        axis.axvspan(0, 8, color=COLORS["cue"], zorder=0)
        axis.axvline(8.08, color="0.35", linewidth=0.7, linestyle=":")
        axis.set_xlim(-5, 15)
        _style_axis(axis)
    top.axhline(0, color="0.65", linewidth=0.7, linestyle="--")
    top.plot(time, actual_mean, color=COLORS["actual"], linewidth=1.4, label="Real photometry")
    top.fill_between(time, actual_mean - actual_sem, actual_mean + actual_sem, color="0.4", alpha=0.15)
    top.plot(time, residual_mean, color=COLORS["residual"], linewidth=1.4, label="Residual")
    top.fill_between(
        time,
        residual_mean - residual_sem,
        residual_mean + residual_sem,
        color=COLORS["residual"],
        alpha=0.15,
    )
    top.set_ylabel("Photometry\nZ-score")
    top.legend(frameon=False, fontsize=8, loc="upper left")
    top.tick_params(labelbottom=False)
    top.text(0.5, 1.05, title, transform=top.transAxes, ha="center", fontsize=10)
    top.text(-0.18, 1.2, letter, transform=top.transAxes, fontsize=14)
    top.text(0.37, 0.93, "Cue", color="#2D9CDB", transform=top.transAxes, fontsize=8)
    top.text(0.67, 0.93, "Ensure", color="0.25", transform=top.transAxes, fontsize=8)

    bottom.plot(time, lick_mean, color=COLORS["actual"], linewidth=1.3)
    bottom.fill_between(time, lick_mean - lick_sem, lick_mean + lick_sem, color="0.4", alpha=0.15)
    bottom.set_ylabel("Licking\nlicks/s")
    bottom.set_xlabel("Time from cue onset (s)")


def _save_panel(fig, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--example-mouse", type=int, default=1)
    parser.add_argument(
        "--input-source", choices=("deposited", "raw"), default="deposited"
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument(
        "--photometry-source", choices=("raw465", "dff"), default="raw465"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    )
    if args.input_source == "raw":
        if args.manifest is None or args.processed_root is None:
            parser.error("--input-source=raw requires --manifest and --processed-root")
        trials = load_raw_figure5_trials(
            args.manifest,
            args.processed_root,
            photometry_source=args.photometry_source,
        )
    else:
        trials = load_figure5_trials(args.data_root)
    mouse_ids = np.unique(trials.mouse_ids).astype(int)
    data = _collect_panel_data(trials, mouse_ids, args.example_mouse)

    figure = plt.figure(figsize=(8.5, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(0.9, 1.35), hspace=0.25, wspace=0.25)
    _kernel_panel(figure, grid[0, 0], data, mode="dry", letter="b", title="Dry-period licking")
    _residual_panel(figure, grid[0, 1], data, mode="dry", letter="f", title="Dry-trained model on full trials")
    _kernel_panel(figure, grid[1, 0], data, mode="full", letter="i", title="Full conditional model")
    _residual_panel(figure, grid[1, 1], data, mode="full", letter="j", title="Full conditional model")
    _save_panel(figure, args.output / "figure5_reanalysis_BFIJ")

    for letter, kind, mode, title in (
        ("B", "kernel", "dry", "Dry-period licking"),
        ("F", "residual", "dry", "Dry-trained model on full trials"),
        ("I", "kernel", "full", "Full conditional model"),
        ("J", "residual", "full", "Full conditional model"),
    ):
        size = (3.1, 4.4) if kind == "kernel" else (4.2, 3.6)
        figure = plt.figure(figsize=size, constrained_layout=True)
        grid = figure.add_gridspec(1, 1)
        if kind == "kernel":
            _kernel_panel(figure, grid[0], data, mode=mode, letter=letter.lower(), title=title)
        else:
            _residual_panel(figure, grid[0], data, mode=mode, letter=letter.lower(), title=title)
        _save_panel(figure, args.output / f"figure5_reanalysis_{letter}")


if __name__ == "__main__":
    main()
