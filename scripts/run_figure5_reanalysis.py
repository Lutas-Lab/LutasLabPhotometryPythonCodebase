"""Run the trial-blocked smooth-basis reanalysis of paper Figure 5."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import wilcoxon

from lutaslab_core.glm import r2_score
from lutaslab_core.provenance import utc_now, write_run_record
from src.figure5_reanalysis import (
    build_figure5_design,
    fit_nested_blocked_ridge,
    load_figure5_trials,
    model_columns,
    reconstruct_figure5_kernels,
)
from src.published_reanalysis_config import (
    load_published_reanalysis_config,
    resolve_dataset_paths,
)


MODEL_TERMS = {
    "full": ("lick", "cue", "ensure"),
    "without_lick": ("cue", "ensure"),
    "without_cue": ("lick", "ensure"),
    "without_ensure": ("lick", "cue"),
    "cue_only": ("cue",),
    "ingestive": ("lick", "ensure"),
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _epoch_r2(
    response: np.ndarray,
    prediction: np.ndarray,
    trial_count: int,
    epoch: slice,
) -> float:
    response_trials = response.reshape(trial_count, -1)[:, epoch].reshape(-1)
    prediction_trials = prediction.reshape(trial_count, -1)[:, epoch].reshape(-1)
    return r2_score(response_trials, prediction_trials)


def _published_r2(data_paths: dict[str, Path]) -> dict[int, dict[str, float]]:
    score_paths = {
        "published_dry_training_r2": data_paths["figure5_dry"],
        "published_total_training_r2": data_paths["figure5_total"],
    }
    output: dict[int, dict[str, float]] = {}
    for name, path in score_paths.items():
        model = loadmat(path, simplify_cells=True)["FilteredGLMData"]
        mouse_ids = np.asarray(model["mouse_ids"], dtype=int).reshape(-1)
        scores = np.asarray(model["r2_mouse"], dtype=float).reshape(-1)
        for mouse_id, score in zip(mouse_ids, scores, strict=True):
            output.setdefault(int(mouse_id), {})[name] = float(score)
    return output


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sem = 0.0 if values.size < 2 else float(np.std(values, ddof=1) / np.sqrt(values.size))
    return float(np.mean(values)), sem


def main() -> None:
    started_at = utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Path to Depository Data")
    parser.add_argument("output", type=Path, help="Directory for tables and figures")
    parser.add_argument("--mice", type=int, nargs="*", help="Mouse IDs; default is all")
    parser.add_argument("--basis-counts", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--primary-basis-count", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config/published_reanalysis.json",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_published_reanalysis_config(args.config)
    data_paths = resolve_dataset_paths(args.data_root, config["datasets"])
    trials = load_figure5_trials(args.data_root, data_paths["figure5_total"])
    available_mice = np.unique(trials.mouse_ids).astype(int)
    mouse_ids = available_mice if not args.mice else np.asarray(args.mice, dtype=int)
    missing = np.setdiff1d(mouse_ids, available_mice)
    if missing.size:
        raise ValueError(f"unknown mouse IDs: {missing.tolist()}")
    alphas = np.logspace(-3, 5, 9)
    published = _published_r2(data_paths)
    primary_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    mouse_kernels: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for mouse_id in mouse_ids:
        designs = {}
        for basis_count in sorted(set([*args.basis_counts, args.primary_basis_count])):
            design = build_figure5_design(
                trials,
                int(mouse_id),
                lick_basis_count=basis_count,
                block_count=args.folds,
            )
            designs[basis_count] = design
            columns = model_columns(design, MODEL_TERMS["full"])
            fit = fit_nested_blocked_ridge(
                design.matrix[:, columns],
                design.response,
                design.block_groups,
                alphas,
                penalty_weights=design.penalty_weights[columns],
            )
            sensitivity_rows.append(
                {
                    "mouse_id": int(mouse_id),
                    "lick_basis_count": basis_count,
                    "heldout_r2": fit.heldout_r2,
                }
            )

        design = designs[args.primary_basis_count]
        fits = {}
        for name, terms in MODEL_TERMS.items():
            columns = model_columns(design, terms)
            fits[name] = fit_nested_blocked_ridge(
                design.matrix[:, columns],
                design.response,
                design.block_groups,
                alphas,
                penalty_weights=design.penalty_weights[columns],
            )
        full_columns = model_columns(design, MODEL_TERMS["full"])
        full_coefficients = np.zeros(design.matrix.shape[1])
        full_coefficients[full_columns] = fits["full"].final_fit.coefficients
        mouse_kernels[int(mouse_id)] = reconstruct_figure5_kernels(
            design, full_coefficients
        )
        trial_count = design.trial_numbers.size
        row = {
            "mouse_id": int(mouse_id),
            "trial_count": int(trial_count),
            "heldout_r2_full": fits["full"].heldout_r2,
            "heldout_r2_cue_only": fits["cue_only"].heldout_r2,
            "heldout_r2_ingestive": fits["ingestive"].heldout_r2,
            "unique_lick_delta_r2": (
                fits["full"].heldout_r2 - fits["without_lick"].heldout_r2
            ),
            "unique_cue_delta_r2": (
                fits["full"].heldout_r2 - fits["without_cue"].heldout_r2
            ),
            "unique_ensure_delta_r2": (
                fits["full"].heldout_r2 - fits["without_ensure"].heldout_r2
            ),
            "ingestive_advantage_r2": (
                fits["ingestive"].heldout_r2 - fits["cue_only"].heldout_r2
            ),
            "full_cue_epoch_r2": _epoch_r2(
                design.response, fits["full"].prediction, trial_count, slice(250, 654)
            ),
            "full_consumption_epoch_r2": _epoch_r2(
                design.response, fits["full"].prediction, trial_count, slice(654, None)
            ),
            "final_alpha": fits["full"].final_fit.best_alpha,
            "median_outer_alpha": float(np.median(fits["full"].outer_alphas)),
            **published[int(mouse_id)],
        }
        primary_rows.append(row)
        print(
            f"mouse {int(mouse_id):2d}: held-out R2={fits['full'].heldout_r2:.3f}, "
            f"unique lick={row['unique_lick_delta_r2']:.3f}, "
            f"unique cue={row['unique_cue_delta_r2']:.3f}, "
            f"unique Ensure={row['unique_ensure_delta_r2']:.3f}"
        )

    _write_csv(args.output / "mouse_results.csv", primary_rows)
    _write_csv(args.output / "basis_sensitivity.csv", sensitivity_rows)

    metrics = [
        "heldout_r2_full",
        "heldout_r2_cue_only",
        "heldout_r2_ingestive",
        "unique_lick_delta_r2",
        "unique_cue_delta_r2",
        "unique_ensure_delta_r2",
        "ingestive_advantage_r2",
        "full_cue_epoch_r2",
        "full_consumption_epoch_r2",
    ]
    summary = {
        "mouse_count": len(primary_rows),
        "cross_validation": "nested contiguous trial-blocked",
        "outer_folds": args.folds,
        "primary_lick_basis_count": args.primary_basis_count,
        "alpha_grid": alphas.tolist(),
        "metrics": {
            metric: dict(
                zip(
                    ("mean", "sem"),
                    _mean_sem(np.array([row[metric] for row in primary_rows])),
                    strict=True,
                )
            )
            for metric in metrics
        },
        "paired_wilcoxon_ingestive_vs_cue": {},
        "wilcoxon_unique_terms_vs_zero": {},
        "notes": [
            "Predictions are out-of-fold; alpha is selected using only outer-training trials.",
            "Each temporal convolution resets at the start of a trial.",
            "Published raw-lag R2 values are retained only as in-sample sensitivity references.",
            "Unique-term Wilcoxon p-values are exploratory and unadjusted for three comparisons.",
        ],
    }
    ingestive = np.array([row["heldout_r2_ingestive"] for row in primary_rows])
    cue_only = np.array([row["heldout_r2_cue_only"] for row in primary_rows])
    statistic, pvalue = wilcoxon(ingestive, cue_only)
    summary["paired_wilcoxon_ingestive_vs_cue"] = {
        "statistic": float(statistic),
        "pvalue_two_sided": float(pvalue),
    }
    for term in ("lick", "cue", "ensure"):
        values = np.array([row[f"unique_{term}_delta_r2"] for row in primary_rows])
        statistic, pvalue = wilcoxon(values)
        summary["wilcoxon_unique_terms_vs_zero"][term] = {
            "statistic": float(statistic),
            "pvalue_two_sided": float(pvalue),
            "positive_mice": int(np.sum(values > 0)),
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    model_names = ["cue_only", "ingestive", "full"]
    model_values = np.array(
        [[row[f"heldout_r2_{name}"] for name in model_names] for row in primary_rows]
    )
    for values in model_values:
        axes[0].plot(range(3), values, color="0.75", linewidth=0.8, alpha=0.7)
        axes[0].scatter(range(3), values, color="0.25", s=12)
    axes[0].set_xticks(range(3), ["Cue", "Lick + Ensure", "Full"])
    axes[0].set_ylabel("Nested held-out $R^2$")
    axes[0].set_title("Model comparison")
    axes[0].axhline(0, color="0.2", linewidth=0.7)

    for basis_count in args.basis_counts:
        values = np.array(
            [
                row["heldout_r2"]
                for row in sensitivity_rows
                if row["lick_basis_count"] == basis_count
            ]
        )
        mean, sem = _mean_sem(values)
        axes[1].errorbar(basis_count, mean, yerr=sem, fmt="o", color="tab:blue")
    axes[1].set_xlabel("Lick basis functions")
    axes[1].set_ylabel("Nested held-out $R^2$")
    axes[1].set_title("Basis-size sensitivity")

    for name, color in [("lick", "tab:green"), ("cue", "tab:blue"), ("ensure", "tab:orange")]:
        time = mouse_kernels[int(mouse_ids[0])][name][0]
        kernels = np.vstack([mouse_kernels[int(mouse)][name][1] for mouse in mouse_ids])
        mean = np.mean(kernels, axis=0)
        sem = (
            np.zeros(kernels.shape[1])
            if kernels.shape[0] < 2
            else np.std(kernels, axis=0, ddof=1) / np.sqrt(kernels.shape[0])
        )
        axes[2].plot(time, mean, label=name.capitalize(), color=color)
        axes[2].fill_between(time, mean - sem, mean + sem, color=color, alpha=0.2)
    axes[2].axhline(0, color="0.2", linewidth=0.7)
    axes[2].set_xlabel("Lag (s)")
    axes[2].set_ylabel("Kernel weight")
    axes[2].set_title("Full-model kernels")
    axes[2].legend(frameon=False)
    fig.savefig(args.output / "reanalysis_summary.png", dpi=200)
    plt.close(fig)
    write_run_record(
        args.output / "run_metadata.json",
        workflow="figure5_reanalysis",
        parameters=vars(args),
        inputs=[data_paths["figure5_dry"], data_paths["figure5_total"], args.config],
        outputs=[
            args.output / "mouse_results.csv",
            args.output / "basis_sensitivity.csv",
            args.output / "summary.json",
            args.output / "reanalysis_summary.png",
        ],
        repository_root=Path(__file__).resolve().parents[1],
        started_at=started_at,
    )


if __name__ == "__main__":
    main()
