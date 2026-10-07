"""Test whether lick-bout structure explains Figure 5 photometry."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lutaslab_core.provenance import utc_now, write_run_record
from scipy.stats import wilcoxon

from lutaslab_photometry.figure5_bout_analysis import bout_model_columns, build_figure5_bout_design
from lutaslab_photometry.figure5_reanalysis import fit_nested_blocked_ridge, load_figure5_trials
from lutaslab_photometry.published_reanalysis_config import (
    default_published_reanalysis_config_path,
    load_published_reanalysis_config,
    resolve_dataset_paths,
)

BOUT_TERMS = ("bout_onset", "bout_size", "bout_occupancy", "bout_rate")
MODEL_TERMS = {
    "cue": ("cue",),
    "lick": ("cue", "lick"),
    "bout": ("cue", *BOUT_TERMS),
    "lick_bout": ("cue", "lick", *BOUT_TERMS),
    "lick_ensure": ("cue", "lick", "ensure"),
    "bout_ensure": ("cue", *BOUT_TERMS, "ensure"),
    "full": ("cue", "lick", *BOUT_TERMS, "ensure"),
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(values.size))


def main() -> None:
    started_at = utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bout-gaps", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--primary-bout-gap", type=float, default=1.0)
    parser.add_argument("--min-licks", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_published_reanalysis_config_path(),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = load_published_reanalysis_config(args.config)
    data_paths = resolve_dataset_paths(args.data_root, config["datasets"])
    trials = load_figure5_trials(args.data_root, data_paths["figure5_total"])
    mouse_ids = np.unique(trials.mouse_ids).astype(int)
    alphas = np.logspace(-3, 5, 9)
    rows: list[dict[str, object]] = []
    for gap in args.bout_gaps:
        for mouse_id in mouse_ids:
            design = build_figure5_bout_design(
                trials,
                int(mouse_id),
                max_interlick_gap_seconds=gap,
                min_licks=args.min_licks,
                block_count=args.folds,
            )
            fits = {}
            for name, terms in MODEL_TERMS.items():
                columns = bout_model_columns(design, terms)
                fits[name] = fit_nested_blocked_ridge(
                    design.matrix[:, columns],
                    design.response,
                    design.block_groups,
                    alphas,
                    penalty_weights=design.penalty_weights[columns],
                )
            row = {
                "mouse_id": int(mouse_id),
                "bout_gap_seconds": gap,
                "trial_count": int(design.trial_numbers.size),
                "bout_count": int(np.sum(design.bout_counts)),
                **{f"heldout_r2_{name}": fit.heldout_r2 for name, fit in fits.items()},
                "bout_beyond_lick_delta_r2": (
                    fits["lick_bout"].heldout_r2 - fits["lick"].heldout_r2
                ),
                "ensure_beyond_lick_delta_r2": (
                    fits["lick_ensure"].heldout_r2 - fits["lick"].heldout_r2
                ),
                "bout_beyond_lick_ensure_delta_r2": (
                    fits["full"].heldout_r2 - fits["lick_ensure"].heldout_r2
                ),
                "ensure_beyond_lick_bout_delta_r2": (
                    fits["full"].heldout_r2 - fits["lick_bout"].heldout_r2
                ),
                "lick_beyond_bout_ensure_delta_r2": (
                    fits["full"].heldout_r2 - fits["bout_ensure"].heldout_r2
                ),
            }
            rows.append(row)
            print(
                f"gap={gap:g}s mouse={int(mouse_id):2d} "
                f"lick={fits['lick'].heldout_r2:.3f} "
                f"bout={fits['bout'].heldout_r2:.3f} "
                f"full={fits['full'].heldout_r2:.3f}"
            )

    _write_csv(args.output / "mouse_results.csv", rows)
    primary = [row for row in rows if row["bout_gap_seconds"] == args.primary_bout_gap]
    model_metrics = [f"heldout_r2_{name}" for name in MODEL_TERMS]
    delta_metrics = [
        "bout_beyond_lick_delta_r2",
        "ensure_beyond_lick_delta_r2",
        "bout_beyond_lick_ensure_delta_r2",
        "ensure_beyond_lick_bout_delta_r2",
        "lick_beyond_bout_ensure_delta_r2",
    ]
    summary = {
        "mouse_count": len(mouse_ids),
        "primary_bout_gap_seconds": args.primary_bout_gap,
        "minimum_licks_per_bout": args.min_licks,
        "models_include_cue_nuisance_term": True,
        "metrics": {},
        "wilcoxon_deltas_vs_zero": {},
        "gap_sensitivity": {},
        "notes": [
            "All scores are nested out-of-fold R2 from contiguous trial blocks.",
            "Bout terms include onset timing, bout-size modulation, occupancy, "
            "and within-bout rate.",
            "Wilcoxon p-values are exploratory and unadjusted for multiple comparisons.",
        ],
    }
    for metric in (*model_metrics, *delta_metrics):
        values = np.array([row[metric] for row in primary], dtype=float)
        mean, sem = _mean_sem(values)
        summary["metrics"][metric] = {"mean": mean, "sem": sem}
    for metric in delta_metrics:
        values = np.array([row[metric] for row in primary], dtype=float)
        statistic, pvalue = wilcoxon(values)
        summary["wilcoxon_deltas_vs_zero"][metric] = {
            "statistic": float(statistic),
            "pvalue_two_sided": float(pvalue),
            "positive_mice": int(np.sum(values > 0)),
        }
    for gap in args.bout_gaps:
        selected = [row for row in rows if row["bout_gap_seconds"] == gap]
        summary["gap_sensitivity"][str(gap)] = {
            metric: dict(
                zip(
                    ("mean", "sem"),
                    _mean_sem(np.array([row[metric] for row in selected])),
                    strict=True,
                )
            )
            for metric in ("bout_beyond_lick_delta_r2", "bout_beyond_lick_ensure_delta_r2")
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    shown = ["lick", "bout", "lick_bout", "lick_ensure", "full"]
    labels = ["Lick", "Bout", "Lick + bout", "Lick + Ensure", "All"]
    values = np.array([[row[f"heldout_r2_{name}"] for name in shown] for row in primary])
    for mouse_values in values:
        axes[0].plot(range(len(shown)), mouse_values, color="0.8", linewidth=0.7)
        axes[0].scatter(range(len(shown)), mouse_values, color="0.25", s=10)
    axes[0].set_xticks(range(len(shown)), labels, rotation=35, ha="right")
    axes[0].set_ylabel("Nested held-out $R^2$")
    axes[0].set_title("Predictive models")
    axes[0].axhline(0, color="0.5", linewidth=0.7)

    delta_names = [
        "bout_beyond_lick_delta_r2",
        "ensure_beyond_lick_delta_r2",
        "bout_beyond_lick_ensure_delta_r2",
        "ensure_beyond_lick_bout_delta_r2",
    ]
    delta_labels = ["Bout | lick", "Ensure | lick", "Bout | lick+Ensure", "Ensure | lick+bout"]
    for index, metric in enumerate(delta_names):
        metric_values = np.array([row[metric] for row in primary])
        axes[1].scatter(np.full(metric_values.size, index), metric_values, color="0.3", s=12)
        mean, sem = _mean_sem(metric_values)
        axes[1].errorbar(index, mean, yerr=sem, fmt="o", color="#2F73D5", capsize=3)
    axes[1].set_xticks(range(len(delta_names)), delta_labels, rotation=35, ha="right")
    axes[1].set_ylabel(r"Unique held-out $\Delta R^2$")
    axes[1].set_title("Added predictive information")
    axes[1].axhline(0, color="0.5", linewidth=0.7)

    for metric, label, color in (
        ("bout_beyond_lick_delta_r2", "Bout beyond lick", "#2F73D5"),
        ("bout_beyond_lick_ensure_delta_r2", "Bout beyond lick + Ensure", "#EC168C"),
    ):
        means, sems = [], []
        for gap in args.bout_gaps:
            selected = [row[metric] for row in rows if row["bout_gap_seconds"] == gap]
            mean, sem = _mean_sem(np.asarray(selected))
            means.append(mean)
            sems.append(sem)
        axes[2].errorbar(args.bout_gaps, means, yerr=sems, marker="o", label=label, color=color)
    axes[2].set_xlabel("Maximum interlick gap (s)")
    axes[2].set_ylabel(r"Unique held-out $\Delta R^2$")
    axes[2].set_title("Bout-definition sensitivity")
    axes[2].axhline(0, color="0.5", linewidth=0.7)
    axes[2].legend(frameon=False, fontsize=7)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "bout_structure_summary.png", dpi=300)
    fig.savefig(args.output / "bout_structure_summary.svg")
    plt.close(fig)
    write_run_record(
        args.output / "run_metadata.json",
        workflow="figure5_bout_analysis",
        parameters=vars(args),
        inputs=[data_paths["figure5_total"], args.config],
        outputs=[
            args.output / "mouse_results.csv",
            args.output / "summary.json",
            args.output / "bout_structure_summary.png",
            args.output / "bout_structure_summary.svg",
        ],
        repository_root=Path(__file__).resolve().parents[2],
        started_at=started_at,
    )


if __name__ == "__main__":
    main()
