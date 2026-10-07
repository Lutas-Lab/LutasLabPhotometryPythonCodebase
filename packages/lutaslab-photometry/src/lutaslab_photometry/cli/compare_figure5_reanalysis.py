"""Compare deposited-data and raw-reprocessed Figure 5 GLM results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

METRICS = (
    "heldout_r2_full",
    "heldout_r2_cue_only",
    "heldout_r2_ingestive",
    "unique_lick_delta_r2",
    "unique_cue_delta_r2",
    "unique_ensure_delta_r2",
    "ingestive_advantage_r2",
    "full_cue_epoch_r2",
    "full_consumption_epoch_r2",
)
LABELS = ("Deposited", "Raw 465", "IRLS dF/F")
COLORS = ("#555555", "#2F73D5", "#E67E22")


def _load(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {
        int(row["mouse_id"]): {key: float(row[key]) for key in METRICS}
        for row in rows
    }


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def _paired_panel(axis, values: np.ndarray, title: str, ylabel: str) -> None:
    for row in values:
        axis.plot(range(3), row, color="0.78", linewidth=0.8, zorder=1)
    for index, color in enumerate(COLORS):
        axis.scatter(
            np.full(values.shape[0], index), values[:, index], color=color, s=17, zorder=2
        )
        mean, sem = _mean_sem(values[:, index])
        axis.errorbar(index, mean, yerr=sem, color="black", marker="_", capsize=3, zorder=3)
    axis.axhline(0, color="0.5", linewidth=0.7, linestyle="--")
    axis.set_xticks(range(3), LABELS, rotation=15)
    axis.set_ylabel(ylabel)
    axis.set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deposited", type=Path)
    parser.add_argument("raw465", type=Path)
    parser.add_argument("dff", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    paths = (args.deposited, args.raw465, args.dff)
    datasets = tuple(_load(path / "mouse_results.csv") for path in paths)
    mouse_ids = sorted(datasets[0])
    if any(sorted(dataset) != mouse_ids for dataset in datasets[1:]):
        raise ValueError("Result directories do not contain identical mouse IDs")

    comparison_rows = []
    for mouse_id in mouse_ids:
        row: dict[str, int | float] = {"mouse_id": mouse_id}
        for label, dataset in zip(("deposited", "raw465", "dff"), datasets, strict=True):
            for metric in METRICS:
                row[f"{label}_{metric}"] = dataset[mouse_id][metric]
        for label, dataset in (("raw465", datasets[1]), ("dff", datasets[2])):
            for metric in METRICS:
                row[f"{label}_minus_deposited_{metric}"] = (
                    dataset[mouse_id][metric] - datasets[0][mouse_id][metric]
                )
        comparison_rows.append(row)
    with (args.output / "mouse_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary: dict[str, object] = {"mouse_count": len(mouse_ids), "metrics": {}}
    for metric in METRICS:
        metric_values = np.array(
            [[dataset[mouse_id][metric] for mouse_id in mouse_ids] for dataset in datasets]
        ).T
        metric_summary: dict[str, object] = {}
        for index, label in enumerate(("deposited", "raw465", "dff")):
            mean, sem = _mean_sem(metric_values[:, index])
            entry: dict[str, float] = {"mean": mean, "sem": sem}
            if index:
                differences = metric_values[:, index] - metric_values[:, 0]
                statistic, pvalue = wilcoxon(differences)
                entry.update(
                    {
                        "mean_difference_from_deposited": float(differences.mean()),
                        "paired_wilcoxon_statistic": float(statistic),
                        "paired_wilcoxon_pvalue_two_sided": float(pvalue),
                        "mousewise_correlation_with_deposited": float(
                            np.corrcoef(metric_values[:, 0], metric_values[:, index])[0, 1]
                        ),
                    }
                )
            metric_summary[label] = entry
        summary["metrics"][metric] = metric_summary
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    def values(metric: str) -> np.ndarray:
        return np.array(
            [[dataset[mouse_id][metric] for dataset in datasets] for mouse_id in mouse_ids]
        )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    _paired_panel(axes[0, 0], values("heldout_r2_full"), "Full model", "Held-out $R^2$")
    _paired_panel(
        axes[0, 1],
        values("ingestive_advantage_r2"),
        "Ingestive model advantage over cue",
        r"$\Delta R^2$",
    )

    terms = ("lick", "cue", "ensure")
    width = 0.24
    for source_index, (label, color) in enumerate(zip(LABELS, COLORS, strict=True)):
        means, sems = [], []
        for term in terms:
            metric_values = values(f"unique_{term}_delta_r2")[:, source_index]
            mean, sem = _mean_sem(metric_values)
            means.append(mean)
            sems.append(sem)
        x = np.arange(3) + (source_index - 1) * width
        axes[1, 0].bar(x, means, width, yerr=sems, color=color, alpha=0.85, label=label)
    axes[1, 0].axhline(0, color="0.5", linewidth=0.7)
    axes[1, 0].set_xticks(range(3), [term.capitalize() for term in terms])
    axes[1, 0].set_ylabel(r"Unique $\Delta R^2$")
    axes[1, 0].set_title("Unique predictor contributions")
    axes[1, 0].legend(frameon=False, fontsize=8)

    epoch_metrics = ("full_cue_epoch_r2", "full_consumption_epoch_r2")
    for source_index, (label, color) in enumerate(zip(LABELS, COLORS, strict=True)):
        means, sems = [], []
        for metric in epoch_metrics:
            mean, sem = _mean_sem(values(metric)[:, source_index])
            means.append(mean)
            sems.append(sem)
        x = np.arange(2) + (source_index - 1) * width
        axes[1, 1].bar(x, means, width, yerr=sems, color=color, alpha=0.85, label=label)
    axes[1, 1].axhline(0, color="0.5", linewidth=0.7)
    axes[1, 1].set_xticks(range(2), ("Cue epoch", "Consumption epoch"))
    axes[1, 1].set_ylabel("Held-out $R^2$")
    axes[1, 1].set_title("Full-model performance by epoch")
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "deposited_vs_raw_comparison.png", dpi=250)
    fig.savefig(args.output / "deposited_vs_raw_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
