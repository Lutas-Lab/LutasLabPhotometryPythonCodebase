"""Test tastant-specific delivery responses after accounting for licking and bouts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lutaslab_core.glm import r2_score
from lutaslab_core.provenance import utc_now, write_run_record
from scipy.stats import wilcoxon

from lutaslab_photometry.figure5_reanalysis import fit_nested_blocked_ridge
from lutaslab_photometry.multitastant_analysis import (
    build_paired_tastant_design,
    build_tastant_design,
    condition_delivery_kernels,
    load_tastant_trials,
    paired_model_columns,
)
from lutaslab_photometry.published_reanalysis_config import (
    default_published_reanalysis_config_path,
    load_published_reanalysis_config,
    resolve_dataset_paths,
)

PAIR_SPECS = {
    "Ensure_vs_Sucralose": ("Ensure", "Sucralose"),
    "Ensure_vs_Sucrose": ("Ensure", "Sucrose"),
    "Sucrose_vs_Sucralose": ("Sucrose", "Sucralose"),
    "Ensure_vs_Quinine": ("EnsureQuinineCohort", "QuinineEnsureMatched"),
    "Water_vs_Quinine": ("Water", "QuinineWaterMatched"),
}

DISPLAY_NAMES = {
    "EnsureQuinineCohort": "Ensure",
    "QuinineEnsureMatched": "Quinine",
    "QuinineWaterMatched": "Quinine",
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(values.size))


def _median_iqr(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def main() -> None:
    started_at = utc_now()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_published_reanalysis_config_path(),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_published_reanalysis_config(args.config)
    data_paths = resolve_dataset_paths(args.data_root, config["datasets"])
    tastant_names = {name for names in PAIR_SPECS.values() for name in names}
    datasets = {
        name: load_tastant_trials(data_paths[name]) for name in tastant_names
    }
    alphas = np.logspace(-3, 5, 9)
    result_rows: list[dict[str, object]] = []
    kernel_rows = []

    for pair_name, (first_name, second_name) in PAIR_SPECS.items():
        first_trials = datasets[first_name]
        second_trials = datasets[second_name]
        common = sorted(set(first_trials.mouse_names) & set(second_trials.mouse_names))
        for mouse in common:
            first_count = int(np.sum(first_trials.mouse_names == mouse))
            second_count = int(np.sum(second_trials.mouse_names == mouse))
            if min(first_count, second_count) < 3:
                continue
            folds = min(5, first_count, second_count)
            first = build_tastant_design(first_trials, mouse, block_count=folds)
            second = build_tastant_design(second_trials, mouse, block_count=folds)
            paired = build_paired_tastant_design(first, second)
            model_terms = {
                "behavior": ("intercept", "condition", "behavior", "behavior_interaction"),
                "shared_delivery": (
                    "intercept",
                    "condition",
                    "behavior",
                    "behavior_interaction",
                    "delivery",
                ),
                "specific_delivery": (
                    "intercept",
                    "condition",
                    "behavior",
                    "behavior_interaction",
                    "delivery",
                    "delivery_interaction",
                ),
            }
            fits = {}
            for model_name, terms in model_terms.items():
                columns = paired_model_columns(paired, terms)
                fits[model_name] = fit_nested_blocked_ridge(
                    paired.matrix[:, columns],
                    paired.response,
                    paired.groups,
                    alphas,
                    penalty_weights=paired.penalty_weights[columns],
                )
            specific_columns = paired_model_columns(paired, model_terms["specific_delivery"])
            coefficients = np.zeros(paired.matrix.shape[1])
            coefficients[specific_columns] = fits["specific_delivery"].final_fit.coefficients
            first_kernel, second_kernel, difference = condition_delivery_kernels(
                paired, coefficients
            )

            first_fit = fit_nested_blocked_ridge(
                first.matrix,
                first.response,
                first.groups,
                alphas,
                penalty_weights=first.penalty_weights,
            )
            second_fit = fit_nested_blocked_ridge(
                second.matrix,
                second.response,
                second.groups,
                alphas,
                penalty_weights=second.penalty_weights,
            )
            first_to_second = r2_score(
                second.response, second.matrix @ first_fit.final_fit.coefficients
            )
            second_to_first = r2_score(
                first.response, first.matrix @ second_fit.final_fit.coefficients
            )
            result_rows.append(
                {
                    "pair": pair_name,
                    "first_tastant": first_name,
                    "second_tastant": second_name,
                    "mouse": mouse,
                    "first_trial_count": first.trial_count,
                    "second_trial_count": second.trial_count,
                    "heldout_r2_behavior": fits["behavior"].heldout_r2,
                    "heldout_r2_shared_delivery": fits["shared_delivery"].heldout_r2,
                    "heldout_r2_specific_delivery": fits["specific_delivery"].heldout_r2,
                    "delivery_delta_r2": (
                        fits["shared_delivery"].heldout_r2 - fits["behavior"].heldout_r2
                    ),
                    "tastant_identity_delta_r2": (
                        fits["specific_delivery"].heldout_r2
                        - fits["shared_delivery"].heldout_r2
                    ),
                    "first_within_r2": first_fit.heldout_r2,
                    "second_within_r2": second_fit.heldout_r2,
                    "first_to_second_r2": first_to_second,
                    "second_to_first_r2": second_to_first,
                    "delivery_difference_auc_0_7s": float(
                        np.trapezoid(
                            difference[paired.delivery_lag_seconds <= 7],
                            paired.delivery_lag_seconds[paired.delivery_lag_seconds <= 7],
                        )
                    ),
                    "unstable_heldout_fit": bool(
                        min(
                            fits["shared_delivery"].heldout_r2,
                            fits["specific_delivery"].heldout_r2,
                        )
                        < -1
                    ),
                }
            )
            for time, first_value, second_value, difference_value in zip(
                paired.delivery_lag_seconds,
                first_kernel,
                second_kernel,
                difference,
                strict=True,
            ):
                kernel_rows.append(
                    {
                        "pair": pair_name,
                        "mouse": mouse,
                        "time_seconds": float(time),
                        "first_kernel": float(first_value),
                        "second_kernel": float(second_value),
                        "difference_kernel": float(difference_value),
                    }
                )
            print(
                f"{pair_name} {mouse}: identity delta R2="
                f"{result_rows[-1]['tastant_identity_delta_r2']:.3f}"
            )

    _write_csv(args.output / "mouse_results.csv", result_rows)
    _write_csv(args.output / "delivery_kernels.csv", kernel_rows)
    summary = {
        "pairs": {},
        "notes": [
            "Delivery kernels are conditional on condition-specific lick and bout predictors.",
            "Identity delta R2 is the held-out gain from a tastant-specific "
            "rather than shared delivery kernel.",
            "Wilcoxon p-values are exploratory and unadjusted across tastant pairs.",
        ],
    }
    for pair_name in PAIR_SPECS:
        selected = [row for row in result_rows if row["pair"] == pair_name]
        if not selected:
            continue
        identity = np.array([row["tastant_identity_delta_r2"] for row in selected])
        auc = np.array([row["delivery_difference_auc_0_7s"] for row in selected])
        identity_test = wilcoxon(identity)
        auc_test = wilcoxon(auc)
        summary["pairs"][pair_name] = {
            "mouse_count": len(selected),
            "identity_delta_r2": dict(zip(("mean", "sem"), _mean_sem(identity), strict=True)),
            "identity_delta_r2_robust": _median_iqr(identity),
            "identity_wilcoxon_p_two_sided": float(identity_test.pvalue),
            "positive_identity_delta_mice": int(np.sum(identity > 0)),
            "delivery_difference_auc_0_7s": dict(zip(("mean", "sem"), _mean_sem(auc), strict=True)),
            "delivery_difference_auc_0_7s_robust": _median_iqr(auc),
            "auc_wilcoxon_p_two_sided": float(auc_test.pvalue),
            "first_to_second_r2": dict(
                zip(
                    ("mean", "sem"),
                    _mean_sem(
                        np.array([row["first_to_second_r2"] for row in selected])
                    ),
                    strict=True,
                )
            ),
            "second_to_first_r2": dict(
                zip(
                    ("mean", "sem"),
                    _mean_sem(
                        np.array([row["second_to_first_r2"] for row in selected])
                    ),
                    strict=True,
                )
            ),
            "unstable_heldout_fit_count": int(
                np.sum([row["unstable_heldout_fit"] for row in selected])
            ),
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), constrained_layout=True)
    axes = axes.ravel()
    colors = ("#2F73D5", "#EC168C")
    for axis, (pair_name, _) in zip(axes, PAIR_SPECS.items(), strict=False):
        pair_kernels = [row for row in kernel_rows if row["pair"] == pair_name]
        mice = sorted({row["mouse"] for row in pair_kernels})
        time = np.array(
            [row["time_seconds"] for row in pair_kernels if row["mouse"] == mice[0]]
        )
        first_values = np.vstack(
            [
                [
                    row["first_kernel"]
                    for row in pair_kernels
                    if row["mouse"] == mouse
                ]
                for mouse in mice
            ]
        )
        second_values = np.vstack(
            [
                [
                    row["second_kernel"]
                    for row in pair_kernels
                    if row["mouse"] == mouse
                ]
                for mouse in mice
            ]
        )
        first_label, second_label = PAIR_SPECS[pair_name]
        for values, label, color in zip(
            (first_values, second_values),
            (first_label, second_label),
            colors,
            strict=True,
        ):
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
            axis.plot(time, mean, color=color, label=DISPLAY_NAMES.get(label, label))
            axis.fill_between(time, mean - sem, mean + sem, color=color, alpha=0.18)
        axis.axhline(0, color="0.6", linestyle="--", linewidth=0.7)
        axis.set_title(pair_name.replace("_", " "))
        axis.set_xlabel("Time from delivery (s)")
        axis.set_ylabel("Conditional kernel")
        axis.legend(frameon=False, fontsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    identity_axis = axes[-1]
    pair_names = list(summary["pairs"])
    for index, pair_name in enumerate(pair_names):
        values = np.array(
            [
                row["tastant_identity_delta_r2"]
                for row in result_rows
                if row["pair"] == pair_name
            ]
        )
        identity_axis.scatter(np.full(values.size, index), values, color="0.3", s=12)
        robust = _median_iqr(values)
        identity_axis.errorbar(
            index,
            robust["median"],
            yerr=np.array(
                [
                    [robust["median"] - robust["q25"]],
                    [robust["q75"] - robust["median"]],
                ]
            ),
            fmt="o",
            color="#2F73D5",
            capsize=3,
        )
    identity_axis.axhline(0, color="0.6", linewidth=0.7)
    identity_axis.set_xticks(
        range(len(pair_names)),
        [name.replace("_vs_", "\nvs\n") for name in pair_names],
        fontsize=7,
    )
    identity_axis.set_ylabel(r"Tastant-specific held-out $\Delta R^2$")
    identity_axis.set_title("Does identity improve prediction?")
    identity_axis.set_yscale("symlog", linthresh=0.02)
    identity_axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "multitastant_delivery_kernels.png", dpi=300)
    fig.savefig(args.output / "multitastant_delivery_kernels.svg")
    plt.close(fig)
    write_run_record(
        args.output / "run_metadata.json",
        workflow="multitastant_analysis",
        parameters=vars(args),
        inputs=[*(data_paths[name] for name in sorted(tastant_names)), args.config],
        outputs=[
            args.output / "mouse_results.csv",
            args.output / "delivery_kernels.csv",
            args.output / "summary.json",
            args.output / "multitastant_delivery_kernels.png",
            args.output / "multitastant_delivery_kernels.svg",
        ],
        repository_root=Path(__file__).resolve().parents[5],
        started_at=started_at,
    )


if __name__ == "__main__":
    main()
