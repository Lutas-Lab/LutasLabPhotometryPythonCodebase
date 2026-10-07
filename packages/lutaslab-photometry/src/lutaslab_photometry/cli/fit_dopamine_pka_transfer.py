"""Fit the Figure 4 population dopamine input to a PKA biosensor time course."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lutaslab_photometry.dopamine_pka_model import (
    fit_dopamine_to_pka,
    load_population_dopamine_input,
    predict_pka_from_dopamine,
)


def _load_pka(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data["time"], dtype=float), np.asarray(
                data["pka"], dtype=float
            )
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"time", "pka"}.issubset(rows[0]):
        raise ValueError("PKA CSV must contain time and pka columns")
    return (
        np.asarray([float(row["time"]) for row in rows]),
        np.asarray([float(row["pka"]) for row in rows]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dopamine_input", type=Path)
    parser.add_argument("pka_data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source", choices=("legacy", "raw465", "dff"), default="legacy"
    )
    parser.add_argument(
        "--allow-negative-dopamine",
        action="store_true",
        help="Do not rectify dopamine before convolution",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dopamine_time, dopamine_mean, dopamine_low, dopamine_high = (
        load_population_dopamine_input(args.dopamine_input, args.source)
    )
    pka_time, pka = _load_pka(args.pka_data)
    if pka_time.ndim != 1 or pka.ndim != 1 or pka_time.size != pka.size:
        raise ValueError("PKA time and signal must be matching one-dimensional arrays")
    if np.any(np.diff(pka_time) <= 0):
        raise ValueError("PKA time must be strictly increasing")
    if pka_time[0] < dopamine_time[0] or pka_time[-1] > dopamine_time[-1]:
        raise ValueError("PKA time range extends beyond the dopamine input")

    dopamine = np.interp(pka_time, dopamine_time, dopamine_mean)
    dopamine_ci_low = np.interp(pka_time, dopamine_time, dopamine_low)
    dopamine_ci_high = np.interp(pka_time, dopamine_time, dopamine_high)
    rectify = not args.allow_negative_dopamine
    fit = fit_dopamine_to_pka(pka_time, dopamine, pka, rectify=rectify)
    prediction_low = predict_pka_from_dopamine(
        pka_time,
        dopamine_ci_low,
        rise_tau_seconds=fit.rise_tau_seconds,
        decay_tau_seconds=fit.decay_tau_seconds,
        delay_seconds=fit.delay_seconds,
        gain=fit.gain,
        baseline=fit.baseline,
        rectify=rectify,
    )
    prediction_high = predict_pka_from_dopamine(
        pka_time,
        dopamine_ci_high,
        rise_tau_seconds=fit.rise_tau_seconds,
        decay_tau_seconds=fit.decay_tau_seconds,
        delay_seconds=fit.delay_seconds,
        gain=fit.gain,
        baseline=fit.baseline,
        rectify=rectify,
    )

    rows = [
        {
            "time": time,
            "dopamine": da,
            "pka_observed": observed,
            "pka_predicted": predicted,
            "pka_prediction_from_da_ci_low": low,
            "pka_prediction_from_da_ci_high": high,
        }
        for time, da, observed, predicted, low, high in zip(
            pka_time,
            dopamine,
            pka,
            fit.prediction,
            prediction_low,
            prediction_high,
            strict=True,
        )
    ]
    with (args.output / "predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    parameters = {
        "dopamine_source": args.source,
        "rectified_dopamine": rectify,
        "rise_tau_seconds": fit.rise_tau_seconds,
        "decay_tau_seconds": fit.decay_tau_seconds,
        "delay_seconds": fit.delay_seconds,
        "gain": fit.gain,
        "baseline": fit.baseline,
        "r2": fit.r2,
        "cost": fit.cost,
        "optimizer_success": fit.success,
        "interpretation_note": (
            "The shaded prediction range propagates dopamine cohort uncertainty only; "
            "it is not a confidence interval for transfer-function parameters."
        ),
    }
    (args.output / "fit.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    axes[0].plot(pka_time, dopamine, color="#6C3483")
    axes[0].fill_between(
        pka_time, dopamine_ci_low, dopamine_ci_high, color="#6C3483", alpha=0.2
    )
    axes[0].set_ylabel("Dopamine input\nZ-score")
    axes[0].set_title(f"Population dopamine input: {args.source}")
    axes[1].plot(pka_time, pka, color="black", label="Observed PKA")
    axes[1].plot(pka_time, fit.prediction, color="#E31A1C", label="Predicted PKA")
    axes[1].fill_between(
        pka_time, prediction_low, prediction_high, color="#E31A1C", alpha=0.15
    )
    axes[1].set_ylabel("PKA biosensor")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "dopamine_pka_transfer.png", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
