"""Minimal synthetic example of the reusable event-GLM and transfer APIs."""

from __future__ import annotations

import numpy as np

from lutaslab_core.glm import (
    build_trialwise_basis_design,
    fit_grouped_ridge_cv,
    raised_cosine_basis,
)
from lutaslab_core.transfer import apply_gamma_transfer, fit_nonnegative_gamma_transfer


def main() -> None:
    rng = np.random.default_rng(7)
    dt = 0.05
    time = np.arange(-2.0, 12.0, dt)
    reward = np.zeros((12, time.size))
    lick = np.zeros_like(reward)
    reward[:, np.argmin(np.abs(time))] = 1.0
    for row in range(lick.shape[0]):
        lick[row, np.flatnonzero((time >= 0.5) & (time < 4.0))[::10]] = 1.0

    bases = {
        "reward": raised_cosine_basis((0.0, 8.0), count=8, dt=dt),
        "lick": raised_cosine_basis((0.0, 3.0), count=5, dt=dt),
    }
    design = build_trialwise_basis_design(
        {"reward": reward, "lick": lick},
        bases,
        dt,
    )
    coefficients = np.zeros(design.matrix.shape[1])
    coefficients[design.column_slices["reward"]] = np.linspace(0.3, 1.2, 8)
    coefficients[design.column_slices["lick"]] = np.linspace(0.2, 0.0, 5)
    response = design.matrix @ coefficients + rng.normal(
        scale=0.05, size=design.matrix.shape[0]
    )
    groups = np.repeat(np.arange(design.trial_count), design.samples_per_trial)
    ridge = fit_grouped_ridge_cv(
        design.matrix,
        response,
        groups,
        alphas=np.logspace(-3, 2, 6),
        penalize=np.r_[0.0, np.ones(design.matrix.shape[1] - 1)],
    )

    dopamine = response.reshape(design.trial_count, -1).mean(axis=0)
    pka = 2.0 * apply_gamma_transfer(dopamine, dt, 5.0)
    transfer = fit_nonnegative_gamma_transfer(
        time,
        dopamine,
        pka,
        tau_candidates=[2.0, 5.0, 10.0, 20.0],
    )
    print(f"GLM alpha: {ridge.best_alpha:g}")
    print(f"Transfer tau: {transfer.tau_seconds:g} s; R2={transfer.r2:.3f}")


if __name__ == "__main__":
    main()
