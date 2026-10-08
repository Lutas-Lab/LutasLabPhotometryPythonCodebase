import numpy as np
from fluopulse_analysis.glm import (
    build_adaptive_ensure_design,
    fit_adaptive_ensure_glm,
    reconstruct_adaptive_kernels,
)


def test_adaptive_ensure_design_and_grouped_fit():
    rng = np.random.default_rng(3)
    time = np.arange(0.0, 30.0, 0.1)
    designs = []
    outcomes = []
    labels = []
    truth = None
    for index, label in enumerate(("m1", "m2", "m3")):
        licks = np.zeros(time.size)
        ensures = np.zeros(time.size)
        licks[20 + index :: 40] = 1
        ensures[[50, 120, 200]] = 1
        design = build_adaptive_ensure_design(
            time,
            licks,
            ensures,
            lick_kernel_seconds=1.0,
            ensure_kernel_seconds=2.0,
            lick_basis_count=3,
            ensure_basis_count=4,
        )
        if truth is None:
            truth = rng.normal(scale=0.2, size=design.matrix.shape[1])
            truth[0] = 1.0
        designs.append(design)
        outcomes.append(design.matrix @ truth + rng.normal(scale=0.01, size=time.size))
        labels.append(label)
    result = fit_adaptive_ensure_glm(designs, outcomes, labels, [0.0, 0.1, 1.0])
    kernels = reconstruct_adaptive_kernels(designs[0], result.coefficients)
    assert result.fold_scores.shape == (3, 3)
    assert kernels["lick"].shape == designs[0].lick_basis.lag_times.shape
    assert kernels["ensure_first"].shape == designs[0].ensure_basis.lag_times.shape
