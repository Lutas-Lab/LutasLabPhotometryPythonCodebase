"""Lightweight installation check for lab users.

This is intentionally smaller than the maintainer test suite. It confirms that
the conventional photometry package and shared GLM code import and execute.
"""

from __future__ import annotations

import numpy as np

import src
from lutaslab_core import fit_ridge, raised_cosine_basis


def main() -> None:
    basis = raised_cosine_basis((0.0, 1.0), count=4, dt=0.05)
    design = np.column_stack([np.ones(40), np.linspace(-1.0, 1.0, 40)])
    response = 0.25 + 0.8 * design[:, 1]
    coefficients = fit_ridge(
        design,
        response,
        alpha=0.01,
        penalize=np.array([0.0, 1.0]),
    )

    if basis.values.shape != (21, 4):
        raise RuntimeError("Raised-cosine GLM basis produced an unexpected shape")
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError("Synthetic ridge fit produced non-finite coefficients")
    if not hasattr(src, "__path__"):
        raise RuntimeError("Conventional photometry package did not import correctly")

    print("Installation check passed: conventional photometry, lutaslab-core, and GLM.")


if __name__ == "__main__":
    main()
