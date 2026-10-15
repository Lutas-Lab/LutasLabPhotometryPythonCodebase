"""Lightweight installation check for lab users.

This is intentionally smaller than the maintainer test suite. It confirms that
the photometry package and shared GLM code import and execute, with an optional
check for the external R/fastFMM backend.
"""

from __future__ import annotations

import argparse

import numpy as np
from lutaslab_core import fit_ridge, raised_cosine_basis

import lutaslab_photometry
from lutaslab_photometry.functional_mixed import flmm_runtime_status


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flmm",
        action="store_true",
        help="Also require Rscript and the optional fastFMM R package.",
    )
    args = parser.parse_args(argv)
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
    if not hasattr(lutaslab_photometry, "__path__"):
        raise RuntimeError("Conventional photometry package did not import correctly")

    print("Installation check passed: conventional photometry, lutaslab-core, and GLM.")
    if args.flmm:
        status = flmm_runtime_status()
        if not status["ready"]:
            raise RuntimeError(
                "FLMM backend unavailable: install R, add Rscript to PATH, and "
                "install the R package fastFMM."
            )
        print(
            "FLMM backend passed: "
            f"Rscript={status['rscript']}, fastFMM={status['fastfmm_version']}."
        )


if __name__ == "__main__":
    main()
