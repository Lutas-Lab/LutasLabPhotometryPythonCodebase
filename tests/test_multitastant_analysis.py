import numpy as np

from src.multitastant_analysis import TastantDesign, build_paired_tastant_design


def _design(offset):
    matrix = np.column_stack([np.ones(12), np.arange(12), np.arange(12) ** 2, np.ones(12) * offset])
    return TastantDesign(
        matrix=matrix,
        response=np.arange(12, dtype=float),
        groups=np.repeat(np.arange(3), 4),
        slices={"intercept": slice(0, 1), "lick": slice(1, 2), "bout": slice(2, 3), "delivery": slice(3, 4)},
        penalty_weights=np.array([0.0, 1.0, 1.0, 1.0]),
        delivery_basis_values=np.ones((3, 1)),
        delivery_lag_seconds=np.arange(3),
        trial_count=3,
    )


def test_paired_design_condition_codes_and_interactions():
    paired = build_paired_tastant_design(_design(2.0), _design(4.0))
    assert paired.matrix.shape == (24, 8)
    np.testing.assert_array_equal(paired.matrix[:12, 1], -0.5)
    np.testing.assert_array_equal(paired.matrix[12:, 1], 0.5)
    np.testing.assert_array_equal(paired.matrix[:12, -1], -1.0)
    np.testing.assert_array_equal(paired.matrix[12:, -1], 2.0)
