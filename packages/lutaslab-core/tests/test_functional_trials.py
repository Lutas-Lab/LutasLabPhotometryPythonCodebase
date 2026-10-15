import numpy as np
import pytest

from lutaslab_core.functional_trials import FunctionalTrialDataset


def _dataset(outcome=None):
    return FunctionalTrialDataset(
        time=np.array([-1.0, 0.0, 1.0]),
        outcome=(
            np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
            if outcome is None
            else outcome
        ),
        mouse=np.array(["M1", "M2"]),
        session=np.array(["M1_s1", "M2_s1"]),
        trial_index=np.array([1, 1]),
        alignment_time=np.array([5.0, 6.0]),
        scalar_covariates={"condition": np.array(["a", "b"])},
    )


def test_functional_trial_dataset_reports_preflight_summary():
    dataset = _dataset()

    assert dataset.n_trials == 2
    assert dataset.n_timepoints == 3
    assert dataset.summary()["levels"]["condition"] == ["a", "b"]
    assert dataset.summary()["level_counts"]["condition"]["a"] == {
        "trials": 1,
        "mice": 1,
        "sessions": 1,
    }
    assert dataset.summary()["warnings"]


def test_functional_trial_dataset_rejects_incomplete_outcome():
    with pytest.raises(ValueError, match="complete and finite"):
        _dataset(np.array([[1.0, np.nan, 3.0], [2.0, 3.0, 4.0]]))


def test_functional_trial_dataset_rejects_bad_covariate_names():
    with pytest.raises(ValueError, match="Invalid scalar covariate"):
        FunctionalTrialDataset(
            time=np.array([0.0, 1.0]),
            outcome=np.ones((2, 2)),
            mouse=np.array(["M1", "M2"]),
            session=np.array(["S1", "S2"]),
            trial_index=np.array([1, 1]),
            alignment_time=np.array([2.0, 2.0]),
            scalar_covariates={"bad name": np.array([1, 2])},
        )
