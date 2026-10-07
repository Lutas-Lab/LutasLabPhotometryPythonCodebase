"""Adapters for validating the published Figure 5 MATLAB GLM outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from lutaslab_core.glm import predict_lagged_signal, r2_score
from scipy.io import loadmat
from scipy.signal import find_peaks

Figure5Epoch = Literal["dry", "total"]


@dataclass(frozen=True)
class Figure5Validation:
    epoch: Figure5Epoch
    mouse_id: int
    sample_rate_hz: float
    pre_seconds: float
    post_seconds: float
    sample_count: int
    lick_count: int
    stored_r2: float
    reconstructed_r2: float
    prediction_max_abs_error: float
    prediction_rmse: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _figure5_paths(data_root: str | Path, epoch: Figure5Epoch) -> tuple[Path, Path]:
    figure_root = Path(data_root) / "Figure 5 - Model GLM"
    if epoch == "dry":
        folder = figure_root / "Figure 5 - ac- GLM dry licking"
        return (
            folder / "FilteredGLM_8sNoEnsureFinalSave.mat",
            folder
            / "EightSecondCue_BEFORE_ENSURE"
            / "24-Jun-2025-cue-cue8sdates-nomeanacrossmice.mat",
        )
    if epoch == "total":
        model = (
            figure_root
            / "Figure 5 - d- GLM total licking"
            / "FilteredGLM8sWithEnsureTotalLickingFinalsave.mat"
        )
        return model, model
    raise ValueError("epoch must be 'dry' or 'total'")


def _cell_item(values: object, index: int) -> np.ndarray:
    items = np.atleast_1d(values)
    return np.asarray(items[index], dtype=float).reshape(-1)


def validate_saved_figure5_glm(
    data_root: str | Path,
    epoch: Figure5Epoch,
    *,
    mouse_id: int = 1,
) -> Figure5Validation:
    """Rebuild one published GLM prediction from processed figure data.

    This is a compatibility validation, not a refit: it reconstructs the
    exact raw-lag design implicit in the archived MATLAB coefficients and
    compares the resulting prediction with ``all_y_pred``.
    """

    model_path, data_path = _figure5_paths(data_root, epoch)
    model_file = loadmat(model_path, simplify_cells=True)
    data_file = model_file if data_path == model_path else loadmat(data_path, simplify_cells=True)
    model = model_file["FilteredGLMData"]
    data = data_file["ConcatData"]

    mouse_ids = np.asarray(model["mouse_ids"], dtype=int).reshape(-1)
    matches = np.flatnonzero(mouse_ids == mouse_id)
    if matches.size != 1:
        raise ValueError(f"mouse_id {mouse_id} is not uniquely represented")
    model_index = int(matches[0])

    row_mouse_ids = np.asarray(data["mouseidnumlist"], dtype=int).reshape(-1)
    rows = row_mouse_ids == mouse_id
    if not np.any(rows):
        raise ValueError(f"mouse_id {mouse_id} has no trial rows")
    photometry = np.asarray(data["concatmicephotom"], dtype=float)[rows, :-1].reshape(-1)
    raw_licks = np.asarray(data["concatmicelickmat"], dtype=float)[rows].reshape(-1)
    lick_indices = find_peaks(raw_licks, height=2)[0]
    lick_events = np.zeros(raw_licks.size, dtype=float)
    lick_events[lick_indices] = 1.0

    kernel = _cell_item(model["KernelPerMouse"], model_index)
    intercept = float(_cell_item(model["intercept"], model_index)[0])
    saved_prediction = _cell_item(model["all_y_pred"], model_index)
    saved_response = _cell_item(model["all_y"], model_index)
    if not np.array_equal(photometry, saved_response):
        raise ValueError("processed photometry does not match the archived GLM response")

    pre_samples = int(model["num_lags_pre"])
    post_samples = int(model["num_lags_post"])
    stride = int(model["stride"])
    expected_kernel_size = len(range(-pre_samples, post_samples + 1, stride))
    if kernel.size != expected_kernel_size:
        raise ValueError("archived kernel length does not match its lag metadata")
    prediction = predict_lagged_signal(
        lick_events,
        kernel,
        pre_samples,
        stride=stride,
        intercept=intercept,
    )
    difference = prediction - saved_prediction
    fs = float(model["Fs"])
    stored_r2 = float(_cell_item(model["r2_mouse"], model_index)[0])
    return Figure5Validation(
        epoch=epoch,
        mouse_id=mouse_id,
        sample_rate_hz=fs,
        pre_seconds=pre_samples / fs,
        post_seconds=post_samples / fs,
        sample_count=prediction.size,
        lick_count=lick_indices.size,
        stored_r2=stored_r2,
        reconstructed_r2=r2_score(saved_response, prediction),
        prediction_max_abs_error=float(np.max(np.abs(difference))),
        prediction_rmse=float(np.sqrt(np.mean(difference**2))),
    )
