from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lutaslab_photometry.plotting import (
    plot_irls_qc,
    plot_perievent_heatmap,
    plot_perievent_mean,
)


def _assert_figure_saves(figure):
    stream = BytesIO()
    figure.savefig(stream, format="png")
    assert stream.tell() > 0
    plt.close(figure)


def test_plot_perievent_mean_and_heatmap_render():
    time = np.linspace(-1.0, 1.0, 21)
    trials = np.vstack([np.sin(time), np.sin(time) + 0.2])

    mean_axis = plot_perievent_mean(time, trials, event_label="Cue", title="Mean")
    assert mean_axis.get_xlabel() == "Time from cue (s)"
    assert mean_axis.get_title() == "Mean"
    assert len(mean_axis.lines) == 2
    _assert_figure_saves(mean_axis.figure)

    heatmap_axis = plot_perievent_heatmap(time, trials, colorbar_label="dF/F")
    assert heatmap_axis.images
    assert heatmap_axis.get_ylabel() == "Trial"
    _assert_figure_saves(heatmap_axis.figure)


def test_plot_perievent_mean_rejects_empty_trials():
    with pytest.raises(ValueError, match="no valid trials"):
        plot_perievent_mean(np.linspace(-1.0, 1.0, 5), np.empty((0, 5)))


def test_plot_irls_qc_renders_all_panels():
    time = np.linspace(0.0, 2.0, 50)
    reference = 1.0 + 0.1 * np.sin(time)
    signal = 1.2 * reference + 0.01 * np.cos(2 * time)
    fitted = 1.2 * reference
    session = {
        "photo_time_465_ch1": time,
        "photometry_465_ch1": signal,
        "photometry_405_aligned_ch1": reference,
        "photometry_405_fitted_ch1": fitted,
        "dff_ch1": (signal - fitted) / fitted,
    }

    figure, axes = plot_irls_qc(session)

    assert len(axes) == 3
    assert all(axis.lines for axis in axes)
    _assert_figure_saves(figure)
