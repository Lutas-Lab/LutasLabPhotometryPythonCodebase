import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lutaslab_core.plotting import plot_perievent_heatmap, plot_perievent_mean


def test_shared_perievent_plots_draw_on_supplied_axes():
    time = np.array([-1.0, -0.5, 0.0, 0.5])
    trials = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    figure, axes = plt.subplots(1, 2)
    try:
        assert plot_perievent_mean(time, trials, ax=axes[0]) is axes[0]
        assert plot_perievent_heatmap(time, trials, ax=axes[1]) is axes[1]
        assert len(axes[0].lines) >= 2
        assert len(axes[1].images) == 1
    finally:
        plt.close(figure)
