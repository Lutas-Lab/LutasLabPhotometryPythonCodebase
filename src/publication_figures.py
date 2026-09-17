from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np


VALID_FORMATS = ("svg", "png", "pdf")
METRIC_LABELS = {
    "mean": "Mean response",
    "auc": "Signed AUC",
    "positive_auc": "Positive AUC",
    "negative_auc": "Negative AUC",
    "peak": "Peak response",
    "peak_latency": "Peak latency (s)",
    "trough": "Trough response",
    "trough_latency": "Trough latency (s)",
}


def configure_publication_style(*, font_family="Arial", font_size=8):
    """Configure Matplotlib for compact figures with editable SVG text."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_family, "DejaVu Sans"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "axes.linewidth": 0.8,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": font_size - 1,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure_formats(
    fig,
    output_base,
    *,
    formats=("svg", "png"),
    dpi=300,
    transparent=True,
):
    """Save one figure in multiple vector/raster formats."""
    output_base = Path(output_base)
    unknown = set(formats).difference(VALID_FORMATS)
    if unknown:
        raise ValueError(f"Unsupported figure formats: {sorted(unknown)}")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for figure_format in formats:
        path = output_base.with_suffix(f".{figure_format}")
        fig.savefig(
            path,
            format=figure_format,
            dpi=dpi,
            transparent=transparent,
        )
        paths.append(path)
    return paths


def _stable_jitter(mouse, width=0.12):
    digest = hashlib.sha256(str(mouse).encode("utf-8")).digest()
    unit = int.from_bytes(digest[:4], "little") / (2**32 - 1)
    return (unit - 0.5) * 2 * width


def _category_key(row):
    return row["group"], row["condition"]


def _comparison_categories(row):
    if row["comparison"] == "condition_within_group":
        return (row["stratum"], row["level_a"]), (row["stratum"], row["level_b"])
    return (row["level_a"], row["stratum"]), (row["level_b"], row["stratum"])


def _p_label(row):
    value = row.get("p_adjusted_holm", row.get("p_value", np.nan))
    if not np.isfinite(value):
        return "q = n/a"
    if value < 0.001:
        return "q < 0.001"
    return f"q = {value:.3g}"


def _add_statistic_brackets(ax, rows, positions, data_values):
    usable = []
    for row in rows:
        first, second = _comparison_categories(row)
        if first in positions and second in positions:
            usable.append((positions[first], positions[second], row))
    if not usable:
        return
    finite = np.asarray(data_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low = float(np.min(finite)) if len(finite) else 0.0
    high = float(np.max(finite)) if len(finite) else 1.0
    span = high - low
    if span <= 0:
        span = max(abs(high), 1.0)
    step = 0.12 * span
    base = high + 0.12 * span
    for index, (first, second, row) in enumerate(usable):
        y = base + index * 1.5 * step
        left, right = sorted((first, second))
        ax.plot(
            [left, left, right, right],
            [y, y + step * 0.25, y + step * 0.25, y],
            color="black",
            linewidth=0.8,
            clip_on=False,
        )
        ax.text(
            (left + right) / 2,
            y + step * 0.35,
            _p_label(row),
            ha="center",
            va="bottom",
        )
    ax.set_ylim(top=base + len(usable) * 1.5 * step)


def plot_mouse_metric(
    mouse_rows,
    *,
    metric,
    test_rows=(),
    show_mice=True,
    show_pairs=True,
    show_statistics=True,
    figsize=(3.6, 3.2),
):
    """Plot mouse values and group summaries for one PSTH response metric."""
    import matplotlib.pyplot as plt
    from scipy import stats

    selected = [row for row in mouse_rows if row["metric"] == metric]
    if not selected:
        raise ValueError(f"No mouse rows were available for metric {metric!r}.")
    categories = sorted({_category_key(row) for row in selected})
    positions = {category: index for index, category in enumerate(categories)}
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=figsize)

    by_mouse = defaultdict(list)
    by_category = defaultdict(list)
    for row in selected:
        by_mouse[row["mouse"]].append(row)
        by_category[_category_key(row)].append(float(row["value"]))
    bracket_values = [row["value"] for row in selected]

    if show_pairs:
        for mouse_rows_for_one in by_mouse.values():
            if len(mouse_rows_for_one) < 2:
                continue
            mouse_rows_for_one = sorted(
                mouse_rows_for_one, key=lambda row: positions[_category_key(row)]
            )
            x = [
                positions[_category_key(row)] + _stable_jitter(row["mouse"])
                for row in mouse_rows_for_one
            ]
            y = [row["value"] for row in mouse_rows_for_one]
            ax.plot(x, y, color="0.72", linewidth=0.8, zorder=1)

    for category_index, category in enumerate(categories):
        rows = [row for row in selected if _category_key(row) == category]
        if show_mice:
            ax.scatter(
                [positions[category] + _stable_jitter(row["mouse"]) for row in rows],
                [row["value"] for row in rows],
                color=colors(category_index % 10),
                edgecolor="white",
                linewidth=0.4,
                s=28,
                zorder=2,
            )
        values = np.asarray(by_category[category], dtype=float)
        mean = float(np.mean(values))
        if len(values) > 1:
            sem = stats.sem(values)
            interval = stats.t.ppf(0.975, len(values) - 1) * sem
        else:
            interval = np.nan
        if np.isfinite(interval):
            bracket_values.extend((mean - interval, mean + interval))
        ax.errorbar(
            positions[category],
            mean,
            yerr=interval,
            color="black",
            marker="_",
            markersize=14,
            markeredgewidth=2,
            capsize=3,
            linewidth=1.3,
            zorder=3,
        )

    relevant_tests = [row for row in test_rows if row["metric"] == metric]
    if show_statistics:
        _add_statistic_brackets(
            ax,
            relevant_tests,
            positions,
            bracket_values,
        )
    labels = []
    for group, condition in categories:
        if group == "all" and condition == "all":
            labels.append("All sessions")
        elif group == "all":
            labels.append(condition)
        elif condition == "all":
            labels.append(group)
        else:
            labels.append(f"{group}\n{condition}")
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " "))
    ax.set(
        xticks=np.arange(len(categories)),
        xticklabels=labels,
        ylabel=metric_label,
        title=metric_label,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    return fig, ax


def save_metric_figures(
    mouse_rows,
    test_rows,
    output_dir,
    *,
    formats=("svg", "png"),
    dpi=300,
    font_family="Arial",
    show_mice=True,
    show_pairs=True,
    show_statistics=True,
):
    """Create and save one publication-style figure per response metric."""
    import matplotlib.pyplot as plt

    configure_publication_style(font_family=font_family)
    output_dir = Path(output_dir)
    metrics = sorted({row["metric"] for row in mouse_rows})
    paths = []
    for metric in metrics:
        fig, _ = plot_mouse_metric(
            mouse_rows,
            metric=metric,
            test_rows=test_rows,
            show_mice=show_mice,
            show_pairs=show_pairs,
            show_statistics=show_statistics,
        )
        paths.extend(
            save_figure_formats(
                fig,
                output_dir / f"psth_{metric}",
                formats=formats,
                dpi=dpi,
            )
        )
        plt.close(fig)
    return paths
