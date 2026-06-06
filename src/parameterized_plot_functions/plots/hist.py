"""Histogram plot builder with optional KDE and summary labels."""

from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
# import diptest

from ..annotations import apply_annotations
from ..axes import apply_axis_style, extend_y_axis_one_tick
from ..configs import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, LineSpec, OutputConfig, SeriesStyle
from ..legends import apply_patch_legend
from ..saving import finalize_figure
from ..style.themes import apply_theme


def plot_histogram(
    data_series: dict[str, np.ndarray],
    bins: Any,
    xlabel: str,
    ylabel: str,
    title: str,
    series_styles: dict[str, SeriesStyle] | None = None,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    hist_stat: str = "probability",
    plot_kde: bool = False,
    plot_mean: bool = False,
    plot_std: bool = False,
    cumulative: bool = False,
    fitted_distribution: str | None = None,
    percentile_markers: list[float] | None = None,
    perform_dip_test: bool = False,
    vertical_lines: list[float] | None = None,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    annotations: list[AnnotationSpec] | None = None,
    line_spec: LineSpec | None = None,
    extend_y_one_tick: bool = False,
) -> Figure | None:
    """
    Function purpose:
        Draw one or more named histograms with optional KDE, summary statistics,
        reference lines, annotations, legends, and axis formatting.

    Args:
        data_series: Mapping from series key to histogram values.
        bins: Histogram bin specification passed to seaborn.
        xlabel: Text for the x-axis label.
        ylabel: Text for the y-axis label.
        title: Figure title text.
        series_styles: Optional mapping from series key to visual style.
        axis_style: Optional axis styling configuration.
        figure_style: Optional figure-level styling and display configuration.
        legend_style: Optional legend styling and placement configuration.
        output_config: Optional output saving and return behavior configuration.
        hist_stat: Histogram statistic passed to seaborn.
        plot_kde: Whether to overlay a kernel density estimate.
        plot_mean: Whether to append the mean to legend labels.
        plot_std: Whether to append the sample standard deviation to legend labels.
        perform_dip_test: Reserved flag for dip-test label support.
        vertical_lines: Optional x-values for legacy vertical reference lines.
        xlims: Optional x-axis limits.
        ylims: Optional y-axis limits.
        xticks: Optional x-axis tick positions.
        yticks: Optional y-axis tick positions.
        xtick_labels: Optional x-axis tick labels.
        ytick_labels: Optional y-axis tick labels.
        annotations: Optional annotations to draw on the axis.
        line_spec: Optional vertical and horizontal reference lines.
        extend_y_one_tick: Whether to extend y-axis by one major tick interval.

    Outputs:
        The figure when ``output_config.return_fig`` is true, otherwise ``None``.
    """
    # Instantiate default configs at call time to avoid shared mutable state.
    if axis_style is None:
        axis_style = AxisStyle()
    if figure_style is None:
        figure_style = FigureStyle()
    if legend_style is None:
        legend_style = LegendStyle()
    if output_config is None:
        output_config = OutputConfig()
    if series_styles is None:
        series_styles = {key: SeriesStyle(label=key) for key in data_series}

    # Disable interactive rendering for batch/script usage.
    plt.ioff()
    apply_theme(figure_style, axis_style)
    import seaborn as sns

    # Build histograms and legend patch specs from the same style mapping.
    fig, ax = plt.subplots(figsize=figure_style.figure_size)
    patch_specs = []

    for key, values in data_series.items():
        style = series_styles.get(key, SeriesStyle(label=key))
        values = np.asarray(values)

        sns.histplot(
            data=values,
            bins=bins,
            stat=hist_stat,
            kde=plot_kde,
            color=style.color,
            alpha=style.alpha,
            edgecolor="none",
            ax=ax,
            cumulative=cumulative,
            line_kws={"linewidth": style.linewidth},
        )

        if fitted_distribution == "normal":
            mean = np.nanmean(values)
            std = np.nanstd(values, ddof=1)
            if std > 0:
                x_pdf = np.linspace(np.nanmin(values), np.nanmax(values), 200)
                y_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pdf - mean) / std) ** 2)
                ax.plot(x_pdf, y_pdf, color=style.color, linestyle="--", linewidth=style.linewidth)

        if percentile_markers is not None:
            for percentile in percentile_markers:
                ax.axvline(np.nanpercentile(values, percentile), color=style.color, linestyle=":", linewidth=1.5)

        # Optional summary statistics are appended to the legend label.
        label = style.label if style.label is not None else key
        suffix_parts = []

        # if perform_dip_test:
        #     _, pval = diptest.diptest(values)
        #     suffix_parts.append(f"Dip p={pval:.3f}")
        if plot_mean:
            suffix_parts.append(f"μ={np.nanmean(values):.2f}")
        if plot_std:
            suffix_parts.append(f"σ={np.nanstd(values, ddof=1):.2f}")

        if suffix_parts:
            label = f"{label}, " + ", ".join(suffix_parts)

        patch_specs.append((label, style.color))

    # Optional reference lines are drawn after data so they overlay histograms.
    if line_spec is not None:
        for spec in line_spec.vertical:
            ax.axvline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)
        for spec in line_spec.horizontal:
            ax.axhline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)

    # Legacy vertical line support is kept separate from LineSpec.
    if vertical_lines is not None:
        for v in vertical_lines:
            ax.axvline(x=v, color="black", linestyle="--", linewidth=1.5)

    # Shared axis helper applies ticks, labels, limits, scales, and spine style.
    apply_axis_style(
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel,
        axis_style=axis_style,
        xlims=xlims,
        ylims=ylims,
        xticks=xticks,
        yticks=yticks,
        xtick_labels=xtick_labels,
        ytick_labels=ytick_labels,
    )

    if extend_y_one_tick:
        extend_y_axis_one_tick(ax)

    apply_patch_legend(ax, patch_specs, legend_style)
    apply_annotations(ax, annotations)

    return finalize_figure(fig, ax, title, figure_style, output_config)
