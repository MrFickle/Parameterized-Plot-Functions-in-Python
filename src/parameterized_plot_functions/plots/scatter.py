"""Scatter plot builder with optional regression fits and colorbars."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ..annotations import apply_annotations
from ..axes import apply_axis_style
from ..configs import (
    AnnotationSpec,
    AxisStyle,
    ColorbarConfig,
    FigureStyle,
    LegendStyle,
    LineSpec,
    OutputConfig,
    SeriesStyle,
)
from ..legends import apply_line_legend
from ..saving import finalize_figure


def plot_scatter(
    x_series: dict[str, np.ndarray],
    y_series: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    series_styles: dict[str, SeriesStyle] | None = None,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    annotations: list[AnnotationSpec] | None = None,
    line_spec: LineSpec | None = None,
    do_linear_reg_fit: bool = False,
    plot_r2_score: bool = False,
    colorbar_config: ColorbarConfig | None = None,
    color_values: dict[str, np.ndarray] | None = None,
) -> Figure | None:
    """
    Function purpose:
        Draw one or more named scatter series with optional linear regression,
        R-squared labels, reference lines, annotations, legends, and colorbar.

    Args:
        x_series: Mapping from series key to x values.
        y_series: Mapping from series key to y values.
        xlabel: Text for the x-axis label.
        ylabel: Text for the y-axis label.
        title: Figure title text.
        series_styles: Optional mapping from series key to visual style.
        axis_style: Optional axis styling configuration.
        figure_style: Optional figure-level styling and display configuration.
        legend_style: Optional legend styling and placement configuration.
        output_config: Optional output saving and return behavior configuration.
        xlims: Optional x-axis limits.
        ylims: Optional y-axis limits.
        xticks: Optional x-axis tick positions.
        yticks: Optional y-axis tick positions.
        xtick_labels: Optional x-axis tick labels.
        ytick_labels: Optional y-axis tick labels.
        annotations: Optional annotations to draw on the axis.
        line_spec: Optional vertical and horizontal reference lines.
        do_linear_reg_fit: Whether to fit and plot a linear regression per series.
        plot_r2_score: Whether to append R-squared values to legend labels.
        colorbar_config: Optional colorbar configuration.
        color_values: Optional mapping from series key to color-mapped values.

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
        series_styles = {key: SeriesStyle(label=key) for key in x_series}

    # Disable interactive rendering for batch/script usage.
    plt.ioff()
    if figure_style.use_seaborn:
        sns.set(style=figure_style.seaborn_style, font_scale=figure_style.seaborn_font_scale)

    fig, ax = plt.subplots(figsize=figure_style.figure_size)
    updated_styles = {}

    # Keep the first color-mapped scatter so one shared colorbar can be attached.
    colorbar_mappable = None

    # Each key identifies one plotted series; missing styles fall back per series.
    for key in x_series:
        x = np.asarray(x_series[key])
        y = np.asarray(y_series[key])
        style = series_styles.get(key, SeriesStyle(label=key))
        label = style.label if style.label is not None else key

        # Color values switch the series from fixed color to colormap rendering.
        series_color_values = None
        if color_values is not None:
            series_color_values = color_values.get(key, None)

        if colorbar_config is not None and colorbar_config.enabled and series_color_values is not None:
            scatter = ax.scatter(
                x,
                y,
                c=series_color_values,
                cmap=colorbar_config.colormap,
                alpha=style.alpha,
                s=(style.markersize * style.m_size_factor) ** 2,
                marker=style.marker if style.marker is not None else "o",
                linewidth=0.8,
            )

            if colorbar_mappable is None:
                colorbar_mappable = scatter
        else:
            ax.scatter(
                x,
                y,
                color=style.color,
                alpha=style.alpha,
                s=(style.markersize * style.m_size_factor) ** 2,
                marker=style.marker if style.marker is not None else "o",
                linewidth=0.8,
            )

        # Optional linear fit is drawn with the same style as its source series.
        if do_linear_reg_fit:
            reg = LinearRegression().fit(x.reshape(-1, 1), y)
            y_pred = reg.predict(x.reshape(-1, 1))
            ax.plot(x, y_pred, color=style.color, linewidth=style.linewidth, linestyle=style.linestyle)

            if plot_r2_score:
                r2 = r2_score(y, y_pred)
                label = f"{label}, R²={r2:.2f}"

        # Legend labels may be augmented with regression diagnostics.
        updated_styles[key] = SeriesStyle(
            color=style.color,
            label=label,
            linewidth=style.linewidth,
            linestyle=style.linestyle,
            marker=style.marker,
            markersize=style.markersize,
            alpha=style.alpha,
        )

    # The colorbar is only meaningful when at least one plotted series is mapped.
    if colorbar_config is not None and colorbar_config.enabled and colorbar_mappable is not None:
        cbar = fig.colorbar(
            colorbar_mappable,
            ax=ax,
            orientation=colorbar_config.orientation,
            location=colorbar_config.location,
            ticks=colorbar_config.ticks,
        )
        if colorbar_config.label is not None:
            cbar.set_label(colorbar_config.label, fontweight="bold")
        if colorbar_config.tick_labels is not None:
            if colorbar_config.orientation == "vertical":
                cbar.ax.set_yticklabels(colorbar_config.tick_labels)
            else:
                cbar.ax.set_xticklabels(colorbar_config.tick_labels)

    # Optional reference lines are drawn after data so they overlay the points.
    if line_spec is not None:
        for spec in line_spec.vertical:
            ax.axvline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)
        for spec in line_spec.horizontal:
            ax.axhline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)

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

    # Use updated labels so regression statistics appear in the legend.
    apply_line_legend(ax, updated_styles, legend_style)
    apply_annotations(ax, annotations)

    return finalize_figure(fig, ax, title, figure_style, output_config)
