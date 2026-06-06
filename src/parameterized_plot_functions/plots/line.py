"""Line plot builder with optional errors, reference lines, and annotations."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
import numpy as np

from ..annotations import apply_annotations
from ..axes import apply_axis_style, extend_y_axis_one_tick
from ..configs import (
    AnnotationSpec,
    AxisStyle,
    FigureStyle,
    LegendStyle,
    LineSpec,
    OutputConfig,
    SeriesStyle,
    ShadedRegionSpec,
)
from ..legends import apply_line_legend
from ..saving import finalize_figure
from ..style.themes import apply_theme


def plot_line(
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
    yerr_series: dict[str, np.ndarray] | None = None,
    ci_series: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    use_fill_between: bool = True,
    step_where: str | None = None,
    rolling_window: int | None = None,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    annotations: list[AnnotationSpec] | None = None,
    line_spec: LineSpec | None = None,
    shaded_regions: list[ShadedRegionSpec] | None = None,
    endpoint_labels: bool = False,
    use_mask: bool = True,
    errorbar_capsize: float = 4,
    errorbar_elinewidth: float = 2,
    errorbar_capthick: float = 2,
    use_line_color_for_error: bool = False,
    rotate_xticks: bool = False,
    plot_minor_ticks: bool = False,
    extend_y_one_tick: bool = False,
) -> Figure | None:
    """
    Function purpose:
        Draw one or more named line series with optional uncertainty, reference
        lines, annotations, legends, and axis formatting.

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
        yerr_series: Optional mapping from series key to y-error values.
        use_fill_between: Whether to render y-error as a filled band.
        xlims: Optional x-axis limits.
        ylims: Optional y-axis limits.
        xticks: Optional x-axis tick positions.
        yticks: Optional y-axis tick positions.
        xtick_labels: Optional x-axis tick labels.
        ytick_labels: Optional y-axis tick labels.
        annotations: Optional annotations to draw on the axis.
        line_spec: Optional vertical and horizontal reference lines.
        use_mask: Whether to drop non-finite y values before plotting.
        errorbar_capsize: Cap size for errorbar rendering.
        errorbar_elinewidth: Errorbar line width.
        errorbar_capthick: Errorbar cap thickness.
        use_line_color_for_error: Whether error bands/errorbars use series color.
        rotate_xticks: Whether to rotate x tick labels by 90 degrees.
        plot_minor_ticks: Whether to add minor ticks.
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
        series_styles = {key: SeriesStyle(label=key) for key in x_series}

    # Disable interactive rendering for batch/script usage.
    plt.ioff()
    apply_theme(figure_style, axis_style)

    fig, ax = plt.subplots(figsize=figure_style.figure_size)

    # Each key identifies one plotted series; missing styles fall back per series.
    for key in x_series:
        x = np.asarray(x_series[key])
        y = np.asarray(y_series[key])
        style = series_styles.get(key, SeriesStyle(label=key))

        # Mask non-finite y values so broken points do not drive plotting errors.
        if use_mask:
            mask = np.isfinite(y)
            x_plot = x[mask]
            y_plot = y[mask]
        else:
            x_plot = x
            y_plot = y

        if rolling_window is not None and rolling_window > 1:
            kernel = np.ones(rolling_window) / rolling_window
            y_plot = np.convolve(y_plot, kernel, mode="same")

        # Error values can be rendered as a filled band or as errorbar caps.
        if yerr_series is not None and key in yerr_series:
            yerr = np.asarray(yerr_series[key])
            if use_mask:
                yerr = yerr[mask]

            if use_fill_between:
                plotter = ax.step if step_where else ax.plot
                plotter(
                    x_plot,
                    y_plot,
                    color=style.color,
                    linewidth=style.linewidth,
                    linestyle=style.linestyle,
                    marker=style.marker,
                    markersize=style.markersize,
                    alpha=style.alpha,
                    **({"where": step_where} if step_where else {}),
                )
                ax.fill_between(
                    x_plot,
                    y_plot - yerr,
                    y_plot + yerr,
                    color=style.color if use_line_color_for_error else "black",
                    alpha=min(style.alpha, 0.20),
                )
            else:
                ax.errorbar(
                    x_plot,
                    y_plot,
                    yerr=yerr,
                    fmt=style.marker if style.marker is not None else "none",
                    color=style.color,
                    ecolor=style.color if use_line_color_for_error else "black",
                    linewidth=style.linewidth,
                    linestyle=style.linestyle,
                    markersize=style.markersize,
                    capsize=errorbar_capsize,
                    elinewidth=errorbar_elinewidth,
                    capthick=errorbar_capthick,
                    alpha=style.alpha,
                )
                if style.marker is None:
                    plotter = ax.step if step_where else ax.plot
                    plotter(
                        x_plot,
                        y_plot,
                        color=style.color,
                        linewidth=style.linewidth,
                        linestyle=style.linestyle,
                        alpha=style.alpha,
                        **({"where": step_where} if step_where else {}),
                    )
        else:
            plotter = ax.step if step_where else ax.plot
            plotter(
                x_plot,
                y_plot,
                color=style.color,
                linewidth=style.linewidth,
                linestyle=style.linestyle,
                marker=style.marker,
                markersize=style.markersize,
                alpha=style.alpha,
                **({"where": step_where} if step_where else {}),
            )

        if ci_series is not None and key in ci_series:
            lower, upper = ci_series[key]
            lower_arr = np.asarray(lower)[mask] if use_mask else np.asarray(lower)
            upper_arr = np.asarray(upper)[mask] if use_mask else np.asarray(upper)
            ax.fill_between(x_plot, lower_arr, upper_arr, color=style.color, alpha=min(style.alpha, 0.18))

        if endpoint_labels and len(x_plot) > 0:
            ax.text(x_plot[-1], y_plot[-1], style.label if style.label is not None else key, color=style.color)

    if shaded_regions is not None:
        for region in shaded_regions:
            ax.axvspan(region.xmin, region.xmax, color=region.color, alpha=region.alpha, label=region.label)

    # Optional reference lines are drawn after data so they overlay the series.
    if line_spec is not None:
        for spec in line_spec.vertical:
            ax.axvline(
                x=spec.value,
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                alpha=spec.alpha,
            )
        for spec in line_spec.horizontal:
            ax.axhline(
                y=spec.value,
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                alpha=spec.alpha,
            )

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

    if rotate_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Minor ticks use a fixed subdivision count for consistent visual density.
    if plot_minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_tick_params(which="minor", width=axis_style.tick_width / 2, length=axis_style.tick_length / 2)
        ax.yaxis.set_tick_params(which="minor", width=axis_style.tick_width / 2, length=axis_style.tick_length / 2)

    if extend_y_one_tick:
        extend_y_axis_one_tick(ax)

    # Legends and annotations are applied after axis setup to avoid stale handles.
    apply_line_legend(ax, series_styles, legend_style, linewidth_multiplier=1.0)
    apply_annotations(ax, annotations)

    return finalize_figure(fig, ax, title, figure_style, output_config)
