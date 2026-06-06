"""Box and violin plot builders for grouped distributions."""

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from ..annotations import apply_annotations
from ..axes import apply_axis_style
from ..configs import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, OutputConfig, SeriesStyle
from ..legends import apply_patch_legend
from ..saving import finalize_figure
from ..style.themes import apply_theme


def plot_box(
    data_series: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    series_styles: dict[str, SeriesStyle] | None = None,
    show_means: bool = False,
    notch: bool = False,
    show_outliers: bool = True,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    positions: list[float] | None = None,
    tick_labels: list[str] | None = None,
    widths: float | list[float] = 0.5,
    box_alpha: float | None = None,
    mean_marker: str = "^",
    median_color: str = "black",
    grid_axis: Literal["x", "y", "both", "none"] = "none",
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    annotations: list[AnnotationSpec] | None = None,
) -> Figure | None:
    """
    Function purpose:
        Draw a parameterized box plot from named value arrays.

    Args:
        data_series: Mapping from group name to numeric values.
        xlabel: Text for the x-axis label.
        ylabel: Text for the y-axis label.
        title: Figure title text.
        axis_style: Optional axis styling configuration.
        figure_style: Optional figure-level styling configuration.
        legend_style: Optional legend styling configuration.
        output_config: Optional output saving and return behavior.
        series_styles: Optional per-group color and label styles.
        show_means: Whether to draw mean markers.
        notch: Whether to draw notched boxes.
        show_outliers: Whether to display outlier markers.
        orientation: Whether boxes are vertical or horizontal.
        positions: Optional numeric positions for each group.
        tick_labels: Optional displayed labels for group ticks.
        widths: Box width or per-box widths.
        box_alpha: Optional alpha override for box fills.
        mean_marker: Marker used for mean points.
        median_color: Color used for median lines.
        grid_axis: Axis on which to draw a grid.
        xlims: Optional x-axis limits.
        ylims: Optional y-axis limits.
        annotations: Optional annotations to draw on the axis.

    Outputs:
        The figure when ``output_config.return_fig`` is true, otherwise ``None``.
    """
    # Instantiate configs at call time so callers can omit boilerplate safely.
    axis_style = axis_style or AxisStyle()
    figure_style = figure_style or FigureStyle()
    legend_style = legend_style or LegendStyle()
    output_config = output_config or OutputConfig()
    series_styles = series_styles or {key: SeriesStyle(label=key) for key in data_series}

    # Disable interactive rendering and apply the requested style preset.
    plt.ioff()
    apply_theme(figure_style, axis_style)

    # Convert the dictionary into ordered arrays for Matplotlib.
    fig, ax = plt.subplots(figsize=figure_style.figure_size)
    keys = list(data_series.keys())
    values = [np.asarray(data_series[key]) for key in keys]
    plot_positions = positions if positions is not None else list(np.arange(1, len(keys) + 1))
    display_labels = tick_labels if tick_labels is not None else keys

    # Draw the core box plot and keep patch handles for styling.
    patch = ax.boxplot(
        values,
        positions=plot_positions,
        widths=widths,
        patch_artist=True,
        showmeans=show_means,
        notch=notch,
        showfliers=show_outliers,
        orientation=orientation,
        meanprops={"marker": mean_marker},
        medianprops={"color": median_color},
    )

    # Apply per-series fill colors from the shared SeriesStyle config.
    for box, key in zip(patch["boxes"], keys):
        style = series_styles.get(key, SeriesStyle(label=key))
        box.set_facecolor(style.color)
        box.set_alpha(style.alpha if box_alpha is None else box_alpha)

    # Apply shared axis styling with orientation-aware tick placement.
    if orientation == "vertical":
        apply_axis_style(ax, xlabel, ylabel, axis_style, xlims=xlims, ylims=ylims, xticks=list(plot_positions), xtick_labels=display_labels)
    else:
        apply_axis_style(ax, xlabel, ylabel, axis_style, xlims=xlims, ylims=ylims, yticks=list(plot_positions), ytick_labels=display_labels)

    # Optional grid support is useful for distribution comparison.
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, alpha=0.25)

    # Build a simple patch legend from the same styles used to color boxes.
    apply_patch_legend(ax, [(series_styles[key].label or key, series_styles[key].color) for key in keys], legend_style)
    apply_annotations(ax, annotations)
    return finalize_figure(fig, ax, title, figure_style, output_config)


def plot_violin(
    data_series: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    series_styles: dict[str, SeriesStyle] | None = None,
    show_means: bool = False,
    show_extrema: bool = True,
    show_medians: bool = True,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    positions: list[float] | None = None,
    tick_labels: list[str] | None = None,
    widths: float = 0.5,
    violin_alpha: float | None = None,
    quantiles: list[list[float]] | None = None,
    grid_axis: Literal["x", "y", "both", "none"] = "none",
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    annotations: list[AnnotationSpec] | None = None,
) -> Figure | None:
    """
    Function purpose:
        Draw a parameterized violin plot from named value arrays.

    Args:
        data_series: Mapping from group name to numeric values.
        xlabel: Text for the x-axis label.
        ylabel: Text for the y-axis label.
        title: Figure title text.
        axis_style: Optional axis styling configuration.
        figure_style: Optional figure-level styling configuration.
        legend_style: Optional legend styling configuration.
        output_config: Optional output saving and return behavior.
        series_styles: Optional per-group color and label styles.
        show_means: Whether to display mean markers.
        show_extrema: Whether to display extrema lines.
        show_medians: Whether to display median markers.
        orientation: Whether violins are vertical or horizontal.
        positions: Optional numeric positions for each group.
        tick_labels: Optional displayed labels for group ticks.
        widths: Violin width.
        violin_alpha: Optional alpha override for violin fills.
        quantiles: Optional quantile lines per violin.
        grid_axis: Axis on which to draw a grid.
        xlims: Optional x-axis limits.
        ylims: Optional y-axis limits.
        annotations: Optional annotations to draw on the axis.

    Outputs:
        The figure when ``output_config.return_fig`` is true, otherwise ``None``.
    """
    # Instantiate configs at call time so callers can omit boilerplate safely.
    axis_style = axis_style or AxisStyle()
    figure_style = figure_style or FigureStyle()
    legend_style = legend_style or LegendStyle()
    output_config = output_config or OutputConfig()
    series_styles = series_styles or {key: SeriesStyle(label=key) for key in data_series}

    # Disable interactive rendering and apply the requested style preset.
    plt.ioff()
    apply_theme(figure_style, axis_style)

    # Convert the dictionary into ordered arrays for Matplotlib.
    fig, ax = plt.subplots(figsize=figure_style.figure_size)
    keys = list(data_series.keys())
    values = [np.asarray(data_series[key]) for key in keys]
    plot_positions = positions if positions is not None else list(np.arange(1, len(keys) + 1))
    display_labels = tick_labels if tick_labels is not None else keys

    # Draw the core violin plot and keep body handles for styling.
    parts = ax.violinplot(
        values,
        positions=plot_positions,
        widths=widths,
        showmeans=show_means,
        showextrema=show_extrema,
        showmedians=show_medians,
        orientation=orientation,
        quantiles=quantiles,
    )

    # Apply per-series fill colors from the shared SeriesStyle config.
    for body, key in zip(parts["bodies"], keys):
        style = series_styles.get(key, SeriesStyle(label=key))
        body.set_facecolor(style.color)
        body.set_alpha(style.alpha if violin_alpha is None else violin_alpha)

    # Apply shared axis styling with orientation-aware tick placement.
    if orientation == "vertical":
        apply_axis_style(ax, xlabel, ylabel, axis_style, xlims=xlims, ylims=ylims, xticks=list(plot_positions), xtick_labels=display_labels)
    else:
        apply_axis_style(ax, xlabel, ylabel, axis_style, xlims=xlims, ylims=ylims, yticks=list(plot_positions), ytick_labels=display_labels)

    # Optional grid support is useful for distribution comparison.
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis, alpha=0.25)

    # Build a simple patch legend from the same styles used to color violins.
    apply_patch_legend(ax, [(series_styles[key].label or key, series_styles[key].color) for key in keys], legend_style)
    apply_annotations(ax, annotations)
    return finalize_figure(fig, ax, title, figure_style, output_config)
