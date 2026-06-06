"""2D density and contour plot builders."""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from ..axes import apply_axis_style
from ..annotations import apply_annotations
from ..configs import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, LineSpec, OutputConfig
from ..saving import finalize_figure
from ..style.themes import apply_theme


def plot_hexbin(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    gridsize: int = 30,
    cmap: str = "viridis",
    mincnt: int | None = 1,
    colorbar: bool = True,
    reduce_function: Any = np.mean,
    values: np.ndarray | None = None,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    colorbar_label: str | None = None,
    extent: tuple[float, float, float, float] | None = None,
    bins: str | None = None,
    linewidths: float = 0.0,
    alpha: float = 1.0,
    line_spec: LineSpec | None = None,
    annotations: list[AnnotationSpec] | None = None,
) -> Figure | None:
    """
    Function purpose:
        Draw a parameterized hexbin / 2D density plot.

    Args:
        x: X coordinates for observations.
        y: Y coordinates for observations.
        xlabel: Text for the x-axis label.
        ylabel: Text for the y-axis label.
        title: Figure title text.
        axis_style: Optional axis styling configuration.
        figure_style: Optional figure-level styling configuration.
        legend_style: Optional legend/colorbar styling configuration.
        output_config: Optional output saving and return behavior.
        gridsize: Number of hexagons in the x direction.
        cmap: Colormap name.
        mincnt: Minimum count needed to color a hexagon.
        colorbar: Whether to draw a colorbar.
        reduce_function: Function used when `values` are supplied.
        values: Optional values aggregated inside each hexagon.
        xlims: Optional x-axis limits.
        ylims: Optional y-axis limits.
        xticks: Optional x-axis tick positions.
        yticks: Optional y-axis tick positions.
        xtick_labels: Optional x-axis tick labels.
        ytick_labels: Optional y-axis tick labels.
        colorbar_label: Optional colorbar label.
        extent: Optional plot extent as xmin, xmax, ymin, ymax.
        bins: Optional count binning mode, such as "log".
        linewidths: Hexagon edge line width.
        alpha: Hexagon alpha.
        line_spec: Optional vertical and horizontal reference lines.
        annotations: Optional text annotations.

    Outputs:
        The figure when ``output_config.return_fig`` is true, otherwise ``None``.
    """
    # Instantiate configs at call time so callers can omit boilerplate safely.
    axis_style = axis_style or AxisStyle()
    figure_style = figure_style or FigureStyle()
    legend_style = legend_style or LegendStyle()
    output_config = output_config or OutputConfig()

    # Disable interactive rendering and apply the requested style preset.
    plt.ioff()
    apply_theme(figure_style, axis_style)
    fig, ax = plt.subplots(figsize=figure_style.figure_size)

    # Draw the hexbin layer with optional aggregated values.
    mappable = ax.hexbin(
        x,
        y,
        C=values,
        gridsize=gridsize,
        cmap=cmap,
        mincnt=mincnt,
        reduce_C_function=reduce_function,
        extent=extent,
        bins=bins,
        linewidths=linewidths,
        alpha=alpha,
    )

    # Attach a colorbar when requested.
    if colorbar and legend_style.enabled:
        cbar = fig.colorbar(mappable, ax=ax)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label, fontweight="bold", fontsize=legend_style.fontsize)
        cbar.ax.tick_params(labelsize=legend_style.fontsize)

    # Draw optional reference lines before final annotations.
    if line_spec is not None:
        for spec in line_spec.vertical:
            ax.axvline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)
        for spec in line_spec.horizontal:
            ax.axhline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)

    # Apply shared axis formatting and optional text annotations.
    apply_axis_style(ax, xlabel, ylabel, axis_style, xlims=xlims, ylims=ylims, xticks=xticks, yticks=yticks, xtick_labels=xtick_labels, ytick_labels=ytick_labels)
    apply_annotations(ax, annotations)
    return finalize_figure(fig, ax, title, figure_style, output_config)


def plot_contour(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    levels: int | list[float] = 10,
    filled: bool = True,
    cmap: str = "viridis",
    colorbar: bool = True,
    label_contours: bool = False,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    colorbar_label: str | None = None,
    linewidths: float = 1.5,
    alpha: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    line_spec: LineSpec | None = None,
    annotations: list[AnnotationSpec] | None = None,
) -> Figure | None:
    """
    Function purpose:
        Draw a parameterized contour or filled contour plot.

    Args:
        x: X coordinate grid.
        y: Y coordinate grid.
        z: Z values defined over the coordinate grid.
        xlabel: Text for the x-axis label.
        ylabel: Text for the y-axis label.
        title: Figure title text.
        axis_style: Optional axis styling configuration.
        figure_style: Optional figure-level styling configuration.
        legend_style: Optional legend/colorbar styling configuration.
        output_config: Optional output saving and return behavior.
        levels: Number of levels or explicit contour levels.
        filled: Whether to draw filled contours.
        cmap: Colormap name.
        colorbar: Whether to draw a colorbar.
        label_contours: Whether to label contour lines.
        xlims: Optional x-axis limits.
        ylims: Optional y-axis limits.
        xticks: Optional x-axis tick positions.
        yticks: Optional y-axis tick positions.
        xtick_labels: Optional x-axis tick labels.
        ytick_labels: Optional y-axis tick labels.
        colorbar_label: Optional colorbar label.
        linewidths: Contour line width for unfilled contours.
        alpha: Contour alpha.
        vmin: Optional lower color bound.
        vmax: Optional upper color bound.
        line_spec: Optional vertical and horizontal reference lines.
        annotations: Optional text annotations.

    Outputs:
        The figure when ``output_config.return_fig`` is true, otherwise ``None``.
    """
    # Instantiate configs at call time so callers can omit boilerplate safely.
    axis_style = axis_style or AxisStyle()
    figure_style = figure_style or FigureStyle()
    legend_style = legend_style or LegendStyle()
    output_config = output_config or OutputConfig()

    # Disable interactive rendering and apply the requested style preset.
    plt.ioff()
    apply_theme(figure_style, axis_style)
    fig, ax = plt.subplots(figsize=figure_style.figure_size)

    # Select filled or line contours while exposing relevant Matplotlib options.
    plotter = ax.contourf if filled else ax.contour
    contour_kwargs = {"levels": levels, "cmap": cmap, "alpha": alpha, "vmin": vmin, "vmax": vmax}
    if not filled:
        contour_kwargs["linewidths"] = linewidths
    contours = plotter(x, y, z, **contour_kwargs)

    # Label contour lines only when line contours are used.
    if label_contours and not filled:
        ax.clabel(contours, inline=True)

    # Attach a colorbar when requested.
    if colorbar and legend_style.enabled:
        cbar = fig.colorbar(contours, ax=ax)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label, fontweight="bold", fontsize=legend_style.fontsize)
        cbar.ax.tick_params(labelsize=legend_style.fontsize)

    # Draw optional reference lines before final annotations.
    if line_spec is not None:
        for spec in line_spec.vertical:
            ax.axvline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)
        for spec in line_spec.horizontal:
            ax.axhline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)

    # Apply shared axis formatting and optional text annotations.
    apply_axis_style(ax, xlabel, ylabel, axis_style, xlims=xlims, ylims=ylims, xticks=xticks, yticks=yticks, xtick_labels=xtick_labels, ytick_labels=ytick_labels)
    apply_annotations(ax, annotations)
    return finalize_figure(fig, ax, title, figure_style, output_config)
