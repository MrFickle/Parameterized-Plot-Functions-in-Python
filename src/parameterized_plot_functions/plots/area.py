"""Area plot builder."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from ..axes import apply_axis_style
from ..annotations import apply_annotations
from ..configs import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, LineSpec, OutputConfig, SeriesStyle, ShadedRegionSpec
from ..legends import apply_line_legend
from ..saving import finalize_figure
from ..style.themes import apply_theme


def plot_area(
    x: np.ndarray,
    y_series: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    series_styles: dict[str, SeriesStyle] | None = None,
    stacked: bool = False,
    baseline: float = 0.0,
    fill_alpha: float | None = None,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    line_spec: LineSpec | None = None,
    shaded_regions: list[ShadedRegionSpec] | None = None,
    annotations: list[AnnotationSpec] | None = None,
) -> Figure | None:
    """
    Function purpose:
        Draw one or more filled area series with optional stacking and annotations.

    Args:
        x: Shared x values for all area series.
        y_series: Mapping from series key to y values.
        xlabel: Text for the x-axis label.
        ylabel: Text for the y-axis label.
        title: Figure title text.
        axis_style: Optional axis styling configuration.
        figure_style: Optional figure-level styling configuration.
        legend_style: Optional legend styling configuration.
        output_config: Optional output saving and return behavior.
        series_styles: Optional per-series visual styles.
        stacked: Whether to stack series cumulatively.
        baseline: Baseline used for non-stacked fills.
        fill_alpha: Optional alpha override for filled areas.
        xlims: Optional x-axis limits.
        ylims: Optional y-axis limits.
        xticks: Optional x-axis tick positions.
        yticks: Optional y-axis tick positions.
        xtick_labels: Optional x-axis tick labels.
        ytick_labels: Optional y-axis tick labels.
        line_spec: Optional vertical and horizontal reference lines.
        shaded_regions: Optional highlighted x-ranges.
        annotations: Optional text annotations.

    Outputs:
        The figure when ``output_config.return_fig`` is true, otherwise ``None``.
    """
    # Instantiate configs at call time so callers can omit boilerplate safely.
    axis_style = axis_style or AxisStyle()
    figure_style = figure_style or FigureStyle()
    legend_style = legend_style or LegendStyle()
    output_config = output_config or OutputConfig()
    series_styles = series_styles or {key: SeriesStyle(label=key) for key in y_series}

    # Disable interactive rendering and apply the requested style preset.
    plt.ioff()
    apply_theme(figure_style, axis_style)

    # Convert x values once because all series share the same x coordinates.
    fig, ax = plt.subplots(figsize=figure_style.figure_size)
    x_values = np.asarray(x)
    bottom = np.full_like(x_values, baseline, dtype=float)

    # Draw each area, stacking values when requested.
    for key, y in y_series.items():
        style = series_styles.get(key, SeriesStyle(label=key))
        y_values = np.asarray(y, dtype=float)
        lower = bottom if stacked else baseline
        upper = bottom + y_values if stacked else y_values
        ax.fill_between(x_values, lower, upper, color=style.color, alpha=style.alpha if fill_alpha is None else fill_alpha)
        ax.plot(x_values, upper, color=style.color, linewidth=style.linewidth, linestyle=style.linestyle)
        if stacked:
            bottom = upper

    # Draw optional shaded regions behind reference lines and annotations.
    if shaded_regions is not None:
        for region in shaded_regions:
            ax.axvspan(region.xmin, region.xmax, color=region.color, alpha=region.alpha, label=region.label)

    # Draw optional vertical and horizontal reference lines.
    if line_spec is not None:
        for spec in line_spec.vertical:
            ax.axvline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)
        for spec in line_spec.horizontal:
            ax.axhline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)

    # Apply shared axis formatting and optional annotations.
    apply_axis_style(ax, xlabel, ylabel, axis_style, xlims=xlims, ylims=ylims, xticks=xticks, yticks=yticks, xtick_labels=xtick_labels, ytick_labels=ytick_labels)
    apply_line_legend(ax, series_styles, legend_style)
    apply_annotations(ax, annotations)
    return finalize_figure(fig, ax, title, figure_style, output_config)
