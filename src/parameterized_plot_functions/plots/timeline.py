"""Timeline and event plot builders."""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from ..axes import apply_axis_style
from ..annotations import apply_annotations
from ..configs import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, LineSpec, OutputConfig, SeriesStyle
from ..legends import apply_patch_legend
from ..saving import finalize_figure
from ..style.themes import apply_theme


def plot_timeline(
    events: dict[str, list[Any]],
    xlabel: str,
    title: str,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    series_styles: dict[str, SeriesStyle] | None = None,
    labels: dict[str, list[str]] | None = None,
    lane_labels: list[str] | None = None,
    marker_size: float = 80.0,
    draw_lane_lines: bool = True,
    label_offset: float = 0.08,
    xlims: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    line_spec: LineSpec | None = None,
    annotations: list[AnnotationSpec] | None = None,
) -> Figure | None:
    """
    Function purpose:
        Draw event markers grouped into horizontal timeline lanes.

    Args:
        events: Mapping from lane name to event x-values.
        xlabel: Text for the x-axis label.
        title: Figure title text.
        axis_style: Optional axis styling configuration.
        figure_style: Optional figure-level styling configuration.
        legend_style: Optional legend styling configuration.
        output_config: Optional output saving and return behavior.
        series_styles: Optional per-lane visual styles.
        labels: Optional text labels for events in each lane.
        lane_labels: Optional display labels for y-axis lanes.
        marker_size: Marker area for timeline events.
        draw_lane_lines: Whether to draw horizontal lane spans.
        label_offset: Vertical offset for event labels.
        xlims: Optional x-axis limits.
        xticks: Optional x-axis tick positions.
        xtick_labels: Optional x-axis tick labels.
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
    series_styles = series_styles or {key: SeriesStyle(label=key) for key in events}

    # Disable interactive rendering and apply the requested style preset.
    plt.ioff()
    apply_theme(figure_style, axis_style)
    fig, ax = plt.subplots(figsize=figure_style.figure_size)
    lane_names = list(events.keys())

    # Draw each event lane as points, optionally connected by a faint line.
    for lane_index, lane in enumerate(lane_names):
        style = series_styles.get(lane, SeriesStyle(label=lane))
        x_values = events[lane]
        y_values = np.full(len(x_values), lane_index)
        ax.scatter(x_values, y_values, color=style.color, marker=style.marker or "o", alpha=style.alpha, s=marker_size)
        if draw_lane_lines and x_values:
            ax.hlines(lane_index, min(x_values), max(x_values), color=style.color, alpha=0.25)

        # Draw optional labels directly above each event marker.
        if labels is not None and lane in labels:
            for x_value, label in zip(x_values, labels[lane]):
                ax.text(x_value, lane_index + label_offset, label, fontsize=max(axis_style.xtick_size - 2, 8), ha="center")

    # Draw optional reference markers after events so they remain visible.
    if line_spec is not None:
        for spec in line_spec.vertical:
            ax.axvline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)
        for spec in line_spec.horizontal:
            ax.axhline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)

    # Apply axis styling with timeline lanes as y tick labels.
    display_lane_labels = lane_labels if lane_labels is not None else lane_names
    apply_axis_style(
        ax,
        xlabel,
        "",
        axis_style,
        xlims=xlims,
        xticks=xticks,
        xtick_labels=xtick_labels,
        yticks=list(range(len(lane_names))),
        ytick_labels=display_lane_labels,
    )
    patch_specs = []
    for lane in lane_names:
        style = series_styles.get(lane, SeriesStyle(label=lane))
        patch_specs.append((style.label or lane, style.color))
    apply_patch_legend(ax, patch_specs, legend_style)
    apply_annotations(ax, annotations)
    return finalize_figure(fig, ax, title, figure_style, output_config)
