"""Heatmap plot builder for matrix-like data."""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns

from ..annotations import apply_annotations
from ..configs import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, OutputConfig
from ..saving import finalize_figure
from ..style.themes import apply_theme


def plot_heatmap(
    data: Any,
    xlabel: str,
    ylabel: str,
    title: str,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    annotations: list[AnnotationSpec] | None = None,
    annotate: bool = True,
    colorbar: bool = False,
    vmin=None,
    vmax=None,
    cmap: str = "rocket",
    rotate_ticks: bool = False,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    triangular_mask: str | None = None,
    center: float | None = None,
    normalize: str | None = None,
    auto_text_contrast: bool = False,
) -> Figure | None:
    """
    Function purpose:
        Draw a seaborn heatmap for matrix-like data and apply standard figure
        finalization.

    Args:
        data: Matrix-like data accepted by ``seaborn.heatmap``.
        xlabel: Text for the x-axis label.
        ylabel: Text for the y-axis label.
        title: Figure title text.
        axis_style: Optional axis styling configuration.
        figure_style: Optional figure-level styling and display configuration.
        legend_style: Optional colorbar styling configuration.
        output_config: Optional output saving and return behavior configuration.
        annotations: Optional annotations to draw on the axis.
        annotate: Whether to display numeric values inside heatmap cells.
        colorbar: Whether to display a heatmap colorbar.
        vmin: Optional lower bound for color scaling.
        vmax: Optional upper bound for color scaling.
        cmap: Colormap name passed to seaborn.
        rotate_ticks: Whether to rotate x tick labels by 90 degrees.

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

    # Disable interactive rendering for batch/script usage.
    plt.ioff()
    apply_theme(figure_style, axis_style)

    fig, ax = plt.subplots(figsize=figure_style.figure_size)
    heatmap_data = np.asarray(data, dtype=float)
    if normalize == "row":
        denom = np.nanmax(heatmap_data, axis=1, keepdims=True) - np.nanmin(heatmap_data, axis=1, keepdims=True)
        heatmap_data = (heatmap_data - np.nanmin(heatmap_data, axis=1, keepdims=True)) / np.where(denom == 0, 1, denom)
    elif normalize == "column":
        denom = np.nanmax(heatmap_data, axis=0, keepdims=True) - np.nanmin(heatmap_data, axis=0, keepdims=True)
        heatmap_data = (heatmap_data - np.nanmin(heatmap_data, axis=0, keepdims=True)) / np.where(denom == 0, 1, denom)
    elif normalize == "global":
        denom = np.nanmax(heatmap_data) - np.nanmin(heatmap_data)
        heatmap_data = (heatmap_data - np.nanmin(heatmap_data)) / (denom if denom != 0 else 1)

    mask = None
    if triangular_mask == "upper":
        mask = np.triu(np.ones_like(heatmap_data, dtype=bool), k=1)
    elif triangular_mask == "lower":
        mask = np.tril(np.ones_like(heatmap_data, dtype=bool), k=-1)

    # Seaborn handles matrix rendering while matplotlib handles final styling.
    sns.heatmap(
        heatmap_data,
        annot=annotate,
        fmt=".2f" if annotate else "",
        cbar=colorbar and legend_style.enabled,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        center=center,
        mask=mask,
        ax=ax,
        annot_kws={"fontsize": axis_style.xtick_size},
        cbar_kws={"pad": 0.01} if colorbar and legend_style.enabled else None,
    )

    if colorbar and legend_style.enabled and len(fig.axes) > 1:
        fig.axes[-1].tick_params(labelsize=legend_style.fontsize)

    # Heatmap axes use direct label styling instead of the generic axis helper.
    ax.set_xlabel(xlabel, fontsize=axis_style.xlabel_size, fontweight="bold", labelpad=20)
    ax.set_ylabel(ylabel, fontsize=axis_style.ylabel_size, fontweight="bold", labelpad=20)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(axis_style.xtick_size)

    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels)

    if rotate_ticks:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
        for label in ax.get_yticklabels():
            label.set_rotation(0)

    if auto_text_contrast and annotate:
        threshold = np.nanmean(heatmap_data)
        for text, value in zip(ax.texts, heatmap_data[~mask].flat if mask is not None else heatmap_data.flat):
            text.set_color("white" if value > threshold else "black")

    apply_annotations(ax, annotations)
    return finalize_figure(fig, ax, title, figure_style, output_config)


def plot_correlation_heatmap(
    data: Any,
    title: str = "Correlation Heatmap",
    labels: list[str] | None = None,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    annotations: list[AnnotationSpec] | None = None,
    annotate: bool = True,
    colorbar: bool = True,
    triangular_mask: str | None = "upper",
    cmap: str = "vlag",
) -> Figure | None:
    """Draw a correlation matrix heatmap from observations or a matrix."""
    values = np.asarray(data, dtype=float)
    corr = values if values.ndim == 2 and values.shape[0] == values.shape[1] else np.corrcoef(values, rowvar=False)
    fig = plot_heatmap(
        corr,
        xlabel="",
        ylabel="",
        title=title,
        axis_style=axis_style,
        figure_style=figure_style,
        legend_style=legend_style,
        output_config=output_config,
        annotations=annotations,
        annotate=annotate,
        colorbar=colorbar,
        vmin=-1,
        vmax=1,
        cmap=cmap,
        xtick_labels=labels,
        ytick_labels=labels,
        triangular_mask=triangular_mask,
        center=0,
        auto_text_contrast=True,
    )
    return fig
