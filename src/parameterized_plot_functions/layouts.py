"""Helpers for creating figures and composing multiple figures into a grid."""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Callable

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .configs import AnnotationSpec, FigureStyle, OutputConfig
from .annotations import apply_annotations
from .saving import save_figure
from .style.themes import apply_theme


def create_empty_figure(figure_size: tuple[float, float] = (10, 8)) -> Figure:
    """
    Function purpose:
        Create an empty matplotlib figure with a specific size.

    Args:
        figure_size: Figure size in inches as ``(width, height)``.

    Outputs:
        A new matplotlib figure.
    """
    return plt.figure(figsize=figure_size)


def draw_figures_grid(
    figures: list[Figure],
    figure_rows: list[int] | np.ndarray,
    figure_style: FigureStyle | None = None,
    output_config: OutputConfig | None = None,
    annotations: list[AnnotationSpec] | None = None,
    grid_hspace: float = 0.0,
) -> Figure | None:
    """
    Function purpose:
        Render multiple existing figures into a single combined grid figure.

    Args:
        figures: Source figures to render into the combined output.
        figure_rows: Row assignment for each source figure.
        figure_style: Optional figure-level styling and display configuration.
        output_config: Optional output saving and return behavior configuration.
        annotations: Optional annotations drawn on a transparent overlay axis.
        grid_hspace: Vertical spacing between grid rows.

    Outputs:
        The combined figure when ``output_config.return_fig`` is true,
        otherwise ``None``.
    """
    if figure_style is None:
        figure_style = FigureStyle()
    if output_config is None:
        output_config = OutputConfig()

    if not figures:
        raise ValueError("figures must contain at least one Figure.")

    figure_rows = np.asarray(figure_rows)
    if len(figure_rows) != len(figures):
        raise ValueError("figure_rows must have the same length as figures.")

    unique_rows = np.unique(figure_rows)

    # Convert figure widths into integer grid units while preserving relative size.
    grid_scale = 10
    figure_sizes = [fig.get_size_inches() for fig in figures]
    figure_width_units = [max(1, int(np.ceil(size[0] * grid_scale))) for size in figure_sizes]
    figure_heights = [float(size[1]) for size in figure_sizes]

    # Group figures by their target row.
    row_to_fig_indices = {row: np.where(figure_rows == row)[0] for row in unique_rows}
    row_width_units = {
        row: int(np.sum([figure_width_units[i] for i in row_to_fig_indices[row]]))
        for row in unique_rows
    }
    row_heights = {
        row: max([figure_heights[i] for i in row_to_fig_indices[row]])
        for row in unique_rows
    }

    # Size the combined figure from the widest row and the sum of row heights.
    grid_cols = max(row_width_units.values())
    image_width = grid_cols / grid_scale
    image_height = sum(row_heights.values()) + grid_hspace * max(0, len(unique_rows) - 1)

    # Render each source figure to an RGBA canvas for placement in the combined figure.
    canvases = []
    for fig in figures:
        fig.canvas.draw()
        canvases.append(np.array(fig.canvas.buffer_rgba()))

    combined_fig = plt.figure(figsize=(image_width, image_height))
    grid = gridspec.GridSpec(
        nrows=len(unique_rows),
        ncols=grid_cols,
        height_ratios=[row_heights[row] for row in unique_rows],
        hspace=grid_hspace,
    )

    for row_idx, row in enumerate(unique_rows):
        indices = row_to_fig_indices[row]
        current_row_width = row_width_units[row]
        num_figs_in_row = len(indices)

        # Spread figures across the row if the row is narrower than the full grid.
        wspace_cols = 0
        if num_figs_in_row > 1:
            wspace_cols = int((grid_cols - current_row_width) / (num_figs_in_row - 1))

        current_col = 0
        for i in indices:
            fig_w = figure_width_units[i]
            ax = combined_fig.add_subplot(grid[row_idx, current_col : (current_col + fig_w)])
            ax.imshow(canvases[i])
            ax.axis("off")
            current_col += fig_w + wspace_cols

    # Fill the combined canvas; source figures already own their internal layout.
    combined_fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1,
        wspace=0,
        hspace=grid_hspace,
    )

    # Use a transparent overlay axis for optional figure-level annotations.
    overlay_ax = combined_fig.add_subplot(111, frameon=False)
    overlay_ax.patch.set_alpha(0.0)
    overlay_ax.axis("off")
    apply_annotations(overlay_ax, annotations)

    if not output_config.return_fig:
        save_figure(combined_fig, output_config)

    if figure_style.show_figure:
        plt.show()
    else:
        plt.close(combined_fig)

    return combined_fig if output_config.return_fig else None


def create_subplots_figure(
    plotters: list[Callable[[Axes], None]],
    rows: int,
    cols: int,
    figure_style: FigureStyle | None = None,
    output_config: OutputConfig | None = None,
    panel_labels: list[str] | None = None,
    sharex: bool = False,
    sharey: bool = False,
    title: str | None = None,
    wspace: float | None = None,
    hspace: float | None = None,
) -> Figure | None:
    """
    Function purpose:
        Create a multi-panel subplot figure from callbacks that draw on provided axes.

    Args:
        plotters: Functions that receive one Matplotlib axis and draw one panel.
        rows: Number of subplot rows.
        cols: Number of subplot columns.
        figure_style: Optional figure-level styling configuration.
        output_config: Optional output saving and return behavior.
        panel_labels: Optional panel labels such as A, B, C.
        sharex: Whether subplot panels share their x-axis.
        sharey: Whether subplot panels share their y-axis.
        title: Optional figure-level title.
        wspace: Optional width spacing between subplots.
        hspace: Optional height spacing between subplots.

    Outputs:
        The figure when ``output_config.return_fig`` is true, otherwise ``None``.
    """
    # Instantiate configs at call time so callers can omit boilerplate safely.
    if figure_style is None:
        figure_style = FigureStyle()
    if output_config is None:
        output_config = OutputConfig()

    # Guard against silently dropping requested panels.
    if len(plotters) > rows * cols:
        raise ValueError("plotters cannot exceed rows * cols.")

    # Disable interactive rendering and apply the requested style preset.
    plt.ioff()
    apply_theme(figure_style)

    # Create the target subplot grid and flatten it for simple sequential filling.
    fig, axes = plt.subplots(rows, cols, figsize=figure_style.figure_size, sharex=sharex, sharey=sharey)
    axes_array = np.asarray(axes).reshape(-1)

    # Let each callback draw into its assigned axis.
    for index, ax in enumerate(axes_array):
        if index < len(plotters):
            plotters[index](ax)
            if panel_labels is not None and index < len(panel_labels):
                ax.text(0.0, 1.02, panel_labels[index], transform=ax.transAxes, fontweight="bold", va="bottom")
        else:
            ax.axis("off")

    # Apply optional figure title and spacing controls.
    if title is not None:
        fig.suptitle(title, fontsize=figure_style.title_size, fontweight=figure_style.title_weight)
    if wspace is not None or hspace is not None:
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # Finalize layout, save outputs, and close or return the figure.
    fig.tight_layout(pad=figure_style.tight_layout_pad)
    save_figure(fig, output_config)

    if figure_style.show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig if output_config.return_fig else None
