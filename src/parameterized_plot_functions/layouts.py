"""Helpers for creating figures and composing multiple figures into a grid."""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .configs import AnnotationSpec, FigureStyle, OutputConfig
from .annotations import apply_annotations
from .saving import save_figure


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
