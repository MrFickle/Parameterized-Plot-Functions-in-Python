import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .configs import AnnotationSpec, FigureStyle, OutputConfig
from .annotations import apply_annotations
from .saving import save_figure


def create_empty_figure(figure_size: tuple[float, float] = (10, 8)) -> Figure:
    return plt.figure(figsize=figure_size)


def draw_figures_grid(
    figures: list[Figure],
    figure_rows: list[int] | np.ndarray,
    figure_style: FigureStyle | None = None,
    output_config: OutputConfig | None = None,
    annotations: list[AnnotationSpec] | None = None,
    grid_hspace: float = 0.0,
):
    if figure_style is None:
        figure_style = FigureStyle()
    if output_config is None:
        output_config = OutputConfig()

    total_figs = len(figures)
    figure_rows = np.asarray(figure_rows)
    unique_rows = np.unique(figure_rows)
    total_rows = len(unique_rows)

    # Get the sizes of each figure
    figure_widths = [int(fig.get_size_inches()[0]) for fig in figures]
    figure_heights = [int(fig.get_size_inches()[1]) for fig in figures]

    # Map rows to figures
    row_to_fig_indices = {row: np.where(figure_rows == row)[0] for row in unique_rows}
    row_widths = {row: np.sum([figure_widths[i] for i in row_to_fig_indices[row]]) for row in unique_rows}
    max_row_width = int(np.max(list(row_widths.values())))

    # Grid dimensions
    grid_cols = max_row_width
    # Note: image_height in legacy used a specific formula. We'll use a simpler one or match if needed.
    # Legacy: figure_heights['F0'] * total_rows + total_rows + 1
    image_height = figure_heights[0] * total_rows
    image_width = grid_cols

    # Render canvases
    canvases = []
    for fig in figures:
        fig.canvas.draw()
        canvases.append(np.array(fig.canvas.buffer_rgba()))

    combined_fig = plt.figure(figsize=(image_width, image_height))
    grid = gridspec.GridSpec(nrows=total_rows, ncols=grid_cols)
    
    overlay_ax = combined_fig.add_subplot(111)
    overlay_ax.axis("off")

    for row_idx, row in enumerate(unique_rows):
        indices = row_to_fig_indices[row]
        current_row_width = row_widths[row]
        num_figs_in_row = len(indices)
        
        # Calculate spacing if multiple figures in row
        wspace_cols = 0
        if num_figs_in_row > 1:
            wspace_cols = int((grid_cols - current_row_width) / (num_figs_in_row - 1))
        
        current_col = 0
        for i in indices:
            fig_w = figure_widths[i]
            ax = combined_fig.add_subplot(grid[row_idx, current_col : (current_col + fig_w)])
            ax.imshow(canvases[i])
            ax.axis("off")
            current_col += fig_w + wspace_cols

    apply_annotations(overlay_ax, annotations)
    combined_fig.tight_layout(pad=figure_style.tight_layout_pad)

    if not output_config.return_fig:
        save_figure(combined_fig, output_config)

    if figure_style.show_figure:
        plt.show()
    else:
        plt.close(combined_fig)

    return combined_fig if output_config.return_fig else None