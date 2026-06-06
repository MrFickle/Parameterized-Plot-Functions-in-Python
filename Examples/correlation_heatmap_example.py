"""Example script for correlation heatmaps."""

import numpy as np
from matplotlib.figure import Figure

from example_helpers import build_output_config
from parameterized_plot_functions import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, TextStyle, plot_correlation_heatmap


def create_correlation_heatmap_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a correlation heatmap from observation data.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Store rows as observations and columns as variables.
    observations = np.array([[1, 2, 3], [2, 3, 5], [3, 5, 8], [4, 6, 9], [5, 8, 13]])

    # Create an annotation positioned in axis-fraction coordinates.
    annotations = [
        AnnotationSpec(
            text="Upper triangle masked",
            xy=(0.04, 0.94),
            xycoords="axes fraction",
            style=TextStyle(fontsize=12, color="black", bold=True),
        )
    ]

    # Render the correlation matrix with labels and an upper-triangle mask.
    return plot_correlation_heatmap(
        data=observations,
        title="Correlation Heatmap Example",
        labels=["x1", "x2", "x3"],
        axis_style=AxisStyle(
            xlabel_size=20,
            ylabel_size=20,
            xtick_size=14,
            ytick_size=14,
            tick_width=1.5,
            tick_length=6.0,
            spine_width=1.5,
            pad_labels=8.0,
            pad_ticks=5.0,
        ),
        figure_style=FigureStyle(
            figure_size=(9, 8),
            title_size=22,
            title_weight="bold",
            tight_layout_pad=0.7,
            show_figure=False,
            use_seaborn=True,
            seaborn_style="ticks",
            seaborn_font_scale=1.1,
            theme="publication",
        ),
        legend_style=LegendStyle(enabled=True, fontsize=12),
        output_config=build_output_config("correlation_heatmap_example", return_fig=return_fig),
        annotations=annotations,
        annotate=True,
        colorbar=True,
        triangular_mask="upper",
        cmap="vlag",
    )


def main() -> None:
    """
    Function purpose:
        Run the correlation heatmap example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_correlation_heatmap_figure()


if __name__ == "__main__":
    main()
