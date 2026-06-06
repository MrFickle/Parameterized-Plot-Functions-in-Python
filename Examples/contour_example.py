"""Example script for contour plots."""

import numpy as np
from matplotlib.figure import Figure

from example_helpers import build_output_config
from parameterized_plot_functions import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, LineSpec, ReferenceLineSpec, TextStyle, plot_contour


def create_contour_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a filled contour plot from gridded x, y, and z data.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Build deterministic grid coordinates.
    grid = np.linspace(-3, 3, 80)
    x, y = np.meshgrid(grid, grid)

    # Compute a smooth Gaussian-like surface over the grid.
    z = np.exp(-(x**2 + y**2))

    # Create a peak annotation for the contour surface.
    annotations = [
        AnnotationSpec(
            text="Peak",
            xy=(0.0, 0.0),
            xycoords="data",
            style=TextStyle(fontsize=13, color="black", bold=True),
        )
    ]

    # Render a filled contour plot with a labeled colorbar.
    return plot_contour(
        x=x,
        y=y,
        z=z,
        xlabel="x",
        ylabel="y",
        title="Contour Plot Example",
        axis_style=AxisStyle(
            xlabel_size=22,
            ylabel_size=22,
            xtick_size=16,
            ytick_size=16,
            tick_width=2.0,
            tick_length=8.0,
            spine_width=2.0,
            pad_labels=10.0,
            pad_ticks=6.0,
        ),
        figure_style=FigureStyle(
            figure_size=(12, 8),
            title_size=24,
            title_weight="bold",
            tight_layout_pad=0.8,
            show_figure=False,
            use_seaborn=True,
            seaborn_style="ticks",
            seaborn_font_scale=1.2,
            theme="publication",
        ),
        legend_style=LegendStyle(enabled=True, fontsize=13),
        output_config=build_output_config("contour_example", return_fig=return_fig),
        levels=12,
        filled=True,
        cmap="viridis",
        colorbar=True,
        label_contours=False,
        xlims=(-3, 3),
        ylims=(-3, 3),
        xticks=[-3, -2, -1, 0, 1, 2, 3],
        yticks=[-3, -2, -1, 0, 1, 2, 3],
        xtick_labels=["-3", "-2", "-1", "0", "1", "2", "3"],
        ytick_labels=["-3", "-2", "-1", "0", "1", "2", "3"],
        colorbar_label="density",
        linewidths=1.5,
        alpha=0.95,
        vmin=0.0,
        vmax=1.0,
        line_spec=LineSpec(
            vertical=[ReferenceLineSpec(0.0, color="white", linestyle="--", linewidth=1.5, alpha=0.8)],
            horizontal=[ReferenceLineSpec(0.0, color="white", linestyle="--", linewidth=1.5, alpha=0.8)],
        ),
        annotations=annotations,
    )


def main() -> None:
    """
    Function purpose:
        Run the contour plot example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_contour_figure()


if __name__ == "__main__":
    main()
