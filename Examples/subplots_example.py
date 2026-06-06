"""Example script for subplot composition."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from example_helpers import build_output_config, require_figure
from parameterized_plot_functions import FigureStyle, OutputConfig, SeriesStyle, create_subplots_figure, plot_line, plot_scatter


def draw_existing_figure_panel(ax: Axes, source_fig: Figure, title: str) -> None:
    """
    Function purpose:
        Draw an existing plotting-function figure into a subplot axis.

    Args:
        ax: Matplotlib axis to draw on.
        source_fig: Existing figure created by a package plotting function.
        title: Panel title to display above the rendered source figure.

    Outputs:
        None. The axis is modified in place.
    """
    # Render the source figure canvas so it can be copied as an RGBA image.
    source_fig.canvas.draw()

    # Convert the rendered canvas into an image array.
    image = np.array(source_fig.canvas.buffer_rgba())

    # Place the rendered plotting-function output inside the subplot panel.
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)


def create_subplots_example_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a composed subplot figure from panel callbacks.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Build deterministic source data for the line plot.
    x_line = np.linspace(0, 6, 80)
    y_line = np.sin(x_line)

    # Create the first source figure using the existing line plotting function.
    line_fig = require_figure(
        plot_line(
            x_series={"sin": x_line},
            y_series={"sin": y_line},
            xlabel="x",
            ylabel="sin(x)",
            title="Source Line Plot",
            series_styles={"sin": SeriesStyle(color="tab:blue", label="sin(x)")},
            output_config=OutputConfig(return_fig=True),
        ),
        "line subplot source",
    )

    # Build deterministic source data for the scatter plot.
    x_scatter = np.linspace(0, 5, 30)
    y_scatter = x_scatter**2

    # Create the second source figure using the existing scatter plotting function.
    scatter_fig = require_figure(
        plot_scatter(
            x_series={"quadratic": x_scatter},
            y_series={"quadratic": y_scatter},
            xlabel="x",
            ylabel="x^2",
            title="Source Scatter Plot",
            series_styles={"quadratic": SeriesStyle(color="tab:orange", label="x^2")},
            polynomial_degree=2,
            output_config=OutputConfig(return_fig=True),
        ),
        "scatter subplot source",
    )

    try:
        # Render a two-panel figure from callbacks that reuse existing plot outputs.
        return create_subplots_figure(
            plotters=[
                lambda ax: draw_existing_figure_panel(ax, line_fig, "Line function output"),
                lambda ax: draw_existing_figure_panel(ax, scatter_fig, "Scatter function output"),
            ],
            rows=1,
            cols=2,
            figure_style=FigureStyle(figure_size=(10, 4), theme="minimal"),
            output_config=build_output_config("subplots_example", return_fig=return_fig),
            panel_labels=["A", "B"],
            sharex=False,
            sharey=False,
            title="Subplot Composition Example",
            wspace=0.3,
            hspace=0.2,
        )
    finally:
        # Close temporary source figures after they have been copied into the subplot output.
        plt.close(line_fig)
        plt.close(scatter_fig)


def main() -> None:
    """
    Function purpose:
        Run the subplot composition example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_subplots_example_figure()


if __name__ == "__main__":
    main()
