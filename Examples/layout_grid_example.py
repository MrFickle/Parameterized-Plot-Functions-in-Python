from pathlib import Path
import sys

import matplotlib
from matplotlib.figure import Figure

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parameterized_plot_functions import FigureStyle, OutputConfig, draw_figures_grid  # noqa: E402
from bar_example import create_bar_figure  # noqa: E402
from heatmap_example import create_heatmap_figure  # noqa: E402
from hist_example import create_histogram_figure  # noqa: E402
from line_example import create_line_figure  # noqa: E402
from scatter_example import create_scatter_figure  # noqa: E402


def require_figure(fig: Figure | None, name: str) -> Figure:
    """
    Function purpose:
        Validate that an example helper returned a Matplotlib Figure.

    Args:
        fig: Figure returned by an example helper.
        name: Human-readable figure name used in the error message.

    Outputs:
        Returns the validated Figure.
    """
    # Fail clearly if an example helper was called without return_fig=True.
    if fig is None:
        raise ValueError(f"{name} did not return a Figure.")

    return fig


def create_layout_grid_figure(output_config: OutputConfig | None = None) -> Figure | None:
    """
    Function purpose:
        Create one combined figure from the standalone example figures.

    Args:
        output_config: Output behavior for saving or returning the combined figure.

    Outputs:
        Returns a Figure when output_config.return_fig is True; otherwise returns None.
    """
    # Use the standalone layout output when no output config is provided.
    if output_config is None:
        output_config = OutputConfig(
            output_dir="showcase_outputs",
            filename="layout_grid_comprehensive_example",
            dpi=150,
        )

    # Ask each example helper to return its figure instead of saving it.
    return_config = OutputConfig(return_fig=True)
    figures = [
        require_figure(create_line_figure(output_config=return_config), "line"),
        require_figure(create_scatter_figure(output_config=return_config), "scatter"),
        require_figure(create_histogram_figure(output_config=return_config), "histogram"),
        require_figure(create_bar_figure(output_config=return_config), "bar"),
        require_figure(create_heatmap_figure(output_config=return_config), "heatmap"),
    ]

    # Arrange two figures on the first row, two on the second row, and one on the third row.
    figure_rows = [0, 0, 1, 1, 2]

    # Compose and save the combined figure.
    return draw_figures_grid(
        figures=figures,
        figure_rows=figure_rows,
        figure_style=FigureStyle(
            tight_layout_pad=0.2,
            show_figure=False,
        ),
        output_config=output_config,
        grid_hspace=0.15,
    )


def main() -> None:
    """
    Function purpose:
        Save the combined layout grid example to disk.

    Args:
        None.

    Outputs:
        Saves PNG and SVG files under showcase_outputs.
    """
    # Build and save the combined grid example.
    create_layout_grid_figure()


if __name__ == "__main__":
    main()
