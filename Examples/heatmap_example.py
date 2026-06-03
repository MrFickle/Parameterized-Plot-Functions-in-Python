from pathlib import Path
import sys

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parameterized_plot_functions import AxisStyle, FigureStyle, OutputConfig, plot_heatmap  # noqa: E402


def create_heatmap_figure(output_config: OutputConfig | None = None) -> Figure | None:
    """
    Function purpose:
        Create a heatmap example figure using the refactored package API.

    Args:
        output_config: Output behavior for saving or returning the figure.

    Outputs:
        Returns a Figure when output_config.return_fig is True; otherwise returns None.
    """
    # Use the standalone example output when no output config is provided.
    if output_config is None:
        output_config = OutputConfig(
            output_dir="showcase_outputs",
            filename="heatmap_comprehensive_example",
        )

    # Use a fixed generator so the matrix is reproducible.
    rng = np.random.default_rng(42)

    # Generate synthetic correlation-like data.
    data = rng.random((10, 10))
    data = (data + data.T) / 2
    np.fill_diagonal(data, 1.0)

    # Create the configured heatmap.
    return plot_heatmap(
        data=data,
        xlabel="Features Space X",
        ylabel="Features Space Y",
        title="High-Resolution Thermal Correlation Map",
        axis_style=AxisStyle(
            xlabel_size=20,
            ylabel_size=20,
            xtick_size=12,
            ytick_size=12,
        ),
        figure_style=FigureStyle(
            figure_size=(14, 12),
            title_size=26,
            show_figure=False,
            tight_layout_pad=1.0,
        ),
        output_config=output_config,
        annotate=True,
        colorbar=True,
        vmin=0,
        vmax=1,
        cmap="magma",
        rotate_ticks=True,
    )


def main() -> None:
    """
    Function purpose:
        Save the heatmap example to disk.

    Args:
        None.

    Outputs:
        Saves PNG and SVG files under showcase_outputs.
    """
    # Build and save the standalone example figure.
    create_heatmap_figure()


if __name__ == "__main__":
    main()
