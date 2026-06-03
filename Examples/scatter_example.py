from pathlib import Path
import sys

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parameterized_plot_functions import (  # noqa: E402
    AxisStyle,
    ColorbarConfig,
    FigureStyle,
    LegendStyle,
    LineSpec,
    OutputConfig,
    ReferenceLineSpec,
    SeriesStyle,
    plot_scatter,
)


def create_scatter_figure(output_config: OutputConfig | None = None) -> Figure | None:
    """
    Function purpose:
        Create a scatter plot example figure using the refactored package API.

    Args:
        output_config: Output behavior for saving or returning the figure.

    Outputs:
        Returns a Figure when output_config.return_fig is True; otherwise returns None.
    """
    # Use the standalone example output when no output config is provided.
    if output_config is None:
        output_config = OutputConfig(
            output_dir="showcase_outputs",
            filename="scatter_comprehensive_example",
        )

    # Use a fixed generator so the clustered data is reproducible.
    rng = np.random.default_rng(42)

    # Build two synthetic clusters.
    x_cluster_1 = rng.normal(2, 0.5, 100)
    y_cluster_1 = rng.normal(2, 0.5, 100)
    x_cluster_2 = rng.normal(5, 0.8, 150)
    y_cluster_2 = rng.normal(5, 0.8, 150)

    # Color only the second cluster by its y-values.
    color_values = {"Cluster 1": None, "Cluster 2": y_cluster_2}

    # Create the configured scatter plot.
    return plot_scatter(
        x_series={"Cluster 1": x_cluster_1, "Cluster 2": x_cluster_2},
        y_series={"Cluster 1": y_cluster_1, "Cluster 2": y_cluster_2},
        xlabel="Dimension X1",
        ylabel="Dimension X2",
        title="Cluster Analysis with Regression and Color Density",
        series_styles={
            "Cluster 1": SeriesStyle(color="teal", marker="o", markersize=10, alpha=0.7, label="Stable Group"),
            "Cluster 2": SeriesStyle(
                color="darkred",
                marker="s",
                markersize=8,
                alpha=0.6,
                label="Dynamic Group",
                m_size_factor=1.5,
            ),
        },
        axis_style=AxisStyle(
            xlabel_size=22,
            ylabel_size=22,
            tick_length=8,
            pad_labels=12,
        ),
        figure_style=FigureStyle(
            figure_size=(16, 12),
            title_size=28,
            use_seaborn=True,
            seaborn_style="white",
            show_figure=False,
        ),
        legend_style=LegendStyle(
            loc="upper left",
            fontsize=16,
        ),
        output_config=output_config,
        do_linear_reg_fit=True,
        plot_r2_score=True,
        colorbar_config=ColorbarConfig(
            enabled=True,
            colormap="viridis",
            label="Density Gradient (Cluster 2)",
            location="right",
        ),
        color_values=color_values,
        line_spec=LineSpec(
            vertical=[ReferenceLineSpec(value=3.5, color="black", linestyle="--", linewidth=1.5)]
        ),
    )


def main() -> None:
    """
    Function purpose:
        Save the scatter plot example to disk.

    Args:
        None.

    Outputs:
        Saves PNG and SVG files under showcase_outputs.
    """
    # Build and save the standalone example figure.
    create_scatter_figure()


if __name__ == "__main__":
    main()
