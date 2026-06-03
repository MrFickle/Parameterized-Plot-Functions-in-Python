from pathlib import Path
import sys

import matplotlib
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parameterized_plot_functions import (  # noqa: E402
    AnnotationSpec,
    AxisStyle,
    FigureStyle,
    LegendStyle,
    LineSpec,
    OutputConfig,
    ReferenceLineSpec,
    SeriesStyle,
    TextStyle,
    plot_histogram,
)


def create_histogram_figure(output_config: OutputConfig | None = None) -> Figure | None:
    """
    Function purpose:
        Create a histogram example figure using the refactored package API.

    Args:
        output_config: Output behavior for saving or returning the figure.

    Outputs:
        Returns a Figure when output_config.return_fig is True; otherwise returns None.
    """
    # Use the standalone example output when no output config is provided.
    if output_config is None:
        output_config = OutputConfig(
            output_dir="showcase_outputs",
            filename="hist_comprehensive_example",
        )

    # Use a fixed generator so the distributions are reproducible.
    rng = np.random.default_rng(42)

    # Generate overlapping control and treatment distributions.
    control = rng.normal(0, 1, 1000)
    treatment = rng.normal(2, 1.5, 1200)

    # Create the configured histogram.
    return plot_histogram(
        data_series={
            "Control": control,
            "Treatment": treatment,
        },
        bins=40,
        xlabel="Observed Value",
        ylabel="Density / Probability",
        title="Distribution Analysis with Statistical Overlays",
        series_styles={
            "Control": SeriesStyle(color="steelblue", alpha=0.6, label="Control Group (N=1k)"),
            "Treatment": SeriesStyle(color="indianred", alpha=0.6, label="Treatment Group (N=1.2k)"),
        },
        axis_style=AxisStyle(
            xlabel_size=22,
            ylabel_size=22,
            remove_first_ytick=True,
            spine_width=3,
        ),
        figure_style=FigureStyle(
            figure_size=(16, 10),
            use_seaborn=True,
            seaborn_style="ticks",
            show_figure=False,
        ),
        legend_style=LegendStyle(
            loc="upper right",
            ncol=1,
            fontsize=16,
        ),
        output_config=output_config,
        plot_mean=True,
        plot_std=True,
        plot_kde=True,
        hist_stat="density",
        xlims=(-5, 8),
        line_spec=LineSpec(
            vertical=[
                ReferenceLineSpec(value=0, color="blue", linestyle=":", linewidth=2),
                ReferenceLineSpec(value=2, color="red", linestyle="--", linewidth=2),
            ]
        ),
        annotations=[
            AnnotationSpec(text="Significant Overlap", xy=(0.3, 0.5), style=TextStyle(bold=True, color="purple"))
        ],
        extend_y_one_tick=True,
    )


def main() -> None:
    """
    Function purpose:
        Save the histogram example to disk.

    Args:
        None.

    Outputs:
        Saves PNG and SVG files under showcase_outputs.
    """
    # Build and save the standalone example figure.
    create_histogram_figure()


if __name__ == "__main__":
    main()
