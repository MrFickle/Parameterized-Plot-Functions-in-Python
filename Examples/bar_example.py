from pathlib import Path
import sys

import matplotlib
from matplotlib.figure import Figure

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parameterized_plot_functions import (  # noqa: E402
    AnnotationSpec,
    AxisStyle,
    FigureStyle,
    LegendStyle,
    OutputConfig,
    SeriesStyle,
    TextStyle,
    plot_bar,
)


def create_bar_figure(output_config: OutputConfig | None = None) -> Figure | None:
    """
    Function purpose:
        Create a bar plot example figure using the refactored package API.

    Args:
        output_config: Output behavior for saving or returning the figure.

    Outputs:
        Returns a Figure when output_config.return_fig is True; otherwise returns None.
    """
    # Use the standalone example output when no output config is provided.
    if output_config is None:
        output_config = OutputConfig(
            output_dir="showcase_outputs",
            filename="bar_comprehensive_example",
            save_svg=True,
            dpi=300,
        )

    # Define bar heights, error values, x positions, and bar widths.
    values = {"Group A": 4.5, "Group B": 7.2, "Group C": 3.1, "Group D": 8.0}
    sem_values = {"Group A": 0.5, "Group B": 0.8, "Group C": 0.3, "Group D": 0.6}
    x_positions = {"Group A": 1.0, "Group B": 2.0, "Group C": 3.0, "Group D": 4.0}
    bar_widths = {"Group A": 0.7, "Group B": 0.7, "Group C": 0.7, "Group D": 0.7}

    # Configure per-bar style.
    series_styles = {
        "Group A": SeriesStyle(color="#4C72B0", label="Discovery A", alpha=0.8, edgecolor="black"),
        "Group B": SeriesStyle(color="#55A868", label="Inquiry B", alpha=0.8, edgecolor="black"),
        "Group C": SeriesStyle(color="#C44E52", label="Control C", alpha=0.8, edgecolor="black"),
        "Group D": SeriesStyle(color="#8172B2", label="Validation D", alpha=0.8, edgecolor="black"),
    }

    # Create plot-level annotations.
    annotations = [
        AnnotationSpec(text="Highest Yield", xy=(4, 8.5), xycoords="data", style=TextStyle(bold=True, color="darkgreen")),
        AnnotationSpec(text="Baseline", xy=(0.05, 0.4), style=TextStyle(fontsize=12, rotation=45)),
    ]

    # Create the configured bar plot.
    return plot_bar(
        values=values,
        xlabel="Experimental Conditions",
        ylabel="Performance Metric (Units)",
        title="Comprehensive Bar Plot Showcase",
        x_positions=x_positions,
        bar_widths=bar_widths,
        xticks=[1, 2, 3, 4],
        xtick_labels=["A", "B", "C", "D"],
        sem_values=sem_values,
        series_styles=series_styles,
        axis_style=AxisStyle(
            xlabel_size=30,
            ylabel_size=30,
            xtick_size=24,
            ytick_size=24,
            tick_width=2,
            tick_length=15,
            spine_width=2,
            remove_first_ytick=True,
        ),
        figure_style=FigureStyle(
            figure_size=(16, 12),
            title_size=30,
            show_figure=False,
            use_seaborn=True,
            seaborn_style="ticks",
        ),
        legend_style=LegendStyle(
            enabled=True,
            loc="upper left",
            ncol=2,
            fontsize=16,
            frameon=True,
            bbox_to_anchor=(0.02, 0.98),
        ),
        output_config=output_config,
        rotate_xticks=True,
        plot_minor_ticks=True,
        extend_y_one_tick=True,
        annotations=annotations,
    )


def main() -> None:
    """
    Function purpose:
        Save the bar plot example to disk.

    Args:
        None.

    Outputs:
        Saves PNG and SVG files under showcase_outputs.
    """
    # Build and save the standalone example figure.
    create_bar_figure()


if __name__ == "__main__":
    main()
