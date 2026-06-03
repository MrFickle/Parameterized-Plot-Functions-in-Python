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
    FigureStyle,
    LegendStyle,
    LineSpec,
    OutputConfig,
    ReferenceLineSpec,
    SeriesStyle,
    plot_line,
)


def create_line_figure(output_config: OutputConfig | None = None) -> Figure | None:
    """
    Function purpose:
        Create a line plot example figure using the refactored package API.

    Args:
        output_config: Output behavior for saving or returning the figure.

    Outputs:
        Returns a Figure when output_config.return_fig is True; otherwise returns None.
    """
    # Use the standalone example output when no output config is provided.
    if output_config is None:
        output_config = OutputConfig(
            output_dir="showcase_outputs",
            filename="line_comprehensive_example",
        )

    # Use a fixed generator so the example is reproducible.
    rng = np.random.default_rng(42)

    # Build two trigonometric series and matching uncertainty envelopes.
    x = np.linspace(0, 10, 50)
    y_sine = np.sin(x)
    y_cosine = np.cos(x)
    y_sine_err = rng.random(50) * 0.2
    y_cosine_err = rng.random(50) * 0.2

    # Create the configured line plot.
    return plot_line(
        x_series={"Sine": x, "Cosine": x},
        y_series={"Sine": y_sine, "Cosine": y_cosine},
        xlabel="Time (s)",
        ylabel="Amplitude (Voltage)",
        title="Trigonometric Time-Series with Error Envelopes",
        series_styles={
            "Sine": SeriesStyle(color="blue", linewidth=4, label="Primary Phase", marker="o", markersize=6),
            "Cosine": SeriesStyle(
                color="orange",
                linewidth=4,
                linestyle="--",
                label="Secondary Phase",
                marker="x",
                markersize=8,
            ),
        },
        axis_style=AxisStyle(
            xlabel_size=20,
            ylabel_size=20,
            tick_width=2.5,
            spine_width=3,
            use_log_x=False,
        ),
        figure_style=FigureStyle(
            figure_size=(14, 8),
            title_size=24,
            show_figure=False,
        ),
        legend_style=LegendStyle(
            loc="lower left",
            ncol=2,
            fontsize=14,
            labelcolor="linecolor",
        ),
        output_config=output_config,
        ylims=(-1.5, 1.5),
        yerr_series={"Sine": y_sine_err, "Cosine": y_cosine_err},
        line_spec=LineSpec(
            horizontal=[ReferenceLineSpec(value=0, color="gray", alpha=0.5, linestyle="-")]
        ),
        rotate_xticks=False,
        plot_minor_ticks=True,
        extend_y_one_tick=False,
    )


def main() -> None:
    """
    Function purpose:
        Save the line plot example to disk.

    Args:
        None.

    Outputs:
        Saves PNG and SVG files under showcase_outputs.
    """
    # Build and save the standalone example figure.
    create_line_figure()


if __name__ == "__main__":
    main()
