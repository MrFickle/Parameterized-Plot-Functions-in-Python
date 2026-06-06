"""Example script for area plots."""

import numpy as np
from matplotlib.figure import Figure

from example_helpers import build_output_config, build_series_styles
from parameterized_plot_functions import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, LineSpec, ReferenceLineSpec, ShadedRegionSpec, TextStyle, plot_area


def create_area_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a stacked area plot with a shaded region and reference line.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Build deterministic x values and two positive series.
    x = np.linspace(0, 10, 80)
    y_series = {"A": np.sin(x) + 1.5, "B": 0.5 * np.cos(x) + 1.0}

    # Create annotations for the stacked area plot.
    annotations = [
        AnnotationSpec(
            text="Highlighted interval",
            xy=(3.0, 3.4),
            xycoords="data",
            style=TextStyle(fontsize=13, color="black", bold=True),
        )
    ]

    # Render a stacked area plot with shared style and reference controls.
    return plot_area(
        x=x,
        y_series=y_series,
        xlabel="x",
        ylabel="Value",
        title="Area Plot Example",
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
        legend_style=LegendStyle(
            enabled=True,
            loc="upper left",
            ncol=2,
            frameon=True,
            fontsize=13,
            handletextpad=0.6,
            handlelength=1.2,
        ),
        output_config=build_output_config("area_example", return_fig=return_fig),
        series_styles=build_series_styles(["A", "B"]),
        stacked=True,
        baseline=0.0,
        fill_alpha=0.45,
        xlims=(0, 10),
        ylims=(0, 4.5),
        xticks=[0, 2, 4, 6, 8, 10],
        yticks=[0, 1, 2, 3, 4],
        xtick_labels=["0", "2", "4", "6", "8", "10"],
        ytick_labels=["0", "1", "2", "3", "4"],
        line_spec=LineSpec(
            vertical=[ReferenceLineSpec(5.0, color="tab:red", linestyle="--", linewidth=2.0, alpha=0.8)],
            horizontal=[ReferenceLineSpec(3.0, color="black", linestyle=":", linewidth=1.5, alpha=0.8)],
        ),
        shaded_regions=[ShadedRegionSpec(2, 4, color="tab:gray", alpha=0.15)],
        annotations=annotations,
    )


def main() -> None:
    """
    Function purpose:
        Run the area plot example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_area_figure()


if __name__ == "__main__":
    main()
