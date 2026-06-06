"""Example script for stacked bar plots."""

from matplotlib.figure import Figure

from example_helpers import build_output_config, build_series_styles
from parameterized_plot_functions import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, TextStyle, plot_stacked_bar


def create_stacked_bar_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a normalized stacked bar plot from category-by-series values.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Store composition values as category -> series -> value.
    values = {
        "Sample 1": {"A": 30, "B": 15, "C": 5},
        "Sample 2": {"A": 18, "B": 26, "C": 14},
        "Sample 3": {"A": 10, "B": 20, "C": 30},
    }

    # Create an annotation to call out the normalized-stack interpretation.
    annotations = [
        AnnotationSpec(
            text="Each stack sums to 1.0",
            xy=(0.03, 0.92),
            xycoords="axes fraction",
            style=TextStyle(fontsize=13, color="black", bold=True),
        )
    ]

    # Render normalized bars so each stack sums to one.
    return plot_stacked_bar(
        values=values,
        xlabel="Sample",
        ylabel="Proportion",
        title="Stacked Bar Example",
        series_styles=build_series_styles(["A", "B", "C"]),
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
            loc="upper right",
            ncol=3,
            frameon=True,
            fontsize=13,
            handletextpad=0.6,
            handlelength=1.2,
        ),
        output_config=build_output_config("stacked_bar_example", return_fig=return_fig),
        annotations=annotations,
        normalize=True,
        value_labels=True,
    )


def main() -> None:
    """
    Function purpose:
        Run the stacked bar example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_stacked_bar_figure()


if __name__ == "__main__":
    main()
