"""Example script for grouped bar plots."""

from matplotlib.figure import Figure

from example_helpers import build_output_config, build_series_styles
from parameterized_plot_functions import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, TextStyle, plot_grouped_bar


def create_grouped_bar_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a grouped bar plot with multiple series per category.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Store values as category -> series -> value for grouped rendering.
    values = {
        "Baseline": {"Model A": 0.72, "Model B": 0.81, "Model C": 0.77},
        "Treatment": {"Model A": 0.84, "Model B": 0.88, "Model C": 0.83},
    }

    # Create plot annotations to demonstrate arbitrary text placement.
    annotations = [
        AnnotationSpec(
            text="Treatment improves all models",
            xy=(0.52, 0.93),
            xycoords="axes fraction",
            style=TextStyle(fontsize=13, color="darkgreen", bold=True),
        )
    ]

    # Render grouped bars with value labels and shared series styles.
    return plot_grouped_bar(
        values=values,
        xlabel="Condition",
        ylabel="Score",
        title="Grouped Bar Example",
        series_styles=build_series_styles(["Model A", "Model B", "Model C"]),
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
            ncol=3,
            frameon=True,
            fontsize=13,
            handletextpad=0.6,
            handlelength=1.2,
            bbox_to_anchor=(0.02, 0.98),
        ),
        output_config=build_output_config("grouped_bar_example", return_fig=return_fig),
        annotations=annotations,
        value_labels=True,
        group_gap=1.2,
        bar_width=0.9,
    )


def main() -> None:
    """
    Function purpose:
        Run the grouped bar example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_grouped_bar_figure()


if __name__ == "__main__":
    main()
