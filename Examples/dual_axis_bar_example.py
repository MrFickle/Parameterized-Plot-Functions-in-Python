"""Example script for dual-axis bar plots."""

from matplotlib.figure import Figure

from example_helpers import build_output_config, build_series_styles
from parameterized_plot_functions import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, TextStyle, plot_dual_axis_bar


def create_dual_axis_bar_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a bar plot with separate left and right y-axis scales.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Use deliberately different scales to demonstrate the dual-axis behavior.
    values = {"Error": 0.18, "Throughput": 125.0}

    # Create an axis-fraction annotation for the primary axis.
    annotations = [
        AnnotationSpec(
            text="Two independent y scales",
            xy=(0.03, 0.92),
            xycoords="axes fraction",
            style=TextStyle(fontsize=13, color="black", bold=True),
        )
    ]

    # Render each bar on its assigned y-axis.
    return plot_dual_axis_bar(
        values=values,
        xlabel="Metric",
        ylabel_left="Error rate",
        ylabel_right="Throughput",
        title="Dual Axis Bar Example",
        x_positions={"Error": 0, "Throughput": 1},
        bar_widths={"Error": 0.5, "Throughput": 0.5},
        axis_assignment={"Error": "left", "Throughput": "right"},
        series_styles=build_series_styles(["Error", "Throughput"]),
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
            bbox_to_anchor=(0.02, 0.98),
        ),
        output_config=build_output_config("dual_axis_bar_example", return_fig=return_fig),
        xticks=[0, 1],
        xtick_labels=["Error", "Throughput"],
        yticks_left=[0.0, 0.1, 0.2, 0.3],
        yticks_right=[0, 50, 100, 150],
        ylims_left=(0.0, 0.3),
        ylims_right=(0, 150),
        sem_values={"Error": 0.02, "Throughput": 8.0},
        edgecolor="black",
        rotate_xticks=False,
        plot_minor_ticks=True,
        extend_y_one_tick=False,
        annotations=annotations,
    )


def main() -> None:
    """
    Function purpose:
        Run the dual-axis bar example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_dual_axis_bar_figure()


if __name__ == "__main__":
    main()
