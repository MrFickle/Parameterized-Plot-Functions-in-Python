"""Example script for pie and donut charts."""

from matplotlib.figure import Figure

from example_helpers import build_output_config, build_series_styles
from parameterized_plot_functions import AnnotationSpec, FigureStyle, LegendStyle, TextStyle, plot_pie


def create_pie_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a donut-style pie chart.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Store scalar category contributions.
    values = {"A": 30, "B": 45, "C": 25}

    # Create a center annotation for the donut chart.
    annotations = [
        AnnotationSpec(
            text="Total\n100%",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            style=TextStyle(fontsize=16, color="black", bold=True),
        )
    ]

    # Render a donut chart with an exploded slice and legend.
    return plot_pie(
        values=values,
        title="Pie Plot Example",
        figure_style=FigureStyle(
            figure_size=(10, 8),
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
            ncol=1,
            frameon=True,
            fontsize=13,
            handletextpad=0.6,
            handlelength=1.2,
        ),
        output_config=build_output_config("pie_example", return_fig=return_fig),
        series_styles=build_series_styles(["A", "B", "C"]),
        annotations=annotations,
        autopct="%1.1f%%",
        startangle=90,
        donut_width=0.45,
        explode=[0.0, 0.05, 0.0],
        shadow=True,
        labeldistance=1.08,
        pctdistance=0.75,
        counterclock=True,
        normalize=True,
        textprops={"fontsize": 12, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        show_legend=True,
        legend_loc="upper right",
    )


def main() -> None:
    """
    Function purpose:
        Run the pie plot example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_pie_figure()


if __name__ == "__main__":
    main()
