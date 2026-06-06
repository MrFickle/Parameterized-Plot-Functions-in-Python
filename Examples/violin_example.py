"""Example script for violin plots."""

import numpy as np
from matplotlib.figure import Figure

from example_helpers import build_output_config, build_series_styles
from parameterized_plot_functions import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, TextStyle, plot_violin


def create_violin_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a violin plot for grouped distributions.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Generate deterministic grouped samples.
    rng = np.random.default_rng(11)
    data = {"A": rng.normal(0, 1, 150), "B": rng.normal(1, 1.1, 150), "C": rng.normal(-0.5, 0.9, 150)}

    # Create a text annotation for quantile-line behavior.
    annotations = [
        AnnotationSpec(
            text="Quartile lines enabled",
            xy=(0.04, 0.92),
            xycoords="axes fraction",
            style=TextStyle(fontsize=13, color="black", bold=True),
        )
    ]

    # Render violins with quantile markers for each group.
    return plot_violin(
        data_series=data,
        xlabel="Group",
        ylabel="Value",
        title="Violin Plot Example",
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
        output_config=build_output_config("violin_example", return_fig=return_fig),
        series_styles=build_series_styles(["A", "B", "C"]),
        show_means=True,
        show_extrema=True,
        show_medians=True,
        orientation="vertical",
        positions=[1.0, 2.0, 3.0],
        tick_labels=["Group A", "Group B", "Group C"],
        widths=0.75,
        violin_alpha=0.72,
        quantiles=[[0.25, 0.5, 0.75], [0.25, 0.5, 0.75], [0.25, 0.5, 0.75]],
        grid_axis="y",
        xlims=(0.4, 3.6),
        ylims=(-3.5, 4.5),
        annotations=annotations,
    )


def main() -> None:
    """
    Function purpose:
        Run the violin plot example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_violin_figure()


if __name__ == "__main__":
    main()
