"""Example script for hexbin density plots."""

import numpy as np
from matplotlib.figure import Figure

from example_helpers import build_output_config
from parameterized_plot_functions import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, LineSpec, ReferenceLineSpec, TextStyle, plot_hexbin


def create_hexbin_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a hexbin density plot from dense scatter data.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Generate deterministic correlated observations.
    rng = np.random.default_rng(9)
    x = rng.normal(0, 1, 600)
    y = 0.5 * x + rng.normal(0, 0.7, 600)

    # Create an annotation and reference lines for density orientation.
    annotations = [
        AnnotationSpec(
            text="Dense center",
            xy=(0.08, 0.9),
            xycoords="axes fraction",
            style=TextStyle(fontsize=13, color="black", bold=True),
        )
    ]

    # Render hexagonal bins with logarithmic count scaling.
    return plot_hexbin(
        x=x,
        y=y,
        xlabel="x",
        ylabel="y",
        title="Hexbin Plot Example",
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
        legend_style=LegendStyle(enabled=True, fontsize=13),
        output_config=build_output_config("hexbin_example", return_fig=return_fig),
        gridsize=28,
        cmap="viridis",
        mincnt=1,
        colorbar=True,
        reduce_function=np.mean,
        values=None,
        xlims=(-3.5, 3.5),
        ylims=(-3.5, 3.5),
        xticks=[-3, -2, -1, 0, 1, 2, 3],
        yticks=[-3, -2, -1, 0, 1, 2, 3],
        xtick_labels=["-3", "-2", "-1", "0", "1", "2", "3"],
        ytick_labels=["-3", "-2", "-1", "0", "1", "2", "3"],
        bins="log",
        colorbar_label="log(count)",
        extent=(-3.5, 3.5, -3.5, 3.5),
        linewidths=0.1,
        alpha=0.95,
        line_spec=LineSpec(
            vertical=[ReferenceLineSpec(0.0, color="white", linestyle="--", linewidth=1.5, alpha=0.9)],
            horizontal=[ReferenceLineSpec(0.0, color="white", linestyle="--", linewidth=1.5, alpha=0.9)],
        ),
        annotations=annotations,
    )


def main() -> None:
    """
    Function purpose:
        Run the hexbin plot example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_hexbin_figure()


if __name__ == "__main__":
    main()
