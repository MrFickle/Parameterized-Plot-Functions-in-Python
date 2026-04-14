import matplotlib.pyplot as plt
import seaborn as sns

from ..configs import AxisStyle, FigureStyle, OutputConfig
from ..saving import finalize_figure


def plot_heatmap(
    data,
    xlabel: str,
    ylabel: str,
    title: str,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    output_config: OutputConfig | None = None,
    annotate: bool = True,
    colorbar: bool = False,
    vmin=None,
    vmax=None,
    cmap: str = "rocket",
    rotate_ticks: bool = False,
):
    if axis_style is None:
        axis_style = AxisStyle()
    if figure_style is None:
        figure_style = FigureStyle()
    if output_config is None:
        output_config = OutputConfig()

    plt.ioff()
    if figure_style.use_seaborn:
        sns.set(style=figure_style.seaborn_style, font_scale=figure_style.seaborn_font_scale)

    fig, ax = plt.subplots(figsize=figure_style.figure_size)

    sns.heatmap(
        data,
        annot=annotate,
        fmt=".2f" if annotate else "",
        cbar=colorbar,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ax=ax,
        annot_kws={"fontsize": axis_style.xtick_size},
        cbar_kws={"pad": 0.01} if colorbar else None,
    )

    ax.set_xlabel(xlabel, fontsize=axis_style.xlabel_size, fontweight="bold", labelpad=20)
    ax.set_ylabel(ylabel, fontsize=axis_style.ylabel_size, fontweight="bold", labelpad=20)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(axis_style.xtick_size)

    if rotate_ticks:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
        for label in ax.get_yticklabels():
            label.set_rotation(0)

    return finalize_figure(fig, ax, title, figure_style, output_config)