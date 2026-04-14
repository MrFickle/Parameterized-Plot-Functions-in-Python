import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

from ..annotations import apply_annotations
from ..axes import apply_axis_style, extend_y_axis_one_tick
from ..configs import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, OutputConfig, SeriesStyle
from ..legends import apply_patch_legend
from ..saving import finalize_figure


def plot_bar(
    values: dict[str, float],
    xlabel: str,
    ylabel: str,
    title: str,
    x_positions: dict[str, float],
    bar_widths: dict[str, float],
    series_styles: dict[str, SeriesStyle] | None = None,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    sem_values: dict[str, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    annotations: list[AnnotationSpec] | None = None,
    edgecolor: dict[str, str] | str | None = None,
    rotate_xticks: bool = False,
    plot_minor_ticks: bool = False,
    extend_y_one_tick: bool = False,
):
    if axis_style is None:
        axis_style = AxisStyle()
    if figure_style is None:
        figure_style = FigureStyle()
    if legend_style is None:
        legend_style = LegendStyle()
    if output_config is None:
        output_config = OutputConfig()
    if series_styles is None:
        series_styles = {key: SeriesStyle(label=key) for key in values}

    plt.ioff()
    if figure_style.use_seaborn:
        sns.set(style=figure_style.seaborn_style, font_scale=figure_style.seaborn_font_scale)

    fig, ax = plt.subplots(figsize=figure_style.figure_size)

    patch_specs = []
    for key, value in values.items():
        style = series_styles.get(key, SeriesStyle(label=key))
        yerr = None if sem_values is None else sem_values.get(key, None)

        # Handle edgecolor logic: can be single color or dict
        current_edgecolor = style.edgecolor
        if isinstance(edgecolor, dict):
            current_edgecolor = edgecolor.get(key, style.edgecolor)
        elif edgecolor is not None:
            current_edgecolor = edgecolor

        ax.bar(
            x_positions[key],
            value,
            width=bar_widths[key],
            color=style.color,
            alpha=style.alpha,
            edgecolor=current_edgecolor,
            yerr=yerr,
            align=style.align,
            capsize=4,
            error_kw={'elinewidth': 2, 'capthick': 2}
        )

        patch_specs.append((style.label if style.label is not None else key, style.color))

    apply_axis_style(
        ax=ax,
        xlabel=xlabel,
        ylabel=ylabel,
        axis_style=axis_style,
        xlims=xlims,
        ylims=ylims,
        xticks=xticks,
        yticks=yticks,
        xtick_labels=xtick_labels,
        ytick_labels=ytick_labels,
    )

    if rotate_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    if plot_minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        # Sync minor tick styling with major
        ax.xaxis.set_tick_params(which='minor', width=axis_style.tick_width/2, length=axis_style.tick_length/2)
        ax.yaxis.set_tick_params(which='minor', width=axis_style.tick_width/2, length=axis_style.tick_length/2)

    if extend_y_one_tick:
        extend_y_axis_one_tick(ax)

    apply_patch_legend(ax, patch_specs, legend_style)
    apply_annotations(ax, annotations)

    return finalize_figure(fig, ax, title, figure_style, output_config)


def plot_dual_axis_bar(
    values: dict[str, float],
    xlabel: str,
    ylabel_left: str,
    ylabel_right: str,
    title: str,
    x_positions: dict[str, float],
    bar_widths: dict[str, float],
    axis_assignment: dict[str, str],
    series_styles: dict[str, SeriesStyle] | None = None,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    xticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    yticks_left: list[float] | None = None,
    yticks_right: list[float] | None = None,
    ylims_left: tuple[float, float] | None = None,
    ylims_right: tuple[float, float] | None = None,
    sem_values: dict[str, float] | None = None,
    edgecolor: dict[str, str] | str | None = None,
    rotate_xticks: bool = False,
    plot_minor_ticks: bool = False,
    extend_y_one_tick: bool = False,
    annotations: list[AnnotationSpec] | None = None,
):
    if axis_style is None:
        axis_style = AxisStyle()
    if figure_style is None:
        figure_style = FigureStyle()
    if legend_style is None:
        legend_style = LegendStyle()
    if output_config is None:
        output_config = OutputConfig()
    if series_styles is None:
        series_styles = {key: SeriesStyle(label=key) for key in values}

    plt.ioff()
    if figure_style.use_seaborn:
        sns.set(style=figure_style.seaborn_style, font_scale=figure_style.seaborn_font_scale)

    fig, ax1 = plt.subplots(figsize=figure_style.figure_size)
    ax2 = ax1.twinx()

    patch_specs = []
    for key, value in values.items():
        style = series_styles.get(key, SeriesStyle(label=key))
        target_ax = ax1 if axis_assignment[key] == "left" else ax2
        yerr = None if sem_values is None else sem_values.get(key, None)

        current_edgecolor = style.edgecolor
        if isinstance(edgecolor, dict):
            current_edgecolor = edgecolor.get(key, style.edgecolor)
        elif edgecolor is not None:
            current_edgecolor = edgecolor

        target_ax.bar(
            x_positions[key],
            value,
            width=bar_widths[key],
            color=style.color,
            alpha=style.alpha,
            edgecolor=current_edgecolor,
            yerr=yerr,
            align=style.align,
            capsize=4,
            error_kw={'elinewidth': 2, 'capthick': 2}
        )
        patch_specs.append((style.label if style.label is not None else key, style.color))

    # Apply style to ax1
    apply_axis_style(
        ax=ax1,
        xlabel=xlabel,
        ylabel=ylabel_left,
        axis_style=axis_style,
        xlims=None,
        ylims=ylims_left,
        xticks=xticks,
        yticks=yticks_left,
        xtick_labels=xtick_labels,
        ytick_labels=None,
    )
    
    # Custom apply to ax2
    ax2.set_ylabel(ylabel_right, fontsize=axis_style.ylabel_size, labelpad=axis_style.pad_labels, fontweight="bold")
    if yticks_right is not None:
        ax2.set_yticks(yticks_right)
    if ylims_right is not None:
        ax2.set_ylim(ylims_right)
    
    ax2.tick_params(
        axis="y",
        which="major",
        width=0 if axis_style.disable_ytick_marks else axis_style.tick_width,
        length=0 if axis_style.disable_ytick_marks else axis_style.tick_length,
        labelsize=axis_style.ytick_size,
        pad=axis_style.pad_ticks,
    )

    for spine in ax2.spines.values():
        spine.set_linewidth(axis_style.spine_width)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    if rotate_xticks:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    if plot_minor_ticks:
        for ax in [ax1, ax2]:
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_tick_params(which='minor', width=axis_style.tick_width/2, length=axis_style.tick_length/2)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.xaxis.set_tick_params(which='minor', width=axis_style.tick_width/2, length=axis_style.tick_length/2)

    if extend_y_one_tick:
        extend_y_axis_one_tick(ax1)
        extend_y_axis_one_tick(ax2)

    apply_patch_legend(ax1, patch_specs, legend_style)
    apply_annotations(ax1, annotations)

    ax1.set_title(title, fontsize=figure_style.title_size, fontweight=figure_style.title_weight)
    fig.tight_layout(pad=figure_style.tight_layout_pad)

    if not output_config.return_fig:
        from ..saving import save_figure
        save_figure(fig, output_config)

    if figure_style.show_figure:
        plt.show()
    else:
        plt.close(fig)

    return fig if output_config.return_fig else None