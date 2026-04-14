import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import seaborn as sns

from ..annotations import apply_annotations
from ..axes import apply_axis_style, extend_y_axis_one_tick
from ..configs import (
    AnnotationSpec,
    AxisStyle,
    FigureStyle,
    LegendStyle,
    LineSpec,
    OutputConfig,
    SeriesStyle,
)
from ..legends import apply_line_legend
from ..saving import finalize_figure


def plot_line(
    x_series: dict[str, np.ndarray],
    y_series: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    series_styles: dict[str, SeriesStyle] | None = None,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    yerr_series: dict[str, np.ndarray] | None = None,
    use_fill_between: bool = True,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    annotations: list[AnnotationSpec] | None = None,
    line_spec: LineSpec | None = None,
    use_mask: bool = True,
    errorbar_capsize: float = 4,
    errorbar_elinewidth: float = 2,
    errorbar_capthick: float = 2,
    use_line_color_for_error: bool = False,
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
        series_styles = {key: SeriesStyle(label=key) for key in x_series}

    plt.ioff()
    if figure_style.use_seaborn:
        sns.set(style=figure_style.seaborn_style, font_scale=figure_style.seaborn_font_scale)

    fig, ax = plt.subplots(figsize=figure_style.figure_size)

    for key in x_series:
        x = np.asarray(x_series[key])
        y = np.asarray(y_series[key])
        style = series_styles.get(key, SeriesStyle(label=key))

        if use_mask:
            mask = np.isfinite(y)
            x_plot = x[mask]
            y_plot = y[mask]
        else:
            x_plot = x
            y_plot = y

        if yerr_series is not None and key in yerr_series:
            yerr = np.asarray(yerr_series[key])
            if use_mask:
                yerr = yerr[mask]

            if use_fill_between:
                ax.plot(
                    x_plot,
                    y_plot,
                    color=style.color,
                    linewidth=style.linewidth,
                    linestyle=style.linestyle,
                    marker=style.marker,
                    markersize=style.markersize,
                    alpha=style.alpha,
                )
                ax.fill_between(
                    x_plot,
                    y_plot - yerr,
                    y_plot + yerr,
                    color=style.color if use_line_color_for_error else "black",
                    alpha=min(style.alpha, 0.20),
                )
            else:
                ax.errorbar(
                    x_plot,
                    y_plot,
                    yerr=yerr,
                    fmt=style.marker if style.marker is not None else "none",
                    color=style.color,
                    ecolor=style.color if use_line_color_for_error else "black",
                    linewidth=style.linewidth,
                    linestyle=style.linestyle,
                    markersize=style.markersize,
                    capsize=errorbar_capsize,
                    elinewidth=errorbar_elinewidth,
                    capthick=errorbar_capthick,
                    alpha=style.alpha,
                )
                if style.marker is None:
                    ax.plot(
                        x_plot,
                        y_plot,
                        color=style.color,
                        linewidth=style.linewidth,
                        linestyle=style.linestyle,
                        alpha=style.alpha,
                    )
        else:
            ax.plot(
                x_plot,
                y_plot,
                color=style.color,
                linewidth=style.linewidth,
                linestyle=style.linestyle,
                marker=style.marker,
                markersize=style.markersize,
                alpha=style.alpha,
            )

    if line_spec is not None:
        for spec in line_spec.vertical:
            ax.axvline(
                x=spec.value,
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                alpha=spec.alpha,
            )
        for spec in line_spec.horizontal:
            ax.axhline(
                y=spec.value,
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=spec.linewidth,
                alpha=spec.alpha,
            )

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
        ax.xaxis.set_tick_params(which="minor", width=axis_style.tick_width / 2, length=axis_style.tick_length / 2)
        ax.yaxis.set_tick_params(which="minor", width=axis_style.tick_width / 2, length=axis_style.tick_length / 2)

    if extend_y_one_tick:
        extend_y_axis_one_tick(ax)

    apply_line_legend(ax, series_styles, legend_style, linewidth_multiplier=1.0)
    apply_annotations(ax, annotations)

    return finalize_figure(fig, ax, title, figure_style, output_config)