import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# import diptest

from ..annotations import apply_annotations
from ..axes import apply_axis_style, extend_y_axis_one_tick
from ..configs import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, LineSpec, OutputConfig, SeriesStyle
from ..legends import apply_patch_legend
from ..saving import finalize_figure


def plot_histogram(
    data_series: dict[str, np.ndarray],
    bins,
    xlabel: str,
    ylabel: str,
    title: str,
    series_styles: dict[str, SeriesStyle] | None = None,
    axis_style: AxisStyle | None = None,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    hist_stat: str = "probability",
    plot_kde: bool = False,
    plot_mean: bool = False,
    plot_std: bool = False,
    perform_dip_test: bool = False,
    vertical_lines: list[float] | None = None,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    xticks: list[float] | None = None,
    yticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    ytick_labels: list[str] | None = None,
    annotations: list[AnnotationSpec] | None = None,
    line_spec: LineSpec | None = None,
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
        series_styles = {key: SeriesStyle(label=key) for key in data_series}

    plt.ioff()
    if figure_style.use_seaborn:
        sns.set(style=figure_style.seaborn_style, font_scale=figure_style.seaborn_font_scale)

    fig, ax = plt.subplots(figsize=figure_style.figure_size)
    patch_specs = []

    for key, values in data_series.items():
        style = series_styles.get(key, SeriesStyle(label=key))
        values = np.asarray(values)

        sns.histplot(
            data=values,
            bins=bins,
            stat=hist_stat,
            kde=plot_kde,
            color=style.color,
            alpha=style.alpha,
            edgecolor="none",
            ax=ax,
            line_kws={"linewidth": style.linewidth},
        )

        label = style.label if style.label is not None else key
        suffix_parts = []

        # if perform_dip_test:
        #     _, pval = diptest.diptest(values)
        #     suffix_parts.append(f"Dip p={pval:.3f}")
        if plot_mean:
            suffix_parts.append(f"μ={np.nanmean(values):.2f}")
        if plot_std:
            suffix_parts.append(f"σ={np.nanstd(values, ddof=1):.2f}")

        if suffix_parts:
            label = f"{label}, " + ", ".join(suffix_parts)

        patch_specs.append((label, style.color))

    if line_spec is not None:
        for spec in line_spec.vertical:
            ax.axvline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)
        for spec in line_spec.horizontal:
            ax.axhline(spec.value, color=spec.color, linestyle=spec.linestyle, linewidth=spec.linewidth, alpha=spec.alpha)

    if vertical_lines is not None:
        for v in vertical_lines:
            ax.axvline(x=v, color="black", linestyle="--", linewidth=1.5)

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

    if extend_y_one_tick:
        extend_y_axis_one_tick(ax)

    apply_patch_legend(ax, patch_specs, legend_style)
    apply_annotations(ax, annotations)

    return finalize_figure(fig, ax, title, figure_style, output_config)