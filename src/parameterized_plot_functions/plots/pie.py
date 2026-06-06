"""Pie chart builder."""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..annotations import apply_annotations
from ..configs import AnnotationSpec, FigureStyle, LegendStyle, OutputConfig, SeriesStyle
from ..saving import finalize_figure
from ..style.themes import apply_theme


def plot_pie(
    values: dict[str, float],
    title: str,
    figure_style: FigureStyle | None = None,
    legend_style: LegendStyle | None = None,
    output_config: OutputConfig | None = None,
    series_styles: dict[str, SeriesStyle] | None = None,
    annotations: list[AnnotationSpec] | None = None,
    autopct: str | None = "%1.1f%%",
    startangle: float = 90,
    donut_width: float | None = None,
    explode: list[float] | None = None,
    shadow: bool = False,
    labeldistance: float = 1.1,
    pctdistance: float = 0.6,
    counterclock: bool = True,
    normalize: bool = True,
    textprops: dict[str, object] | None = None,
    wedgeprops: dict[str, object] | None = None,
    show_legend: bool = False,
    legend_loc: str = "best",
) -> Figure | None:
    """
    Function purpose:
        Draw a parameterized pie or donut chart from named scalar values.

    Args:
        values: Mapping from slice label to numeric size.
        title: Figure title text.
        figure_style: Optional figure-level styling configuration.
        legend_style: Optional legend styling configuration.
        output_config: Optional output saving and return behavior.
        series_styles: Optional per-slice color and label styles.
        annotations: Optional annotations to draw on the axis.
        autopct: Percent label format passed to Matplotlib.
        startangle: Starting angle in degrees.
        donut_width: Optional wedge width for donut charts.
        explode: Optional radial offset for each slice.
        shadow: Whether to draw a pie shadow.
        labeldistance: Distance of labels from the center.
        pctdistance: Distance of percent labels from the center.
        counterclock: Whether slices are drawn counterclockwise.
        normalize: Whether Matplotlib normalizes values to one full pie.
        textprops: Optional text properties for labels.
        wedgeprops: Optional wedge properties.
        show_legend: Whether to show a legend instead of relying only on labels.
        legend_loc: Legend location when enabled.

    Outputs:
        The figure when ``output_config.return_fig`` is true, otherwise ``None``.
    """
    # Instantiate configs at call time so callers can omit boilerplate safely.
    figure_style = figure_style or FigureStyle()
    legend_style = legend_style or LegendStyle()
    output_config = output_config or OutputConfig()
    series_styles = series_styles or {key: SeriesStyle(label=key) for key in values}

    # Disable interactive rendering and apply the requested style preset.
    plt.ioff()
    apply_theme(figure_style)

    # Convert the value mapping into ordered lists for Matplotlib.
    fig, ax = plt.subplots(figsize=figure_style.figure_size)
    keys = list(values.keys())
    colors = [series_styles[key].color for key in keys]
    labels = [series_styles[key].label if series_styles[key].label is not None else key for key in keys]

    # Merge donut configuration with caller-supplied wedge properties.
    pie_wedgeprops = dict(wedgeprops or {})
    if donut_width is not None:
        pie_wedgeprops["width"] = donut_width

    # Draw the pie with Matplotlib's native pie controls exposed.
    wedges, _, _ = ax.pie(
        [values[key] for key in keys],
        labels=labels,
        colors=colors,
        autopct=autopct,
        startangle=startangle,
        explode=explode,
        shadow=shadow,
        labeldistance=labeldistance,
        pctdistance=pctdistance,
        counterclock=counterclock,
        normalize=normalize,
        textprops=textprops,
        wedgeprops=pie_wedgeprops if pie_wedgeprops else None,
    )

    # Force a circular aspect ratio so the pie is not distorted by figure size.
    ax.axis("equal")

    # Optional legend is useful when labels are dense or donut labels are hidden.
    if show_legend and legend_style.enabled:
        ax.legend(
            wedges,
            labels,
            loc=legend_style.loc if legend_style.loc != "best" else legend_loc,
            ncol=legend_style.ncol,
            frameon=legend_style.frameon,
            fontsize=legend_style.fontsize,
            handletextpad=legend_style.handletextpad,
            handlelength=legend_style.handlelength,
            bbox_to_anchor=legend_style.bbox_to_anchor,
            labelcolor=legend_style.labelcolor,
        )

    # Draw optional annotations after the pie so labels can sit above wedges.
    apply_annotations(ax, annotations)

    return finalize_figure(fig, ax, title, figure_style, output_config)
