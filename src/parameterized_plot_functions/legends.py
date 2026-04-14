import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.axes import Axes

from .configs import LegendStyle, SeriesStyle


def apply_line_legend(
    ax: Axes,
    series_styles: dict[str, SeriesStyle],
    legend_style: LegendStyle,
    linewidth_multiplier: float = 1.0,
) -> None:
    if not legend_style.enabled:
        return

    handles = []
    for key, style in series_styles.items():
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=style.color,
                marker=style.marker,
                markersize=style.markersize,
                linestyle=style.linestyle,
                linewidth=style.linewidth * linewidth_multiplier,
                label=style.label if style.label is not None else key,
                alpha=style.alpha,
            )
        )

    ax.legend(
        handles=handles,
        loc=legend_style.loc,
        ncol=legend_style.ncol,
        frameon=legend_style.frameon,
        prop={"size": legend_style.fontsize, "weight": "bold"},
        handletextpad=legend_style.handletextpad,
        handlelength=legend_style.handlelength,
        bbox_to_anchor=legend_style.bbox_to_anchor,
        labelcolor=legend_style.labelcolor,
    )


def apply_patch_legend(
    ax: Axes,
    patch_specs: list[tuple[str, str]],
    legend_style: LegendStyle,
) -> None:
    if not legend_style.enabled:
        return

    patches = [mpatches.Patch(color=color, label=label) for label, color in patch_specs]

    legend = ax.legend(
        handles=patches,
        loc=legend_style.loc,
        ncol=legend_style.ncol,
        frameon=legend_style.frameon,
        handletextpad=legend_style.handletextpad,
        bbox_to_anchor=legend_style.bbox_to_anchor,
    )

    for text in legend.get_texts():
        text.set_fontsize(legend_style.fontsize)
        text.set_fontweight("bold")