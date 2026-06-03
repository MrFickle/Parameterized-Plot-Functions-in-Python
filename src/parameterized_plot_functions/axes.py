"""Axis styling helpers shared by plot builders."""

from typing import Optional

from matplotlib.axes import Axes

from .configs import AxisStyle


def apply_axis_style(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    axis_style: AxisStyle,
    xlims: Optional[tuple[float, float]] = None,
    ylims: Optional[tuple[float, float]] = None,
    xticks: Optional[list[float]] = None,
    yticks: Optional[list[float]] = None,
    xtick_labels: Optional[list[str]] = None,
    ytick_labels: Optional[list[str]] = None,
) -> None:
    """
    Function purpose:
        Apply common axis labels, limits, ticks, scales, spine styling, and tick
        typography to a matplotlib axis.

    Args:
        ax: Matplotlib axis to style.
        xlabel: Text for the x-axis label.
        ylabel: Text for the y-axis label.
        axis_style: Axis styling configuration.
        xlims: Optional x-axis limits.
        ylims: Optional y-axis limits.
        xticks: Optional x-axis tick positions.
        yticks: Optional y-axis tick positions.
        xtick_labels: Optional x-axis tick labels.
        ytick_labels: Optional y-axis tick labels.

    Outputs:
        None. The axis is modified in place.
    """
    # Set axis labels before limits/ticks so downstream formatting only changes scale.
    ax.set_xlabel(
        xlabel,
        fontsize=axis_style.xlabel_size,
        labelpad=axis_style.pad_labels,
        fontweight="bold",
    )
    ax.set_ylabel(
        ylabel,
        fontsize=axis_style.ylabel_size,
        labelpad=axis_style.pad_labels,
        fontweight="bold",
    )

    # Optional explicit limits and tick positions override matplotlib defaults.
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Tick labels are optional because callers often only need custom tick locations.
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels, fontsize=axis_style.xtick_size)
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels, fontsize=axis_style.ytick_size)

    # Log scaling happens after ticks are assigned so matplotlib can validate them.
    if axis_style.use_log_x:
        ax.set_xscale("log")
    if axis_style.use_log_y:
        ax.set_yscale("log")

    ax.tick_params(
        axis="x",
        which="major",
        width=0 if axis_style.disable_xtick_marks else axis_style.tick_width,
        length=0 if axis_style.disable_xtick_marks else axis_style.tick_length,
        labelsize=axis_style.xtick_size,
        pad=axis_style.pad_ticks,
    )
    ax.tick_params(
        axis="y",
        which="major",
        width=0 if axis_style.disable_ytick_marks else axis_style.tick_width,
        length=0 if axis_style.disable_ytick_marks else axis_style.tick_length,
        labelsize=axis_style.ytick_size,
        pad=axis_style.pad_ticks,
    )

    # Hide only the first visible label while preserving the tick location itself.
    if axis_style.remove_first_xtick:
        ticks = ax.xaxis.get_major_ticks()
        if len(ticks) > 0:
            ticks[0].label1.set_visible(False)

    if axis_style.remove_first_ytick:
        ticks = ax.yaxis.get_major_ticks()
        if len(ticks) > 0:
            ticks[0].label1.set_visible(False)

    # Keep the left/bottom frame prominent while using a cleaner top/right edge.
    for spine in ax.spines.values():
        spine.set_linewidth(axis_style.spine_width)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def extend_y_axis_one_tick(ax: Axes) -> None:
    """
    Function purpose:
        Extend the y-axis upper limit by one major tick interval.

    Args:
        ax: Matplotlib axis whose y-axis range should be extended.

    Outputs:
        None. The axis limit is modified in place when enough ticks exist.
    """
    yticks = ax.get_yticks()
    if len(yticks) < 2:
        return
    # Use the current major tick spacing to preserve the existing tick cadence.
    step = yticks[1] - yticks[0]
    ax.set_ylim(yticks[0], yticks[-1] + step)
