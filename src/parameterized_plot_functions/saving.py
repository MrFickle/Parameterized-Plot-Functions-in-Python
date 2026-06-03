"""Figure saving and finalization helpers."""

import os

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .configs import FigureStyle, OutputConfig


def save_figure(fig: Figure, output_config: OutputConfig) -> None:
    """
    Function purpose:
        Save a figure as PNG and/or SVG according to the output configuration.

    Args:
        fig: Matplotlib figure to save.
        output_config: Output path, filename, format, and DPI configuration.

    Outputs:
        None. Files are written only when both output directory and filename are set.
    """
    if not output_config.output_dir or not output_config.filename:
        return

    # PNGs are saved directly in output_dir; SVGs are grouped in a subfolder.
    os.makedirs(output_config.output_dir, exist_ok=True)

    if output_config.save_png:
        png_path = os.path.join(output_config.output_dir, f"{output_config.filename}.png")
        fig.savefig(png_path, dpi=output_config.dpi, bbox_inches="tight")

    if output_config.save_svg:
        svg_dir = os.path.join(output_config.output_dir, "SVG")
        os.makedirs(svg_dir, exist_ok=True)
        svg_path = os.path.join(svg_dir, f"{output_config.filename}.svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")


def finalize_figure(
    fig: Figure,
    ax: Axes,
    title: str,
    figure_style: FigureStyle,
    output_config: OutputConfig,
) -> Figure | None:
    """
    Function purpose:
        Apply final figure formatting, handle save/show/close behavior, and
        optionally return the figure.

    Args:
        fig: Matplotlib figure to finalize.
        ax: Main matplotlib axis used for the figure title.
        title: Figure title text.
        figure_style: Figure-level styling and display configuration.
        output_config: Output saving and return behavior configuration.

    Outputs:
        The finalized figure when ``output_config.return_fig`` is true,
        otherwise ``None``.
    """
    # Finalization is centralized so all plot builders share save/show behavior.
    ax.set_title(
        title,
        fontsize=figure_style.title_size,
        fontweight=figure_style.title_weight,
    )
    fig.tight_layout(pad=figure_style.tight_layout_pad)

    if not output_config.return_fig:
        save_figure(fig, output_config)

    if figure_style.show_figure:        
        plt.show()
    else:
        plt.close(fig)

    return fig if output_config.return_fig else None
