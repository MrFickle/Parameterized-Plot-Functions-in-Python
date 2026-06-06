"""Named style presets shared by plotting functions."""

import matplotlib.pyplot as plt
import seaborn as sns

from ..configs import AxisStyle, FigureStyle


def apply_theme(figure_style: FigureStyle, axis_style: AxisStyle | None = None) -> None:
    """Apply a named plotting theme before a figure is created."""
    if figure_style.theme == "default":
        if figure_style.use_seaborn:
            sns.set(style=figure_style.seaborn_style, font_scale=figure_style.seaborn_font_scale)
        return

    if figure_style.theme == "publication":
        sns.set(style="ticks", font_scale=1.2)
        plt.rcParams.update({"axes.linewidth": 1.5, "font.size": 12})
    elif figure_style.theme == "presentation":
        sns.set(style="whitegrid", font_scale=1.6)
        plt.rcParams.update({"lines.linewidth": 3.0, "font.size": 16})
    elif figure_style.theme == "minimal":
        sns.set(style="white", font_scale=1.1)
        plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    elif figure_style.theme == "dark":
        sns.set_theme(
            style="darkgrid",
            palette="bright",
            rc={
                "figure.facecolor": "#1f1f1f",
                "axes.facecolor": "#1f1f1f",
                "axes.edgecolor": "white",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "text.color": "white",
            },
        )
    elif figure_style.theme == "paper_bw":
        sns.set(style="ticks", palette="gray", font_scale=1.1)
        plt.rcParams.update({"image.cmap": "gray", "axes.prop_cycle": plt.cycler(color=["black", "0.35", "0.6"])})

    if axis_style is not None and figure_style.theme == "presentation":
        axis_style.xlabel_size = max(axis_style.xlabel_size, 22)
        axis_style.ylabel_size = max(axis_style.ylabel_size, 22)
        axis_style.xtick_size = max(axis_style.xtick_size, 16)
        axis_style.ytick_size = max(axis_style.ytick_size, 16)
