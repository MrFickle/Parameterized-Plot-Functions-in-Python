from matplotlib.axes import Axes

from .configs import AnnotationSpec


def apply_annotations(ax: Axes, annotations: list[AnnotationSpec] | None) -> None:
    if not annotations:
        return

    for ann in annotations:
        ax.annotate(
            ann.text,
            xy=ann.xy,
            xycoords=ann.xycoords,
            fontsize=ann.style.fontsize,
            color=ann.style.color,
            fontweight="bold" if ann.style.bold else "normal",
            rotation=ann.style.rotation,
        )