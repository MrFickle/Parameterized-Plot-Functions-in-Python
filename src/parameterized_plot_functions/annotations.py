"""Annotation helpers for plot and figure overlays."""

from matplotlib.axes import Axes

from .configs import AnnotationSpec


def apply_annotations(ax: Axes, annotations: list[AnnotationSpec] | None) -> None:
    """
    Function purpose:
        Draw text annotations on a matplotlib axis.

    Args:
        ax: Matplotlib axis that receives the annotations.
        annotations: Optional list of annotation specifications to draw.

    Outputs:
        None. The axis is modified in place.
    """
    if not annotations:
        return

    # AnnotationSpec carries both position and text styling for each label.
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
