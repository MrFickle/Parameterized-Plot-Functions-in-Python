"""Public API for parameterized matplotlib plotting helpers."""

import matplotlib

matplotlib.use("Agg", force=True)

from .configs import (
    AnnotationSpec,
    AxisStyle,
    ColorbarConfig,
    FigureStyle,
    LegendStyle,
    LineSpec,
    OutputConfig,
    ReferenceLineSpec,
    SeriesStyle,
    ShadedRegionSpec,
    SignificanceBracketSpec,
    TextStyle,
)

from .layouts import create_empty_figure, create_subplots_figure, draw_figures_grid

from .plots.line import plot_line
from .plots.hist import plot_histogram
from .plots.scatter import plot_scatter
from .plots.bar import plot_bar, plot_dual_axis_bar, plot_grouped_bar, plot_stacked_bar
from .plots.heatmap import plot_heatmap, plot_correlation_heatmap
from .plots.distribution import plot_box, plot_violin
from .plots.pie import plot_pie
from .plots.area import plot_area
from .plots.density import plot_contour, plot_hexbin
from .plots.timeline import plot_timeline

__all__ = [
    "AnnotationSpec",
    "AxisStyle",
    "ColorbarConfig",
    "FigureStyle",
    "LegendStyle",
    "LineSpec",
    "OutputConfig",
    "ReferenceLineSpec",
    "SeriesStyle",
    "ShadedRegionSpec",
    "SignificanceBracketSpec",
    "TextStyle",
    "create_empty_figure",
    "create_subplots_figure",
    "draw_figures_grid",
    "plot_line",
    "plot_histogram",
    "plot_scatter",
    "plot_bar",
    "plot_dual_axis_bar",
    "plot_grouped_bar",
    "plot_stacked_bar",
    "plot_heatmap",
    "plot_correlation_heatmap",
    "plot_box",
    "plot_violin",
    "plot_pie",
    "plot_area",
    "plot_contour",
    "plot_hexbin",
    "plot_timeline",
]
