"""Configuration dataclasses used by the plotting functions."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class AxisStyle:
    """Styling and scale options for matplotlib axes."""

    xlabel_size: int = 18
    ylabel_size: int = 18
    xtick_size: int = 14
    ytick_size: int = 14
    tick_width: float = 2.0
    tick_length: float = 6.0
    spine_width: float = 1.5
    pad_labels: float = 8.0
    pad_ticks: float = 6.0
    use_log_x: bool = False
    use_log_y: bool = False
    remove_first_xtick: bool = False
    remove_first_ytick: bool = False
    disable_xtick_marks: bool = False
    disable_ytick_marks: bool = False


@dataclass
class FigureStyle:
    """Figure-level sizing, title, layout, display, and seaborn options."""

    figure_size: tuple[float, float] = (10, 8)
    title_size: int = 18
    title_weight: str = "bold"
    tight_layout_pad: float = 0.5
    show_figure: bool = False
    use_seaborn: bool = True
    seaborn_style: str = "ticks"
    seaborn_font_scale: float = 1.5
    theme: Literal["default", "publication", "presentation", "minimal", "dark", "paper_bw"] = "default"


@dataclass
class OutputConfig:
    """Controls whether and where generated figures are saved or returned."""

    output_dir: Optional[str] = None
    filename: Optional[str] = None
    save_svg: bool = True
    save_png: bool = True
    save_pdf: bool = False
    dpi: int = 300
    return_fig: bool = False
    transparent: bool = False
    metadata: dict[str, Any] | None = None
    save_metadata: bool = False


@dataclass
class ShadedRegionSpec:
    """Vertical x-range shading used to highlight a plot region."""

    xmin: float
    xmax: float
    color: str = "gray"
    alpha: float = 0.2
    label: Optional[str] = None


@dataclass
class SignificanceBracketSpec:
    """Bracket annotation between two x positions."""

    x1: float
    x2: float
    y: float
    text: str
    height: float = 0.05
    color: str = "black"
    linewidth: float = 1.5
    fontsize: int = 12


@dataclass
class LegendStyle:
    """Legend visibility, placement, and typography options."""

    enabled: bool = True
    loc: str = "best"
    ncol: int = 1
    frameon: bool = False
    fontsize: int = 14
    handletextpad: float = 0.8
    handlelength: float = 1.5
    bbox_to_anchor: Optional[tuple[float, float]] = None
    labelcolor: Optional[str] = None


@dataclass
class SeriesStyle:
    """Visual style for one named data series."""

    color: str = "blue"
    label: Optional[str] = None
    linewidth: float = 2.0
    linestyle: str = "-"
    marker: Optional[str] = None
    markersize: float = 6.0
    alpha: float = 1.0
    edgecolor: Optional[str] = None
    align: str = "center"  # For bars: center or edge
    m_size_factor: float = 1.0  # For scatter: multiplicative factor for markersize


@dataclass
class TextStyle:
    """Text styling used by annotations."""

    fontsize: int = 14
    color: str = "black"
    bold: bool = False
    rotation: float = 0.0


@dataclass
class AnnotationSpec:
    """Text annotation positioned in a matplotlib coordinate system."""

    text: str
    xy: tuple[float, float]
    xycoords: str = "axes fraction"
    style: TextStyle = field(default_factory=TextStyle)


@dataclass
class ReferenceLineSpec:
    """Style and coordinate for one vertical or horizontal reference line."""

    value: float
    color: str = "black"
    linestyle: str = "--"
    linewidth: float = 1.5
    alpha: float = 1.0


@dataclass
class LineSpec:
    """Collections of reference lines to draw on an axis."""

    vertical: list[ReferenceLineSpec] = field(default_factory=list)
    horizontal: list[ReferenceLineSpec] = field(default_factory=list)


@dataclass
class ColorbarConfig:
    """Colorbar settings for scatter plots with mapped color values."""

    enabled: bool = False
    colormap: str = "viridis"
    label: Optional[str] = None
    location: str = "right"
    ticks: Optional[list[float]] = None
    tick_labels: Optional[list[str]] = None
    orientation: str = "vertical"
