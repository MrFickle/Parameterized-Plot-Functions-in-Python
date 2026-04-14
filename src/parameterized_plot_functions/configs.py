from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AxisStyle:
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
    figure_size: tuple[float, float] = (10, 8)
    title_size: int = 18
    title_weight: str = "bold"
    tight_layout_pad: float = 0.5
    show_figure: bool = False
    use_seaborn: bool = True
    seaborn_style: str = "ticks"
    seaborn_font_scale: float = 1.5


@dataclass
class OutputConfig:
    output_dir: Optional[str] = None
    filename: Optional[str] = None
    save_svg: bool = True
    save_png: bool = True
    dpi: int = 300
    return_fig: bool = False


@dataclass
class LegendStyle:
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
    fontsize: int = 14
    color: str = "black"
    bold: bool = False
    rotation: float = 0.0


@dataclass
class AnnotationSpec:
    text: str
    xy: tuple[float, float]
    xycoords: str = "axes fraction"
    style: TextStyle = field(default_factory=TextStyle)


@dataclass
class ReferenceLineSpec:
    value: float
    color: str = "black"
    linestyle: str = "--"
    linewidth: float = 1.5
    alpha: float = 1.0


@dataclass
class LineSpec:
    vertical: list[ReferenceLineSpec] = field(default_factory=list)
    horizontal: list[ReferenceLineSpec] = field(default_factory=list)


@dataclass
class ColorbarConfig:
    enabled: bool = False
    colormap: str = "viridis"
    label: Optional[str] = None
    location: str = "right"
    ticks: Optional[list[float]] = None
    tick_labels: Optional[list[str]] = None
    orientation: str = "vertical"