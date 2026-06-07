"""Pydantic PlotSpec models, validation, and schema helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator, model_validator


SUPPORTED_PLOT_TYPES = [
    "line",
    "scatter",
    "histogram",
    "bar",
    "dual_axis_bar",
    "grouped_bar",
    "stacked_bar",
    "heatmap",
    "correlation_heatmap",
    "box",
    "violin",
    "pie",
    "area",
    "hexbin",
    "contour",
    "timeline",
]

PlotType = Literal[
    "line",
    "scatter",
    "histogram",
    "bar",
    "dual_axis_bar",
    "grouped_bar",
    "stacked_bar",
    "heatmap",
    "correlation_heatmap",
    "box",
    "violin",
    "pie",
    "area",
    "hexbin",
    "contour",
    "timeline",
]


class StrictModel(BaseModel):
    """Base model that rejects misspelled fields in PlotSpec dictionaries."""

    model_config = ConfigDict(extra="forbid")


class TextStyleSpec(StrictModel):
    """Pydantic equivalent of `TextStyle`."""

    fontsize: int = 14
    color: str = "black"
    bold: bool = False
    rotation: float = 0.0


class AnnotationSpecModel(StrictModel):
    """Pydantic equivalent of `AnnotationSpec`."""

    text: str
    xy: tuple[float, float] = (0.05, 0.95)
    xycoords: str = "axes fraction"
    style: TextStyleSpec = Field(default_factory=TextStyleSpec)


class ReferenceLineSpecModel(StrictModel):
    """Pydantic equivalent of `ReferenceLineSpec`."""

    value: float
    color: str = "black"
    linestyle: str = "--"
    linewidth: float = 1.5
    alpha: float = 1.0


class LineSpecModel(StrictModel):
    """Pydantic equivalent of `LineSpec`."""

    vertical: list[ReferenceLineSpecModel] = Field(default_factory=list)
    horizontal: list[ReferenceLineSpecModel] = Field(default_factory=list)


class ShadedRegionSpecModel(StrictModel):
    """Pydantic equivalent of `ShadedRegionSpec`."""

    xmin: float
    xmax: float
    color: str = "gray"
    alpha: float = 0.2
    label: str | None = None


class SignificanceBracketSpecModel(StrictModel):
    """Pydantic equivalent of `SignificanceBracketSpec`."""

    x1: float
    x2: float
    y: float
    text: str
    height: float = 0.05
    color: str = "black"
    linewidth: float = 1.5
    fontsize: int = 12


class AxisStyleSpec(StrictModel):
    """Pydantic equivalent of `AxisStyle`."""

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


class FigureStyleSpec(StrictModel):
    """Pydantic equivalent of `FigureStyle`."""

    figure_size: tuple[float, float] = (10, 8)
    title_size: int = 18
    title_weight: str = "bold"
    tight_layout_pad: float = 0.5
    show_figure: bool = False
    use_seaborn: bool = True
    seaborn_style: str = "ticks"
    seaborn_font_scale: float = 1.5
    theme: Literal["default", "publication", "presentation", "minimal", "dark", "paper_bw"] = "default"


class LegendStyleSpec(StrictModel):
    """Pydantic equivalent of `LegendStyle`."""

    enabled: bool = True
    loc: str = "best"
    ncol: int = 1
    frameon: bool = False
    fontsize: int = 14
    handletextpad: float = 0.8
    handlelength: float = 1.5
    bbox_to_anchor: tuple[float, float] | None = None
    labelcolor: str | None = None


class SeriesStyleSpec(StrictModel):
    """Pydantic equivalent of `SeriesStyle`."""

    color: str = "blue"
    label: str | None = None
    linewidth: float = 2.0
    linestyle: str = "-"
    marker: str | None = None
    markersize: float = 6.0
    alpha: float = 1.0
    edgecolor: str | None = None
    align: str = "center"
    m_size_factor: float = 1.0


class ColorbarConfigSpec(StrictModel):
    """Pydantic equivalent of `ColorbarConfig`."""

    enabled: bool = False
    colormap: str = "viridis"
    label: str | None = None
    location: str = "right"
    ticks: list[float] | None = None
    tick_labels: list[str] | None = None
    orientation: str = "vertical"


class OutputSpec(StrictModel):
    """Output configuration used by PlotSpec rendering."""

    output_dir: str = "plot_outputs"
    filename: str | None = None
    formats: list[Literal["png", "svg", "pdf"]] = Field(default_factory=lambda: ["png", "svg"])
    dpi: int = 300
    transparent: bool = False
    save_metadata: bool = True


class InlineDataSpec(RootModel[dict[str, Any]]):
    """Inline plot data payload."""


class CsvDataSpec(StrictModel):
    """CSV-backed data payload."""

    path: str
    mappings: dict[str, Any]

    @field_validator("path")
    @classmethod
    def validate_existing_path(cls, value: str) -> str:
        """
        Function purpose:
            Validate that a referenced CSV path exists.

        Args:
            value: Candidate CSV path.

        Outputs:
            Validated CSV path.
        """
        if not Path(value).exists():
            raise ValueError(f"CSV file does not exist: {value}")
        return value


class DataFrameDataSpec(StrictModel):
    """DataFrame-backed data payload for Python callers."""

    name: str
    mappings: dict[str, Any]


class PlotDataSpec(StrictModel):
    """Data payload container supporting inline, CSV, and DataFrame modes."""

    inline: dict[str, Any] | None = None
    csv: CsvDataSpec | None = None
    dataframe: DataFrameDataSpec | None = None

    @model_validator(mode="after")
    def validate_single_data_mode(self) -> PlotDataSpec:
        """
        Function purpose:
            Validate that exactly one data source mode is provided.

        Args:
            None.

        Outputs:
            The validated data payload.
        """
        provided = [self.inline is not None, self.csv is not None, self.dataframe is not None]
        if sum(provided) != 1:
            raise ValueError("PlotSpec data must contain exactly one of: inline, csv, dataframe.")
        return self


class CommonPlotSpec(StrictModel):
    """Common fields available to every plot-specific PlotSpec."""

    plot_type: PlotType
    title: str
    data: PlotDataSpec
    xlabel: str = ""
    ylabel: str = ""
    axis: AxisStyleSpec = Field(default_factory=AxisStyleSpec)
    style: FigureStyleSpec = Field(default_factory=FigureStyleSpec)
    legend: LegendStyleSpec = Field(default_factory=LegendStyleSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    annotations: list[AnnotationSpecModel] = Field(default_factory=list)
    series_styles: dict[str, SeriesStyleSpec] | None = None
    line_spec: LineSpecModel | None = None
    options: dict[str, Any] = Field(default_factory=dict, description="Deprecated compatibility field. Prefer plot-specific top-level fields.")


class LinePlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_line`."""

    plot_type: Literal["line"]
    yerr_series: dict[str, list[float]] | None = None
    ci_series: dict[str, tuple[list[float], list[float]]] | None = None
    use_fill_between: bool = True
    step_where: str | None = None
    rolling_window: int | None = None
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    xtick_labels: list[str] | None = None
    ytick_labels: list[str] | None = None
    shaded_regions: list[ShadedRegionSpecModel] | None = None
    endpoint_labels: bool = False
    use_mask: bool = True
    errorbar_capsize: float = 4
    errorbar_elinewidth: float = 2
    errorbar_capthick: float = 2
    use_line_color_for_error: bool = False
    rotate_xticks: bool = False
    plot_minor_ticks: bool = False
    extend_y_one_tick: bool = False


class ScatterPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_scatter`."""

    plot_type: Literal["scatter"]
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    xtick_labels: list[str] | None = None
    ytick_labels: list[str] | None = None
    do_linear_reg_fit: bool = False
    plot_r2_score: bool = False
    polynomial_degree: int | None = None
    regression_confidence_band: bool = False
    point_labels: dict[str, list[str]] | None = None
    jitter: float = 0.0
    size_values: dict[str, list[float]] | None = None
    colorbar_config: ColorbarConfigSpec | None = None
    color_values: dict[str, list[float]] | None = None


class HistogramPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_histogram`."""

    plot_type: Literal["histogram"]
    bins: Any = 20
    hist_stat: str = "probability"
    plot_kde: bool = False
    plot_mean: bool = False
    plot_std: bool = False
    cumulative: bool = False
    fitted_distribution: str | None = None
    percentile_markers: list[float] | None = None
    perform_dip_test: bool = False
    vertical_lines: list[float] | None = None
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    xtick_labels: list[str] | None = None
    ytick_labels: list[str] | None = None
    extend_y_one_tick: bool = False


class BarPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_bar`."""

    plot_type: Literal["bar"]
    x_positions: dict[str, float] | None = None
    bar_widths: dict[str, float] | None = None
    sem_values: dict[str, float] | None = None
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    xtick_labels: list[str] | None = None
    ytick_labels: list[str] | None = None
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None
    edgecolor: dict[str, str] | str | None = None
    horizontal: bool = False
    sort_values: bool = False
    value_labels: bool = False
    significance_brackets: list[SignificanceBracketSpecModel] | None = None
    rotate_xticks: bool = False
    plot_minor_ticks: bool = False
    extend_y_one_tick: bool = False


class DualAxisBarPlotSpec(BarPlotSpec):
    """PlotSpec for `plot_dual_axis_bar`."""

    plot_type: Literal["dual_axis_bar"]
    ylabel_left: str = ""
    ylabel_right: str = ""
    axis_assignment: dict[str, Literal["left", "right"]] | None = None
    yticks_left: list[float] | None = None
    yticks_right: list[float] | None = None
    ylims_left: tuple[float, float] | None = None
    ylims_right: tuple[float, float] | None = None


class GroupedBarPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_grouped_bar`."""

    plot_type: Literal["grouped_bar"]
    value_labels: bool = False
    group_gap: float = 1.0
    bar_width: float = 0.8


class StackedBarPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_stacked_bar`."""

    plot_type: Literal["stacked_bar"]
    normalize: bool = False
    value_labels: bool = False


class HeatmapPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_heatmap`."""

    plot_type: Literal["heatmap"]
    annotate: bool = True
    colorbar: bool = False
    vmin: float | None = None
    vmax: float | None = None
    cmap: str = "rocket"
    rotate_ticks: bool = False
    xtick_labels: list[str] | None = None
    ytick_labels: list[str] | None = None
    triangular_mask: Literal["upper", "lower"] | None = None
    center: float | None = None
    normalize: Literal["row", "column", "global"] | None = None
    auto_text_contrast: bool = False


class CorrelationHeatmapPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_correlation_heatmap`."""

    plot_type: Literal["correlation_heatmap"]
    labels: list[str] | None = None
    annotate: bool = True
    colorbar: bool = True
    triangular_mask: Literal["upper", "lower"] | None = "upper"
    cmap: str = "vlag"


class BoxPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_box`."""

    plot_type: Literal["box"]
    show_means: bool = False
    notch: bool = False
    show_outliers: bool = True
    orientation: Literal["vertical", "horizontal"] = "vertical"
    positions: list[float] | None = None
    tick_labels: list[str] | None = None
    widths: float | list[float] = 0.5
    box_alpha: float | None = None
    mean_marker: str = "^"
    median_color: str = "black"
    grid_axis: Literal["x", "y", "both", "none"] = "none"
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None


class ViolinPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_violin`."""

    plot_type: Literal["violin"]
    show_means: bool = False
    show_extrema: bool = True
    show_medians: bool = True
    orientation: Literal["vertical", "horizontal"] = "vertical"
    positions: list[float] | None = None
    tick_labels: list[str] | None = None
    widths: float = 0.5
    violin_alpha: float | None = None
    quantiles: list[list[float]] | None = None
    grid_axis: Literal["x", "y", "both", "none"] = "none"
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None


class PiePlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_pie`."""

    plot_type: Literal["pie"]
    autopct: str | None = "%1.1f%%"
    startangle: float = 90
    donut_width: float | None = None
    explode: list[float] | None = None
    shadow: bool = False
    labeldistance: float = 1.1
    pctdistance: float = 0.6
    counterclock: bool = True
    normalize: bool = True
    textprops: dict[str, Any] | None = None
    wedgeprops: dict[str, Any] | None = None
    show_legend: bool = False
    legend_loc: str = "best"


class AreaPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_area`."""

    plot_type: Literal["area"]
    stacked: bool = False
    baseline: float = 0.0
    fill_alpha: float | None = None
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    xtick_labels: list[str] | None = None
    ytick_labels: list[str] | None = None
    shaded_regions: list[ShadedRegionSpecModel] | None = None


class HexbinPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_hexbin`."""

    plot_type: Literal["hexbin"]
    gridsize: int = 30
    cmap: str = "viridis"
    mincnt: int | None = 1
    colorbar: bool = True
    reduce_function: str = "mean"
    values: list[float] | None = None
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    xtick_labels: list[str] | None = None
    ytick_labels: list[str] | None = None
    colorbar_label: str | None = None
    extent: tuple[float, float, float, float] | None = None
    bins: str | None = None
    linewidths: float = 0.0
    alpha: float = 1.0


class ContourPlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_contour`."""

    plot_type: Literal["contour"]
    levels: int | list[float] = 10
    filled: bool = True
    cmap: str = "viridis"
    colorbar: bool = True
    label_contours: bool = False
    xlims: tuple[float, float] | None = None
    ylims: tuple[float, float] | None = None
    xticks: list[float] | None = None
    yticks: list[float] | None = None
    xtick_labels: list[str] | None = None
    ytick_labels: list[str] | None = None
    colorbar_label: str | None = None
    linewidths: float = 1.5
    alpha: float = 1.0
    vmin: float | None = None
    vmax: float | None = None


class TimelinePlotSpec(CommonPlotSpec):
    """PlotSpec for `plot_timeline`."""

    plot_type: Literal["timeline"]
    labels: dict[str, list[str]] | None = None
    lane_labels: list[str] | None = None
    marker_size: float = 80.0
    draw_lane_lines: bool = True
    label_offset: float = 0.08
    xlims: tuple[float, float] | None = None
    xticks: list[float] | None = None
    xtick_labels: list[str] | None = None


PLOT_SPEC_MODELS: dict[str, type[CommonPlotSpec]] = {
    "line": LinePlotSpec,
    "scatter": ScatterPlotSpec,
    "histogram": HistogramPlotSpec,
    "bar": BarPlotSpec,
    "dual_axis_bar": DualAxisBarPlotSpec,
    "grouped_bar": GroupedBarPlotSpec,
    "stacked_bar": StackedBarPlotSpec,
    "heatmap": HeatmapPlotSpec,
    "correlation_heatmap": CorrelationHeatmapPlotSpec,
    "box": BoxPlotSpec,
    "violin": ViolinPlotSpec,
    "pie": PiePlotSpec,
    "area": AreaPlotSpec,
    "hexbin": HexbinPlotSpec,
    "contour": ContourPlotSpec,
    "timeline": TimelinePlotSpec,
}


@dataclass
class RenderResult:
    """
    Function purpose:
        Store structured results from rendering a PlotSpec.

    Args:
        plot_type: Rendered plot type.
        figure_path: PNG output path when generated.
        svg_path: SVG output path when generated.
        pdf_path: PDF output path when generated.
        metadata_path: Metadata JSON output path when generated.
        warnings: Non-fatal warnings collected during rendering.
        normalized_spec: Validated and normalized spec dictionary.

    Outputs:
        Dataclass containing render outputs and metadata.
    """

    plot_type: str
    figure_path: str | None = None
    svg_path: str | None = None
    pdf_path: str | None = None
    metadata_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    normalized_spec: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Function purpose:
            Convert the render result to a JSON-serializable dictionary.

        Args:
            None.

        Outputs:
            Dictionary representation of the render result.
        """
        return {
            "plot_type": self.plot_type,
            "figure_path": self.figure_path,
            "svg_path": self.svg_path,
            "pdf_path": self.pdf_path,
            "metadata_path": self.metadata_path,
            "warnings": self.warnings,
            "normalized_spec": self.normalized_spec,
        }


def list_plot_types() -> list[str]:
    """
    Function purpose:
        List plot types supported by the structured PlotSpec renderer.

    Args:
        None.

    Outputs:
        List of supported plot type names.
    """
    return list(SUPPORTED_PLOT_TYPES)


def parse_plot_spec(spec: dict[str, Any] | CommonPlotSpec) -> CommonPlotSpec:
    """
    Function purpose:
        Validate a dictionary or return an existing Pydantic PlotSpec model.

    Args:
        spec: Candidate PlotSpec dictionary or Pydantic model.

    Outputs:
        Validated plot-specific Pydantic model.
    """
    if isinstance(spec, CommonPlotSpec):
        return spec
    if not isinstance(spec, dict):
        raise ValueError("PlotSpec must be a dictionary or PlotSpec model.")
    plot_type = spec.get("plot_type")
    if plot_type not in PLOT_SPEC_MODELS:
        raise ValueError(f"PlotSpec field 'plot_type' must be one of {SUPPORTED_PLOT_TYPES}.")
    try:
        return PLOT_SPEC_MODELS[plot_type].model_validate(spec)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def validate_plot_spec(spec: dict[str, Any] | CommonPlotSpec) -> dict[str, Any]:
    """
    Function purpose:
        Validate and normalize a PlotSpec dictionary.

    Args:
        spec: Candidate PlotSpec dictionary or Pydantic model.

    Outputs:
        Normalized PlotSpec dictionary.
    """
    return parse_plot_spec(spec).model_dump(mode="json", exclude_none=True)


def get_plot_schema(plot_type: str) -> dict[str, Any]:
    """
    Function purpose:
        Return JSON Schema for one supported plot type.

    Args:
        plot_type: Plot type to describe.

    Outputs:
        Dictionary containing JSON Schema and LLM-oriented metadata.
    """
    if plot_type not in PLOT_SPEC_MODELS:
        raise ValueError(f"Unsupported plot_type '{plot_type}'. Supported types: {SUPPORTED_PLOT_TYPES}")
    schema = PLOT_SPEC_MODELS[plot_type].model_json_schema()
    schema["data_modes"] = ["inline", "csv", "dataframe"]
    schema["llm_notes"] = [
        "Return JSON only when generating a PlotSpec.",
        "Use csv mode for local tabular files and provide explicit column mappings.",
        "Use inline mode for small arrays, matrices, scalar mappings, or examples.",
        "All optional fields map directly to the corresponding plotting function arguments.",
    ]
    return schema


def get_all_plot_schemas() -> dict[str, Any]:
    """
    Function purpose:
        Return JSON Schemas for every supported plot type.

    Args:
        None.

    Outputs:
        Mapping from plot type to JSON Schema.
    """
    return {plot_type: get_plot_schema(plot_type) for plot_type in SUPPORTED_PLOT_TYPES}
