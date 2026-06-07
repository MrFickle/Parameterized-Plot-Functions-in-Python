"""Structured PlotSpec validation and schema helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_PLOT_TYPES = [
    "line",
    "scatter",
    "histogram",
    "bar",
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
        # Return only plain Python containers for CLI and MCP adapter output.
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
    # Return a copy so callers cannot mutate the module-level registry.
    return list(SUPPORTED_PLOT_TYPES)


def get_plot_schema(plot_type: str) -> dict[str, Any]:
    """
    Function purpose:
        Return a compact schema-like description for one supported plot type.

    Args:
        plot_type: Plot type to describe.

    Outputs:
        Dictionary describing required fields and accepted data mappings.
    """
    # Validate the requested plot type before returning a schema.
    if plot_type not in SUPPORTED_PLOT_TYPES:
        raise ValueError(f"Unsupported plot_type '{plot_type}'. Supported types: {SUPPORTED_PLOT_TYPES}")

    # Describe common fields shared across all PlotSpecs.
    base_schema: dict[str, Any] = {
        "required": ["plot_type", "title", "data"],
        "common_optional": ["xlabel", "ylabel", "style", "legend", "output", "annotations"],
        "data_modes": ["inline", "csv", "dataframe"],
        "output": {
            "output_dir": "Directory for generated files.",
            "filename": "Base filename without extension.",
            "formats": ["png", "svg", "pdf"],
        },
    }

    # Define concise data expectations per plot family.
    data_schemas: dict[str, dict[str, Any]] = {
        "line": {"inline": {"series": [{"name": "str", "x": ["number"], "y": ["number"]}]}, "mapping": {"x": "column", "y": "column", "group": "optional column"}},
        "scatter": {"inline": {"series": [{"name": "str", "x": ["number"], "y": ["number"]}]}, "mapping": {"x": "column", "y": "column", "group": "optional column"}},
        "area": {"inline": {"x": ["number"], "series": [{"name": "str", "y": ["number"]}]}, "mapping": {"x": "column", "y": "column", "group": "optional column"}},
        "histogram": {"inline": {"series": [{"name": "str", "values": ["number"]}]}, "mapping": {"values": "column", "group": "optional column"}},
        "box": {"inline": {"series": [{"name": "str", "values": ["number"]}]}, "mapping": {"values": "column", "group": "column"}},
        "violin": {"inline": {"series": [{"name": "str", "values": ["number"]}]}, "mapping": {"values": "column", "group": "column"}},
        "bar": {"inline": {"values": {"label": "number"}}, "mapping": {"label": "column", "value": "column"}},
        "pie": {"inline": {"values": {"label": "number"}}, "mapping": {"label": "column", "value": "column"}},
        "grouped_bar": {"inline": {"values": {"category": {"series": "number"}}}, "mapping": {"category": "column", "series": "column", "value": "column"}},
        "stacked_bar": {"inline": {"values": {"category": {"series": "number"}}}, "mapping": {"category": "column", "series": "column", "value": "column"}},
        "heatmap": {"inline": {"matrix": [["number"]]}, "mapping": {"columns": ["numeric columns"]}},
        "correlation_heatmap": {"inline": {"matrix": [["number"]]}, "mapping": {"columns": ["numeric columns"]}},
        "hexbin": {"inline": {"x": ["number"], "y": ["number"]}, "mapping": {"x": "column", "y": "column"}},
        "contour": {"inline": {"x": [["number"]], "y": [["number"]], "z": [["number"]]}, "mapping": "inline grid recommended"},
        "timeline": {"inline": {"events": {"lane": ["number"]}}, "mapping": {"lane": "column", "time": "column", "label": "optional column"}},
    }

    # Attach the plot-specific data shape to the common schema.
    base_schema["data"] = data_schemas[plot_type]
    return base_schema


def validate_plot_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Function purpose:
        Validate and normalize a PlotSpec dictionary.

    Args:
        spec: Candidate PlotSpec dictionary.

    Outputs:
        Normalized PlotSpec dictionary.
    """
    # Ensure the top-level object is a mapping.
    if not isinstance(spec, dict):
        raise ValueError("PlotSpec must be a dictionary.")

    # Validate required top-level fields.
    plot_type = spec.get("plot_type")
    if plot_type not in SUPPORTED_PLOT_TYPES:
        raise ValueError(f"PlotSpec field 'plot_type' must be one of {SUPPORTED_PLOT_TYPES}.")
    if not spec.get("title"):
        raise ValueError("PlotSpec field 'title' is required.")
    if "data" not in spec:
        raise ValueError("PlotSpec field 'data' is required.")
    if not isinstance(spec["data"], dict):
        raise ValueError("PlotSpec field 'data' must be a dictionary.")

    # Copy the spec so downstream code can add defaults without mutating caller input.
    normalized = dict(spec)

    # Fill common optional text defaults.
    normalized.setdefault("xlabel", "")
    normalized.setdefault("ylabel", "")
    normalized.setdefault("style", {})
    normalized.setdefault("legend", {})
    normalized.setdefault("output", {})
    normalized.setdefault("options", {})
    normalized.setdefault("annotations", [])

    # Validate output formats when present.
    output = normalized["output"]
    if not isinstance(output, dict):
        raise ValueError("PlotSpec field 'output' must be a dictionary when provided.")
    formats = output.get("formats", ["png", "svg"])
    if not isinstance(formats, list):
        raise ValueError("PlotSpec output.formats must be a list.")
    unsupported_formats = [fmt for fmt in formats if fmt not in {"png", "svg", "pdf"}]
    if unsupported_formats:
        raise ValueError(f"Unsupported output formats: {unsupported_formats}.")

    # Validate CSV references early enough to produce clear user-facing errors.
    data = normalized["data"]
    if "csv" in data:
        csv_info = data["csv"]
        if not isinstance(csv_info, dict):
            raise ValueError("PlotSpec data.csv must be a dictionary.")
        if not csv_info.get("path"):
            raise ValueError("PlotSpec data.csv.path is required.")
        if not Path(csv_info["path"]).exists():
            raise ValueError(f"CSV file does not exist: {csv_info['path']}")
        if not isinstance(csv_info.get("mappings"), dict):
            raise ValueError("PlotSpec data.csv.mappings must be provided as a dictionary.")

    # Validate DataFrame references for Python callers.
    if "dataframe" in data:
        dataframe_info = data["dataframe"]
        if not isinstance(dataframe_info, dict):
            raise ValueError("PlotSpec data.dataframe must be a dictionary.")
        if not dataframe_info.get("name"):
            raise ValueError("PlotSpec data.dataframe.name is required.")
        if not isinstance(dataframe_info.get("mappings"), dict):
            raise ValueError("PlotSpec data.dataframe.mappings must be provided as a dictionary.")

    return normalized
