"""Render PlotSpec dictionaries into concrete plot files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .configs import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, OutputConfig, SeriesStyle, TextStyle
from .plots.area import plot_area
from .plots.bar import plot_bar, plot_grouped_bar, plot_stacked_bar
from .plots.density import plot_contour, plot_hexbin
from .plots.distribution import plot_box, plot_violin
from .plots.heatmap import plot_correlation_heatmap, plot_heatmap
from .plots.hist import plot_histogram
from .plots.line import plot_line
from .plots.pie import plot_pie
from .plots.scatter import plot_scatter
from .plots.timeline import plot_timeline
from .specs import RenderResult, validate_plot_spec


def load_plot_spec_file(path: str | Path) -> dict[str, Any]:
    """
    Function purpose:
        Load a JSON or YAML PlotSpec file.

    Args:
        path: Path to a JSON, YAML, or YML spec file.

    Outputs:
        PlotSpec dictionary loaded from disk.
    """
    # Resolve the input path once so error messages are stable.
    spec_path = Path(path)
    if not spec_path.exists():
        raise ValueError(f"Spec file does not exist: {spec_path}")

    # Read JSON specs with the standard library.
    if spec_path.suffix.lower() == ".json":
        return json.loads(spec_path.read_text(encoding="utf-8"))

    # Load YAML only when PyYAML is available in the environment.
    if spec_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError("YAML specs require PyYAML to be installed.") from exc
        return yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    # Reject unknown file types explicitly.
    raise ValueError("Spec file must end with .json, .yaml, or .yml.")


def render_plot_file(path: str | Path, dataframes: dict[str, Any] | None = None) -> RenderResult:
    """
    Function purpose:
        Render a PlotSpec loaded from a JSON or YAML file.

    Args:
        path: Path to a PlotSpec file.
        dataframes: Optional named DataFrame-like objects for Python callers.

    Outputs:
        Structured render result containing generated paths and normalized spec.
    """
    # Load the spec from disk and pass it through the shared render path.
    spec = load_plot_spec_file(path)
    return render_plot(spec, dataframes=dataframes)


def render_plot(spec: dict[str, Any], dataframes: dict[str, Any] | None = None) -> RenderResult:
    """
    Function purpose:
        Render a validated PlotSpec dictionary using the existing plotting functions.

    Args:
        spec: PlotSpec dictionary.
        dataframes: Optional named DataFrame-like objects for specs using dataframe data.

    Outputs:
        Structured render result containing generated paths and normalized spec.
    """
    # Validate and normalize the spec before using any renderer.
    normalized = validate_plot_spec(spec)

    # Build shared configs from optional style, legend, output, and annotation sections.
    figure_style = _build_figure_style(normalized.get("style", {}))
    axis_style = _build_axis_style(normalized.get("axis", {}))
    legend_style = _build_legend_style(normalized.get("legend", {}))
    output_config = _build_output_config(normalized)
    annotations = _build_annotations(normalized.get("annotations", []))

    # Resolve inline, CSV, or DataFrame data into renderer-friendly structures.
    resolved_data = _resolve_data(normalized["data"], dataframes=dataframes)

    # Dispatch to the backend plotting function for the requested plot type.
    plot_type = normalized["plot_type"]
    if plot_type == "line":
        series = _xy_series(resolved_data)
        plot_line(series["x"], series["y"], normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, series_styles=_series_styles(series["x"]), annotations=annotations)
    elif plot_type == "scatter":
        series = _xy_series(resolved_data)
        plot_scatter(series["x"], series["y"], normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, series_styles=_series_styles(series["x"]), annotations=annotations)
    elif plot_type == "area":
        area_data = _area_series(resolved_data)
        plot_area(area_data["x"], area_data["y"], normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, series_styles=_series_styles(area_data["y"]), annotations=annotations, stacked=bool(normalized["options"].get("stacked", False)))
    elif plot_type == "histogram":
        values = _value_series(resolved_data)
        plot_histogram(values, normalized["options"].get("bins", 20), normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, series_styles=_series_styles(values), annotations=annotations, plot_kde=bool(normalized["options"].get("plot_kde", False)))
    elif plot_type == "box":
        values = _value_series(resolved_data)
        plot_box(values, normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, series_styles=_series_styles(values), annotations=annotations)
    elif plot_type == "violin":
        values = _value_series(resolved_data)
        plot_violin(values, normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, series_styles=_series_styles(values), annotations=annotations)
    elif plot_type == "bar":
        values = _scalar_values(resolved_data)
        positions = {key: float(index) for index, key in enumerate(values)}
        widths = {key: 0.7 for key in values}
        plot_bar(values, normalized["xlabel"], normalized["ylabel"], normalized["title"], positions, widths, figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, series_styles=_series_styles(values), annotations=annotations, xticks=list(positions.values()), xtick_labels=list(values.keys()), value_labels=bool(normalized["options"].get("value_labels", False)))
    elif plot_type == "pie":
        values = _scalar_values(resolved_data)
        plot_pie(values, normalized["title"], figure_style=figure_style, legend_style=legend_style, output_config=output_config, series_styles=_series_styles(values), annotations=annotations, show_legend=bool(normalized["options"].get("show_legend", True)))
    elif plot_type == "grouped_bar":
        values = _nested_values(resolved_data)
        plot_grouped_bar(values, normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, annotations=annotations, value_labels=bool(normalized["options"].get("value_labels", False)))
    elif plot_type == "stacked_bar":
        values = _nested_values(resolved_data)
        plot_stacked_bar(values, normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, annotations=annotations, normalize=bool(normalized["options"].get("normalize", False)), value_labels=bool(normalized["options"].get("value_labels", False)))
    elif plot_type == "heatmap":
        matrix = _matrix_values(resolved_data)
        plot_heatmap(matrix, normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, annotations=annotations, colorbar=bool(normalized["options"].get("colorbar", True)), annotate=bool(normalized["options"].get("annotate", True)))
    elif plot_type == "correlation_heatmap":
        matrix = _matrix_values(resolved_data)
        plot_correlation_heatmap(matrix, title=normalized["title"], labels=resolved_data.get("labels"), figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, annotations=annotations)
    elif plot_type == "hexbin":
        xy = _flat_xy(resolved_data)
        plot_hexbin(xy["x"], xy["y"], normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, annotations=annotations, colorbar_label=normalized["options"].get("colorbar_label"))
    elif plot_type == "contour":
        grid = _contour_values(resolved_data)
        plot_contour(grid["x"], grid["y"], grid["z"], normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, annotations=annotations, colorbar_label=normalized["options"].get("colorbar_label"))
    elif plot_type == "timeline":
        events = _timeline_values(resolved_data)
        plot_timeline(events["events"], normalized["xlabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, annotations=annotations, labels=events.get("labels"))
    else:
        raise ValueError(f"Unsupported plot_type '{plot_type}'.")

    # Return structured paths for generated outputs.
    return _build_render_result(normalized)


def _build_figure_style(style: dict[str, Any]) -> FigureStyle:
    """Build a FigureStyle from a spec dictionary."""
    return FigureStyle(
        figure_size=tuple(style.get("figure_size", (10, 8))),
        title_size=int(style.get("title_size", 18)),
        theme=style.get("theme", "default"),
    )


def _build_axis_style(axis: dict[str, Any]) -> AxisStyle:
    """Build an AxisStyle from a spec dictionary."""
    return AxisStyle(
        xlabel_size=int(axis.get("xlabel_size", 18)),
        ylabel_size=int(axis.get("ylabel_size", 18)),
        xtick_size=int(axis.get("xtick_size", 14)),
        ytick_size=int(axis.get("ytick_size", 14)),
    )


def _build_legend_style(legend: dict[str, Any]) -> LegendStyle:
    """Build a LegendStyle from a spec dictionary."""
    return LegendStyle(
        enabled=bool(legend.get("enabled", True)),
        loc=legend.get("loc", "best"),
        ncol=int(legend.get("ncol", 1)),
        frameon=bool(legend.get("frameon", False)),
        fontsize=int(legend.get("fontsize", 14)),
    )


def _build_output_config(spec: dict[str, Any]) -> OutputConfig:
    """Build an OutputConfig from a spec dictionary."""
    output = spec.get("output", {})
    formats = output.get("formats", ["png", "svg"])
    return OutputConfig(
        output_dir=output.get("output_dir", "plot_outputs"),
        filename=output.get("filename", spec["plot_type"]),
        save_png="png" in formats,
        save_svg="svg" in formats,
        save_pdf="pdf" in formats,
        dpi=int(output.get("dpi", 300)),
        return_fig=False,
        metadata={"plot_spec": spec},
        save_metadata=bool(output.get("save_metadata", True)),
    )


def _build_annotations(annotation_specs: list[dict[str, Any]]) -> list[AnnotationSpec]:
    """Build AnnotationSpec objects from plain dictionaries."""
    annotations = []
    for item in annotation_specs:
        style = item.get("style", {})
        annotations.append(
            AnnotationSpec(
                text=item["text"],
                xy=tuple(item.get("xy", (0.05, 0.95))),
                xycoords=item.get("xycoords", "axes fraction"),
                style=TextStyle(fontsize=int(style.get("fontsize", 14)), color=style.get("color", "black"), bold=bool(style.get("bold", False))),
            )
        )
    return annotations


def _resolve_data(data_spec: dict[str, Any], dataframes: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve inline, CSV, or DataFrame data into a normalized dictionary."""
    if "inline" in data_spec:
        return dict(data_spec["inline"])
    if "csv" in data_spec:
        csv_info = data_spec["csv"]
        rows = _read_csv_rows(Path(csv_info["path"]))
        return _map_rows(rows, csv_info["mappings"])
    if "dataframe" in data_spec:
        dataframe_info = data_spec["dataframe"]
        if dataframes is None or dataframe_info["name"] not in dataframes:
            raise ValueError(f"DataFrame '{dataframe_info['name']}' was not provided.")
        rows = _dataframe_to_rows(dataframes[dataframe_info["name"]])
        return _map_rows(rows, dataframe_info["mappings"])
    raise ValueError("PlotSpec data must contain one of: inline, csv, dataframe.")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows into dictionaries with numeric conversion."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return [{key: _coerce_value(value) for key, value in row.items()} for row in csv.DictReader(file)]


def _dataframe_to_rows(dataframe: Any) -> list[dict[str, Any]]:
    """Convert a DataFrame-like object to row dictionaries."""
    if hasattr(dataframe, "to_dict"):
        records = dataframe.to_dict(orient="records")
        return [{key: _coerce_value(value) for key, value in row.items()} for row in records]
    if isinstance(dataframe, list):
        return [{key: _coerce_value(value) for key, value in row.items()} for row in dataframe]
    raise ValueError("DataFrame-like data must provide to_dict(orient='records') or be a list of dictionaries.")


def _map_rows(rows: list[dict[str, Any]], mappings: dict[str, Any]) -> dict[str, Any]:
    """Map tabular rows into generic plotting data."""
    for column in _required_columns(mappings):
        if rows and column not in rows[0]:
            raise ValueError(f"Mapped column '{column}' is missing from data.")
    return {"rows": rows, "mappings": mappings}


def _required_columns(mappings: dict[str, Any]) -> list[str]:
    """Collect explicit column names from a mappings dictionary."""
    columns = []
    for value in mappings.values():
        if isinstance(value, str):
            columns.append(value)
        elif isinstance(value, list):
            columns.extend(value)
    return columns


def _coerce_value(value: Any) -> Any:
    """Convert numeric-looking values to floats while preserving text labels."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _xy_series(data: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Resolve line/scatter data to x and y series dictionaries."""
    if "series" in data:
        return {"x": {item["name"]: np.asarray(item["x"]) for item in data["series"]}, "y": {item["name"]: np.asarray(item["y"]) for item in data["series"]}}
    rows, mappings = data["rows"], data["mappings"]
    group_column = mappings.get("group")
    groups = sorted({row[group_column] for row in rows}) if group_column else ["series"]
    return {
        "x": {str(group): np.asarray([row[mappings["x"]] for row in rows if not group_column or row[group_column] == group]) for group in groups},
        "y": {str(group): np.asarray([row[mappings["y"]] for row in rows if not group_column or row[group_column] == group]) for group in groups},
    }


def _area_series(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve area data to x values and y series."""
    if "x" in data and "series" in data:
        return {"x": np.asarray(data["x"]), "y": {item["name"]: np.asarray(item["y"]) for item in data["series"]}}
    xy = _xy_series(data)
    first_key = next(iter(xy["x"]))
    return {"x": xy["x"][first_key], "y": xy["y"]}


def _value_series(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Resolve histogram/box/violin data to value series."""
    if "series" in data:
        return {item["name"]: np.asarray(item["values"]) for item in data["series"]}
    rows, mappings = data["rows"], data["mappings"]
    group_column = mappings.get("group")
    groups = sorted({row[group_column] for row in rows}) if group_column else ["values"]
    return {str(group): np.asarray([row[mappings["values"]] for row in rows if not group_column or row[group_column] == group]) for group in groups}


def _scalar_values(data: dict[str, Any]) -> dict[str, float]:
    """Resolve bar/pie data to label-value mappings."""
    if "values" in data:
        return {str(key): float(value) for key, value in data["values"].items()}
    rows, mappings = data["rows"], data["mappings"]
    return {str(row[mappings["label"]]): float(row[mappings["value"]]) for row in rows}


def _nested_values(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Resolve grouped/stacked bar data to category-series-value mappings."""
    if "values" in data:
        return {str(category): {str(series): float(value) for series, value in series_values.items()} for category, series_values in data["values"].items()}
    rows, mappings = data["rows"], data["mappings"]
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        category = str(row[mappings["category"]])
        series = str(row[mappings["series"]])
        output.setdefault(category, {})[series] = float(row[mappings["value"]])
    return output


def _matrix_values(data: dict[str, Any]) -> np.ndarray:
    """Resolve heatmap matrix data."""
    if "matrix" in data:
        return np.asarray(data["matrix"], dtype=float)
    rows, mappings = data["rows"], data["mappings"]
    return np.asarray([[row[column] for column in mappings["columns"]] for row in rows], dtype=float)


def _flat_xy(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Resolve flat x/y data for hexbin plots."""
    if "x" in data and "y" in data:
        return {"x": np.asarray(data["x"], dtype=float), "y": np.asarray(data["y"], dtype=float)}
    rows, mappings = data["rows"], data["mappings"]
    return {"x": np.asarray([row[mappings["x"]] for row in rows], dtype=float), "y": np.asarray([row[mappings["y"]] for row in rows], dtype=float)}


def _contour_values(data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Resolve contour grid data."""
    return {"x": np.asarray(data["x"], dtype=float), "y": np.asarray(data["y"], dtype=float), "z": np.asarray(data["z"], dtype=float)}


def _timeline_values(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve timeline event data."""
    if "events" in data:
        return {"events": data["events"], "labels": data.get("labels")}
    rows, mappings = data["rows"], data["mappings"]
    events: dict[str, list[Any]] = {}
    labels: dict[str, list[str]] = {}
    for row in rows:
        lane = str(row[mappings["lane"]])
        events.setdefault(lane, []).append(row[mappings["time"]])
        if "label" in mappings:
            labels.setdefault(lane, []).append(str(row[mappings["label"]]))
    return {"events": events, "labels": labels or None}


def _series_styles(series: dict[str, Any]) -> dict[str, SeriesStyle]:
    """Create default styles for series-like keys."""
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    return {key: SeriesStyle(color=colors[index % len(colors)], label=key) for index, key in enumerate(series.keys())}


def _build_render_result(spec: dict[str, Any]) -> RenderResult:
    """Build generated output paths from output configuration."""
    output = spec.get("output", {})
    output_dir = Path(output.get("output_dir", "plot_outputs"))
    filename = output.get("filename", spec["plot_type"])
    formats = output.get("formats", ["png", "svg"])
    return RenderResult(
        plot_type=spec["plot_type"],
        figure_path=str(output_dir / f"{filename}.png") if "png" in formats else None,
        svg_path=str(output_dir / "SVG" / f"{filename}.svg") if "svg" in formats else None,
        pdf_path=str(output_dir / f"{filename}.pdf") if "pdf" in formats else None,
        metadata_path=str(output_dir / f"{filename}.json") if output.get("save_metadata", True) else None,
        warnings=[],
        normalized_spec=spec,
    )
