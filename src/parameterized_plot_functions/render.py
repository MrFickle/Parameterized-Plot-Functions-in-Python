"""Render PlotSpec dictionaries into concrete plot files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

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
from .plots.area import plot_area
from .plots.bar import plot_bar, plot_dual_axis_bar, plot_grouped_bar, plot_stacked_bar
from .plots.density import plot_contour, plot_hexbin
from .plots.distribution import plot_box, plot_violin
from .plots.heatmap import plot_correlation_heatmap, plot_heatmap
from .plots.hist import plot_histogram
from .plots.line import plot_line
from .plots.pie import plot_pie
from .plots.scatter import plot_scatter
from .plots.timeline import plot_timeline
from .specs import CommonPlotSpec, RenderResult, parse_plot_spec


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


def render_plot(spec: dict[str, Any] | CommonPlotSpec, dataframes: dict[str, Any] | None = None) -> RenderResult:
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
    spec_model = parse_plot_spec(spec)
    normalized = spec_model.model_dump(mode="json", exclude_none=True)

    # Build shared configs from optional style, legend, output, and annotation sections.
    figure_style = _build_figure_style(normalized.get("style", {}))
    axis_style = _build_axis_style(normalized.get("axis", {}))
    legend_style = _build_legend_style(normalized.get("legend", {}))
    output_config = _build_output_config(normalized)
    annotations = _build_annotations(normalized.get("annotations", []))
    series_styles = _build_series_styles(normalized.get("series_styles"))
    line_spec = _build_line_spec(normalized.get("line_spec"))

    # Resolve inline, CSV, or DataFrame data into renderer-friendly structures.
    resolved_data = _resolve_data(normalized["data"], dataframes=dataframes)

    # Dispatch to the backend plotting function for the requested plot type.
    plot_type = normalized["plot_type"]
    if plot_type == "line":
        series = _xy_series(resolved_data)
        plot_line(
            series["x"],
            series["y"],
            normalized["xlabel"],
            normalized["ylabel"],
            normalized["title"],
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            series_styles=series_styles or _series_styles(series["x"]),
            yerr_series=_array_mapping(normalized.get("yerr_series")),
            ci_series=_ci_mapping(normalized.get("ci_series")),
            use_fill_between=bool(_option(normalized, "use_fill_between", True)),
            step_where=normalized.get("step_where"),
            rolling_window=normalized.get("rolling_window"),
            xlims=_tuple_or_none(normalized.get("xlims")),
            ylims=_tuple_or_none(normalized.get("ylims")),
            xticks=normalized.get("xticks"),
            yticks=normalized.get("yticks"),
            xtick_labels=normalized.get("xtick_labels"),
            ytick_labels=normalized.get("ytick_labels"),
            annotations=annotations,
            line_spec=line_spec,
            shaded_regions=_build_shaded_regions(normalized.get("shaded_regions")),
            endpoint_labels=bool(_option(normalized, "endpoint_labels", False)),
            use_mask=bool(_option(normalized, "use_mask", True)),
            errorbar_capsize=float(_option(normalized, "errorbar_capsize", 4)),
            errorbar_elinewidth=float(_option(normalized, "errorbar_elinewidth", 2)),
            errorbar_capthick=float(_option(normalized, "errorbar_capthick", 2)),
            use_line_color_for_error=bool(_option(normalized, "use_line_color_for_error", False)),
            rotate_xticks=bool(_option(normalized, "rotate_xticks", False)),
            plot_minor_ticks=bool(_option(normalized, "plot_minor_ticks", False)),
            extend_y_one_tick=bool(_option(normalized, "extend_y_one_tick", False)),
        )
    elif plot_type == "scatter":
        series = _xy_series(resolved_data)
        plot_scatter(
            series["x"],
            series["y"],
            normalized["xlabel"],
            normalized["ylabel"],
            normalized["title"],
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            series_styles=series_styles or _series_styles(series["x"]),
            xlims=_tuple_or_none(normalized.get("xlims")),
            ylims=_tuple_or_none(normalized.get("ylims")),
            xticks=normalized.get("xticks"),
            yticks=normalized.get("yticks"),
            xtick_labels=normalized.get("xtick_labels"),
            ytick_labels=normalized.get("ytick_labels"),
            annotations=annotations,
            line_spec=line_spec,
            do_linear_reg_fit=bool(_option(normalized, "do_linear_reg_fit", False)),
            plot_r2_score=bool(_option(normalized, "plot_r2_score", False)),
            polynomial_degree=normalized.get("polynomial_degree"),
            regression_confidence_band=bool(_option(normalized, "regression_confidence_band", False)),
            point_labels=normalized.get("point_labels"),
            jitter=float(_option(normalized, "jitter", 0.0)),
            size_values=_array_mapping(normalized.get("size_values")),
            colorbar_config=_build_colorbar_config(normalized.get("colorbar_config")),
            color_values=_array_mapping(normalized.get("color_values")),
        )
    elif plot_type == "area":
        area_data = _area_series(resolved_data)
        plot_area(
            area_data["x"],
            area_data["y"],
            normalized["xlabel"],
            normalized["ylabel"],
            normalized["title"],
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            series_styles=series_styles or _series_styles(area_data["y"]),
            annotations=annotations,
            stacked=bool(_option(normalized, "stacked", False)),
            baseline=float(_option(normalized, "baseline", 0.0)),
            fill_alpha=normalized.get("fill_alpha"),
            xlims=_tuple_or_none(normalized.get("xlims")),
            ylims=_tuple_or_none(normalized.get("ylims")),
            xticks=normalized.get("xticks"),
            yticks=normalized.get("yticks"),
            xtick_labels=normalized.get("xtick_labels"),
            ytick_labels=normalized.get("ytick_labels"),
            line_spec=line_spec,
            shaded_regions=_build_shaded_regions(normalized.get("shaded_regions")),
        )
    elif plot_type == "histogram":
        values = _value_series(resolved_data)
        plot_histogram(
            values,
            _option(normalized, "bins", 20),
            normalized["xlabel"],
            normalized["ylabel"],
            normalized["title"],
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            series_styles=series_styles or _series_styles(values),
            hist_stat=str(_option(normalized, "hist_stat", "probability")),
            plot_kde=bool(_option(normalized, "plot_kde", False)),
            plot_mean=bool(_option(normalized, "plot_mean", False)),
            plot_std=bool(_option(normalized, "plot_std", False)),
            cumulative=bool(_option(normalized, "cumulative", False)),
            fitted_distribution=normalized.get("fitted_distribution"),
            percentile_markers=normalized.get("percentile_markers"),
            perform_dip_test=bool(_option(normalized, "perform_dip_test", False)),
            vertical_lines=normalized.get("vertical_lines"),
            xlims=_tuple_or_none(normalized.get("xlims")),
            ylims=_tuple_or_none(normalized.get("ylims")),
            xticks=normalized.get("xticks"),
            yticks=normalized.get("yticks"),
            xtick_labels=normalized.get("xtick_labels"),
            ytick_labels=normalized.get("ytick_labels"),
            annotations=annotations,
            line_spec=line_spec,
            extend_y_one_tick=bool(_option(normalized, "extend_y_one_tick", False)),
        )
    elif plot_type == "box":
        values = _value_series(resolved_data)
        plot_box(
            values,
            normalized["xlabel"],
            normalized["ylabel"],
            normalized["title"],
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            series_styles=series_styles or _series_styles(values),
            annotations=annotations,
            show_means=bool(_option(normalized, "show_means", False)),
            notch=bool(_option(normalized, "notch", False)),
            show_outliers=bool(_option(normalized, "show_outliers", True)),
            orientation=str(_option(normalized, "orientation", "vertical")),
            positions=normalized.get("positions"),
            tick_labels=normalized.get("tick_labels"),
            widths=_option(normalized, "widths", 0.5),
            box_alpha=normalized.get("box_alpha"),
            mean_marker=str(_option(normalized, "mean_marker", "^")),
            median_color=str(_option(normalized, "median_color", "black")),
            grid_axis=str(_option(normalized, "grid_axis", "none")),
            xlims=_tuple_or_none(normalized.get("xlims")),
            ylims=_tuple_or_none(normalized.get("ylims")),
        )
    elif plot_type == "violin":
        values = _value_series(resolved_data)
        plot_violin(
            values,
            normalized["xlabel"],
            normalized["ylabel"],
            normalized["title"],
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            series_styles=series_styles or _series_styles(values),
            annotations=annotations,
            show_means=bool(_option(normalized, "show_means", False)),
            show_extrema=bool(_option(normalized, "show_extrema", True)),
            show_medians=bool(_option(normalized, "show_medians", True)),
            orientation=str(_option(normalized, "orientation", "vertical")),
            positions=normalized.get("positions"),
            tick_labels=normalized.get("tick_labels"),
            widths=float(_option(normalized, "widths", 0.5)),
            violin_alpha=normalized.get("violin_alpha"),
            quantiles=normalized.get("quantiles"),
            grid_axis=str(_option(normalized, "grid_axis", "none")),
            xlims=_tuple_or_none(normalized.get("xlims")),
            ylims=_tuple_or_none(normalized.get("ylims")),
        )
    elif plot_type == "bar":
        values = _scalar_values(resolved_data)
        positions = {key: float(index) for index, key in enumerate(values)}
        widths = {key: 0.7 for key in values}
        positions = normalized.get("x_positions") or positions
        widths = normalized.get("bar_widths") or widths
        plot_bar(
            values,
            normalized["xlabel"],
            normalized["ylabel"],
            normalized["title"],
            positions,
            widths,
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            series_styles=series_styles or _series_styles(values),
            annotations=annotations,
            sem_values=normalized.get("sem_values"),
            xticks=normalized.get("xticks", list(positions.values())),
            yticks=normalized.get("yticks"),
            xtick_labels=normalized.get("xtick_labels", list(values.keys())),
            ytick_labels=normalized.get("ytick_labels"),
            xlims=_tuple_or_none(normalized.get("xlims")),
            ylims=_tuple_or_none(normalized.get("ylims")),
            edgecolor=normalized.get("edgecolor"),
            horizontal=bool(_option(normalized, "horizontal", False)),
            sort_values=bool(_option(normalized, "sort_values", False)),
            value_labels=bool(_option(normalized, "value_labels", False)),
            significance_brackets=_build_significance_brackets(normalized.get("significance_brackets")),
            rotate_xticks=bool(_option(normalized, "rotate_xticks", False)),
            plot_minor_ticks=bool(_option(normalized, "plot_minor_ticks", False)),
            extend_y_one_tick=bool(_option(normalized, "extend_y_one_tick", False)),
        )
    elif plot_type == "dual_axis_bar":
        values = _scalar_values(resolved_data)
        positions = normalized.get("x_positions") or {key: float(index) for index, key in enumerate(values)}
        widths = normalized.get("bar_widths") or {key: 0.7 for key in values}
        axis_assignment = normalized.get("axis_assignment") or {key: "left" for key in values}
        plot_dual_axis_bar(
            values,
            normalized["xlabel"],
            str(_option(normalized, "ylabel_left", normalized.get("ylabel", ""))),
            str(_option(normalized, "ylabel_right", "")),
            normalized["title"],
            positions,
            widths,
            axis_assignment,
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            series_styles=series_styles or _series_styles(values),
            xticks=normalized.get("xticks", list(positions.values())),
            xtick_labels=normalized.get("xtick_labels", list(values.keys())),
            yticks_left=normalized.get("yticks_left"),
            yticks_right=normalized.get("yticks_right"),
            ylims_left=_tuple_or_none(normalized.get("ylims_left")),
            ylims_right=_tuple_or_none(normalized.get("ylims_right")),
            sem_values=normalized.get("sem_values"),
            edgecolor=normalized.get("edgecolor"),
            rotate_xticks=bool(_option(normalized, "rotate_xticks", False)),
            plot_minor_ticks=bool(_option(normalized, "plot_minor_ticks", False)),
            extend_y_one_tick=bool(_option(normalized, "extend_y_one_tick", False)),
            annotations=annotations,
        )
    elif plot_type == "pie":
        values = _scalar_values(resolved_data)
        plot_pie(
            values,
            normalized["title"],
            figure_style=figure_style,
            legend_style=legend_style,
            output_config=output_config,
            series_styles=series_styles or _series_styles(values),
            annotations=annotations,
            autopct=normalized.get("autopct", "%1.1f%%"),
            startangle=float(_option(normalized, "startangle", 90)),
            donut_width=normalized.get("donut_width"),
            explode=normalized.get("explode"),
            shadow=bool(_option(normalized, "shadow", False)),
            labeldistance=float(_option(normalized, "labeldistance", 1.1)),
            pctdistance=float(_option(normalized, "pctdistance", 0.6)),
            counterclock=bool(_option(normalized, "counterclock", True)),
            normalize=bool(_option(normalized, "normalize", True)),
            textprops=normalized.get("textprops"),
            wedgeprops=normalized.get("wedgeprops"),
            show_legend=bool(_option(normalized, "show_legend", False)),
            legend_loc=str(_option(normalized, "legend_loc", "best")),
        )
    elif plot_type == "grouped_bar":
        values = _nested_values(resolved_data)
        plot_grouped_bar(values, normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, series_styles=series_styles, annotations=annotations, value_labels=bool(_option(normalized, "value_labels", False)), group_gap=float(_option(normalized, "group_gap", 1.0)), bar_width=float(_option(normalized, "bar_width", 0.8)))
    elif plot_type == "stacked_bar":
        values = _nested_values(resolved_data)
        plot_stacked_bar(values, normalized["xlabel"], normalized["ylabel"], normalized["title"], figure_style=figure_style, axis_style=axis_style, legend_style=legend_style, output_config=output_config, series_styles=series_styles, annotations=annotations, normalize=bool(_option(normalized, "normalize", False)), value_labels=bool(_option(normalized, "value_labels", False)))
    elif plot_type == "heatmap":
        matrix = _matrix_values(resolved_data)
        plot_heatmap(
            matrix,
            normalized["xlabel"],
            normalized["ylabel"],
            normalized["title"],
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            annotations=annotations,
            annotate=bool(_option(normalized, "annotate", True)),
            colorbar=bool(_option(normalized, "colorbar", False)),
            vmin=normalized.get("vmin"),
            vmax=normalized.get("vmax"),
            cmap=str(_option(normalized, "cmap", "rocket")),
            rotate_ticks=bool(_option(normalized, "rotate_ticks", False)),
            xtick_labels=normalized.get("xtick_labels"),
            ytick_labels=normalized.get("ytick_labels"),
            triangular_mask=normalized.get("triangular_mask"),
            center=normalized.get("center"),
            normalize=normalized.get("normalize"),
            auto_text_contrast=bool(_option(normalized, "auto_text_contrast", False)),
        )
    elif plot_type == "correlation_heatmap":
        matrix = _matrix_values(resolved_data)
        plot_correlation_heatmap(
            matrix,
            title=normalized["title"],
            labels=normalized.get("labels") or resolved_data.get("labels"),
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            annotations=annotations,
            annotate=bool(_option(normalized, "annotate", True)),
            colorbar=bool(_option(normalized, "colorbar", True)),
            triangular_mask=normalized.get("triangular_mask", "upper"),
            cmap=str(_option(normalized, "cmap", "vlag")),
        )
    elif plot_type == "hexbin":
        xy = _flat_xy(resolved_data)
        plot_hexbin(
            xy["x"],
            xy["y"],
            normalized["xlabel"],
            normalized["ylabel"],
            normalized["title"],
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            annotations=annotations,
            gridsize=int(_option(normalized, "gridsize", 30)),
            cmap=str(_option(normalized, "cmap", "viridis")),
            mincnt=normalized.get("mincnt", 1),
            colorbar=bool(_option(normalized, "colorbar", True)),
            reduce_function=_reduce_function(str(_option(normalized, "reduce_function", "mean"))),
            values=np.asarray(normalized["values"], dtype=float) if "values" in normalized else None,
            xlims=_tuple_or_none(normalized.get("xlims")),
            ylims=_tuple_or_none(normalized.get("ylims")),
            xticks=normalized.get("xticks"),
            yticks=normalized.get("yticks"),
            xtick_labels=normalized.get("xtick_labels"),
            ytick_labels=normalized.get("ytick_labels"),
            colorbar_label=normalized.get("colorbar_label"),
            extent=_tuple_or_none(normalized.get("extent")),
            bins=normalized.get("bins"),
            linewidths=float(_option(normalized, "linewidths", 0.0)),
            alpha=float(_option(normalized, "alpha", 1.0)),
            line_spec=line_spec,
        )
    elif plot_type == "contour":
        grid = _contour_values(resolved_data)
        plot_contour(
            grid["x"],
            grid["y"],
            grid["z"],
            normalized["xlabel"],
            normalized["ylabel"],
            normalized["title"],
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            annotations=annotations,
            levels=_option(normalized, "levels", 10),
            filled=bool(_option(normalized, "filled", True)),
            cmap=str(_option(normalized, "cmap", "viridis")),
            colorbar=bool(_option(normalized, "colorbar", True)),
            label_contours=bool(_option(normalized, "label_contours", False)),
            xlims=_tuple_or_none(normalized.get("xlims")),
            ylims=_tuple_or_none(normalized.get("ylims")),
            xticks=normalized.get("xticks"),
            yticks=normalized.get("yticks"),
            xtick_labels=normalized.get("xtick_labels"),
            ytick_labels=normalized.get("ytick_labels"),
            colorbar_label=normalized.get("colorbar_label"),
            linewidths=float(_option(normalized, "linewidths", 1.5)),
            alpha=float(_option(normalized, "alpha", 1.0)),
            vmin=normalized.get("vmin"),
            vmax=normalized.get("vmax"),
            line_spec=line_spec,
        )
    elif plot_type == "timeline":
        events = _timeline_values(resolved_data)
        plot_timeline(
            events["events"],
            normalized["xlabel"],
            normalized["title"],
            figure_style=figure_style,
            axis_style=axis_style,
            legend_style=legend_style,
            output_config=output_config,
            series_styles=series_styles,
            annotations=annotations,
            labels=normalized.get("labels") or events.get("labels"),
            lane_labels=normalized.get("lane_labels"),
            marker_size=float(_option(normalized, "marker_size", 80.0)),
            draw_lane_lines=bool(_option(normalized, "draw_lane_lines", True)),
            label_offset=float(_option(normalized, "label_offset", 0.08)),
            xlims=_tuple_or_none(normalized.get("xlims")),
            xticks=normalized.get("xticks"),
            xtick_labels=normalized.get("xtick_labels"),
            line_spec=line_spec,
        )
    else:
        raise ValueError(f"Unsupported plot_type '{plot_type}'.")

    # Return structured paths for generated outputs.
    return _build_render_result(normalized)


def _build_figure_style(style: dict[str, Any]) -> FigureStyle:
    """Build a FigureStyle from a spec dictionary."""
    if "figure_size" in style:
        style = {**style, "figure_size": tuple(style["figure_size"])}
    return FigureStyle(**style)


def _build_axis_style(axis: dict[str, Any]) -> AxisStyle:
    """Build an AxisStyle from a spec dictionary."""
    return AxisStyle(**axis)


def _build_legend_style(legend: dict[str, Any]) -> LegendStyle:
    """Build a LegendStyle from a spec dictionary."""
    if "bbox_to_anchor" in legend and legend["bbox_to_anchor"] is not None:
        legend = {**legend, "bbox_to_anchor": tuple(legend["bbox_to_anchor"])}
    return LegendStyle(**legend)


def _build_output_config(spec: dict[str, Any]) -> OutputConfig:
    """Build an OutputConfig from a spec dictionary."""
    output = spec.get("output", {})
    formats = output.get("formats", ["png", "svg"])
    return OutputConfig(
        output_dir=output.get("output_dir", "plot_outputs"),
        filename=output.get("filename") or spec["plot_type"],
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


def _build_series_styles(style_specs: dict[str, dict[str, Any]] | None) -> dict[str, SeriesStyle] | None:
    """Build SeriesStyle objects from plain dictionaries."""
    if style_specs is None:
        return None
    return {key: SeriesStyle(**value) for key, value in style_specs.items()}


def _build_line_spec(line_spec: dict[str, Any] | None) -> LineSpec | None:
    """Build a LineSpec from a plain dictionary."""
    if line_spec is None:
        return None
    return LineSpec(
        vertical=[ReferenceLineSpec(**item) for item in line_spec.get("vertical", [])],
        horizontal=[ReferenceLineSpec(**item) for item in line_spec.get("horizontal", [])],
    )


def _build_shaded_regions(region_specs: list[dict[str, Any]] | None) -> list[ShadedRegionSpec] | None:
    """Build ShadedRegionSpec objects from plain dictionaries."""
    if region_specs is None:
        return None
    return [ShadedRegionSpec(**item) for item in region_specs]


def _build_significance_brackets(bracket_specs: list[dict[str, Any]] | None) -> list[SignificanceBracketSpec] | None:
    """Build SignificanceBracketSpec objects from plain dictionaries."""
    if bracket_specs is None:
        return None
    return [SignificanceBracketSpec(**item) for item in bracket_specs]


def _build_colorbar_config(colorbar_spec: dict[str, Any] | None) -> ColorbarConfig | None:
    """Build a ColorbarConfig from a plain dictionary."""
    if colorbar_spec is None:
        return None
    return ColorbarConfig(**colorbar_spec)


def _tuple_or_none(value: Any) -> Any:
    """Convert JSON lists to tuples where plotting functions expect tuples."""
    if value is None:
        return None
    return tuple(value)


def _array_mapping(value: dict[str, Any] | None) -> dict[str, np.ndarray] | None:
    """Convert a mapping of JSON arrays to NumPy arrays."""
    if value is None:
        return None
    return {key: np.asarray(items) for key, items in value.items()}


def _ci_mapping(value: dict[str, Any] | None) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Convert confidence interval JSON arrays to NumPy array tuples."""
    if value is None:
        return None
    return {key: (np.asarray(bounds[0]), np.asarray(bounds[1])) for key, bounds in value.items()}


def _reduce_function(name: str) -> Any:
    """Map JSON-safe reducer names to NumPy callables."""
    reducers = {"mean": np.mean, "sum": np.sum, "min": np.min, "max": np.max, "median": np.median}
    if name not in reducers:
        raise ValueError(f"Unsupported reduce_function '{name}'. Supported values: {sorted(reducers)}")
    return reducers[name]


def _option(spec: dict[str, Any], key: str, default: Any) -> Any:
    """Read a plot-specific field with fallback to deprecated options."""
    if key in spec:
        return spec[key]
    return spec.get("options", {}).get(key, default)


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
