import json
import subprocess
import sys

from parameterized_plot_functions import get_plot_schema, list_plot_types, render_plot, render_plot_file, validate_plot_spec
from parameterized_plot_functions.mcp_adapter import mcp_get_plot_schema, mcp_list_plot_types, mcp_render_plot, mcp_validate_plot_spec


def test_list_and_schema_helpers():
    plot_types = list_plot_types()

    assert "line" in plot_types
    assert get_plot_schema("line")["data_modes"] == ["inline", "csv", "dataframe"]


def test_validate_rejects_missing_data():
    try:
        validate_plot_spec({"plot_type": "line", "title": "bad"})
    except ValueError as exc:
        assert "data" in str(exc)
    else:
        raise AssertionError("validate_plot_spec should reject specs without data.")


def test_render_inline_line_spec(tmp_path):
    spec = {
        "plot_type": "line",
        "title": "Inline Line",
        "xlabel": "x",
        "ylabel": "y",
        "data": {"inline": {"series": [{"name": "a", "x": [0, 1, 2], "y": [0, 1, 4]}]}},
        "output": {"output_dir": str(tmp_path), "filename": "line", "formats": ["png", "svg"], "save_metadata": True},
    }

    result = render_plot(spec)

    assert result.figure_path is not None
    assert result.svg_path is not None
    assert (tmp_path / "line.png").exists()
    assert (tmp_path / "SVG" / "line.svg").exists()


def test_render_csv_scatter_spec(tmp_path):
    csv_path = tmp_path / "points.csv"
    csv_path.write_text("x,y,group\n0,0,A\n1,1,A\n0,1,B\n1,2,B\n", encoding="utf-8")

    spec = {
        "plot_type": "scatter",
        "title": "CSV Scatter",
        "xlabel": "x",
        "ylabel": "y",
        "data": {"csv": {"path": str(csv_path), "mappings": {"x": "x", "y": "y", "group": "group"}}},
        "output": {"output_dir": str(tmp_path), "filename": "scatter", "formats": ["png"], "save_metadata": False},
    }

    result = render_plot(spec)

    assert result.figure_path is not None
    assert (tmp_path / "scatter.png").exists()


def test_render_dataframe_like_bar_spec(tmp_path):
    rows = [{"label": "A", "value": 1.0}, {"label": "B", "value": 2.0}]
    spec = {
        "plot_type": "bar",
        "title": "DataFrame Bar",
        "xlabel": "label",
        "ylabel": "value",
        "data": {"dataframe": {"name": "table", "mappings": {"label": "label", "value": "value"}}},
        "output": {"output_dir": str(tmp_path), "filename": "bar", "formats": ["png"], "save_metadata": False},
    }

    result = render_plot(spec, dataframes={"table": rows})

    assert result.figure_path is not None
    assert (tmp_path / "bar.png").exists()


def test_render_plot_file_json(tmp_path):
    spec_path = tmp_path / "heatmap.json"
    spec_path.write_text(
        json.dumps(
            {
                "plot_type": "heatmap",
                "title": "Heatmap",
                "data": {"inline": {"matrix": [[1, 2], [3, 4]]}},
                "output": {"output_dir": str(tmp_path), "filename": "heatmap", "formats": ["png"], "save_metadata": False},
            }
        ),
        encoding="utf-8",
    )

    result = render_plot_file(spec_path)

    assert result.figure_path is not None
    assert (tmp_path / "heatmap.png").exists()


def test_cli_list_plots_and_schema():
    list_result = subprocess.run(
        [sys.executable, "-m", "parameterized_plot_functions.cli", "list-plots"],
        check=True,
        capture_output=True,
        text=True,
    )
    schema_result = subprocess.run(
        [sys.executable, "-m", "parameterized_plot_functions.cli", "schema", "line"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "line" in list_result.stdout
    assert "series" in schema_result.stdout


def test_mcp_adapter_functions(tmp_path):
    spec = {
        "plot_type": "pie",
        "title": "Pie",
        "data": {"inline": {"values": {"A": 1, "B": 2}}},
        "output": {"output_dir": str(tmp_path), "filename": "pie", "formats": ["png"], "save_metadata": False},
    }

    assert "line" in mcp_list_plot_types()["plot_types"]
    assert mcp_get_plot_schema("pie")["plot_type"] == "pie"
    assert mcp_validate_plot_spec(spec)["valid"] is True
    assert mcp_render_plot(spec)["ok"] is True


def test_render_all_supported_inline_plot_types(tmp_path):
    base_output = {"output_dir": str(tmp_path), "formats": ["png"], "save_metadata": False}
    specs = [
        {
            "plot_type": "line",
            "title": "line",
            "data": {"inline": {"series": [{"name": "a", "x": [0, 1, 2], "y": [0, 1, 2]}]}},
        },
        {
            "plot_type": "scatter",
            "title": "scatter",
            "data": {"inline": {"series": [{"name": "a", "x": [0, 1, 2], "y": [0, 1, 4]}]}},
        },
        {
            "plot_type": "histogram",
            "title": "histogram",
            "data": {"inline": {"series": [{"name": "a", "values": [0, 1, 1, 2]}]}},
        },
        {
            "plot_type": "bar",
            "title": "bar",
            "data": {"inline": {"values": {"A": 1, "B": 2}}},
        },
        {
            "plot_type": "grouped_bar",
            "title": "grouped",
            "data": {"inline": {"values": {"G1": {"A": 1, "B": 2}, "G2": {"A": 2, "B": 3}}}},
        },
        {
            "plot_type": "stacked_bar",
            "title": "stacked",
            "data": {"inline": {"values": {"G1": {"A": 1, "B": 2}, "G2": {"A": 2, "B": 3}}}},
        },
        {
            "plot_type": "heatmap",
            "title": "heatmap",
            "data": {"inline": {"matrix": [[1, 2], [3, 4]]}},
        },
        {
            "plot_type": "correlation_heatmap",
            "title": "correlation",
            "data": {"inline": {"matrix": [[1, 2], [2, 4]]}},
        },
        {
            "plot_type": "box",
            "title": "box",
            "data": {"inline": {"series": [{"name": "a", "values": [0, 1, 2]}]}},
        },
        {
            "plot_type": "violin",
            "title": "violin",
            "data": {"inline": {"series": [{"name": "a", "values": [0, 1, 2, 3]}]}},
        },
        {
            "plot_type": "pie",
            "title": "pie",
            "data": {"inline": {"values": {"A": 1, "B": 2}}},
        },
        {
            "plot_type": "area",
            "title": "area",
            "data": {"inline": {"x": [0, 1, 2], "series": [{"name": "a", "y": [0, 1, 2]}]}},
        },
        {
            "plot_type": "hexbin",
            "title": "hexbin",
            "data": {"inline": {"x": [0, 1, 2, 3], "y": [0, 1, 1, 2]}},
        },
        {
            "plot_type": "contour",
            "title": "contour",
            "data": {"inline": {"x": [[0, 1], [0, 1]], "y": [[0, 0], [1, 1]], "z": [[0, 1], [1, 2]]}},
        },
        {
            "plot_type": "timeline",
            "title": "timeline",
            "xlabel": "time",
            "data": {"inline": {"events": {"A": [1, 2], "B": [2, 3]}}},
        },
    ]

    for spec in specs:
        spec["output"] = {**base_output, "filename": spec["plot_type"]}
        result = render_plot(spec)
        assert result.figure_path is not None
        assert (tmp_path / f"{spec['plot_type']}.png").exists()
