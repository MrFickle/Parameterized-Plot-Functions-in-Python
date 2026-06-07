from __future__ import annotations

import subprocess
import sys

from parameterized_plot_functions import LLMPlotConfig, get_all_plot_schemas, get_plot_schema, plot_from_instructions
from parameterized_plot_functions.mcp_server import build_mcp_server


def test_plot_schema_exposes_plot_function_parameters():
    schema = get_plot_schema("histogram")
    properties = schema["properties"]

    assert "plot_kde" in properties
    assert "fitted_distribution" in properties
    assert "percentile_markers" in properties
    assert "line_spec" in properties


def test_all_plot_schemas_include_scatter_and_line():
    schemas = get_all_plot_schemas()

    assert "scatter" in schemas
    assert "line" in schemas
    assert "do_linear_reg_fit" in schemas["scatter"]["properties"]
    assert "rolling_window" in schemas["line"]["properties"]


def test_cli_schema_all_outputs_formal_schemas():
    result = subprocess.run(
        [sys.executable, "-m", "parameterized_plot_functions.cli", "schema", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "histogram" in result.stdout
    assert "plot_kde" in result.stdout


def test_mcp_server_can_be_constructed():
    server = build_mcp_server()

    assert server is not None


def test_plot_from_instructions_uses_cache_without_second_llm_call(tmp_path, monkeypatch):
    calls = {"count": 0}

    def fake_call(prompt, config):
        calls["count"] += 1
        return {
            "plot_type": "scatter",
            "title": "Cached Scatter",
            "xlabel": "x",
            "ylabel": "y",
            "data": {"dataframe": {"name": "data", "mappings": {"x": "x", "y": "y"}}},
            "output": {"output_dir": str(tmp_path), "filename": "cached_scatter", "formats": ["png"], "save_metadata": False},
        }

    monkeypatch.setattr("parameterized_plot_functions.llm._call_openai_for_spec", fake_call)

    config = LLMPlotConfig(api_key="test-key", cache_dir=str(tmp_path / "cache"))
    rows = [{"x": 0, "y": 0}, {"x": 1, "y": 2}, {"x": 2, "y": 4}]

    first_result = plot_from_instructions(rows, "plot x against y", config=config)
    second_result = plot_from_instructions(rows, "plot x against y", config=config)

    assert calls["count"] == 1
    assert first_result.figure_path == second_result.figure_path
    assert (tmp_path / "cached_scatter.png").exists()
