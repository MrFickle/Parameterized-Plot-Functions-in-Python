"""MCP-ready adapter functions for PlotSpec operations."""

from __future__ import annotations

from typing import Any

from .render import render_plot
from .specs import get_plot_schema, list_plot_types, validate_plot_spec


def mcp_list_plot_types() -> dict[str, Any]:
    """
    Function purpose:
        Return supported plot types in an MCP-tool-friendly shape.

    Args:
        None.

    Outputs:
        Dictionary containing supported plot types.
    """
    # Wrap the list so MCP clients receive a named field.
    return {"plot_types": list_plot_types()}


def mcp_get_plot_schema(plot_type: str) -> dict[str, Any]:
    """
    Function purpose:
        Return schema-like PlotSpec information for MCP clients.

    Args:
        plot_type: Plot type to describe.

    Outputs:
        Dictionary containing the requested plot schema.
    """
    # Delegate to the same schema helper used by the CLI.
    return {"plot_type": plot_type, "schema": get_plot_schema(plot_type)}


def mcp_validate_plot_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Function purpose:
        Validate a PlotSpec dictionary for MCP clients.

    Args:
        spec: PlotSpec dictionary.

    Outputs:
        Dictionary with validation status and normalized spec or error text.
    """
    # Return errors as data because MCP tools commonly need structured failures.
    try:
        return {"valid": True, "normalized_spec": validate_plot_spec(spec), "error": None}
    except ValueError as exc:
        return {"valid": False, "normalized_spec": None, "error": str(exc)}


def mcp_render_plot(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Function purpose:
        Render a PlotSpec dictionary for MCP clients.

    Args:
        spec: PlotSpec dictionary.

    Outputs:
        Dictionary with render status, paths, and any error text.
    """
    # Return errors as data instead of raising across a tool boundary.
    try:
        return {"ok": True, "result": render_plot(spec).to_dict(), "error": None}
    except ValueError as exc:
        return {"ok": False, "result": None, "error": str(exc)}
