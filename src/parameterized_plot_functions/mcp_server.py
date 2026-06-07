"""Runnable MCP server for PlotSpec-based plotting tools."""

from __future__ import annotations

from typing import Any

from .mcp_adapter import mcp_get_plot_schema, mcp_list_plot_types, mcp_render_plot, mcp_validate_plot_spec


def build_mcp_server() -> Any:
    """
    Function purpose:
        Build a FastMCP server exposing PlotSpec plotting tools.

    Args:
        None.

    Outputs:
        Configured FastMCP server instance.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError("The MCP server requires the optional dependency: pip install 'parameterized-plot-functions-in-python[mcp]'") from exc

    server = FastMCP("parameterized-plot-functions")

    @server.tool()
    def list_plot_types() -> dict[str, Any]:
        """
        Function purpose:
            List supported PlotSpec plot types.

        Args:
            None.

        Outputs:
            Dictionary containing supported plot types.
        """
        return mcp_list_plot_types()

    @server.tool()
    def get_plot_schema(plot_type: str) -> dict[str, Any]:
        """
        Function purpose:
            Return JSON Schema for a supported PlotSpec plot type.

        Args:
            plot_type: Supported plot type name.

        Outputs:
            Dictionary containing the requested plot schema.
        """
        return mcp_get_plot_schema(plot_type)

    @server.tool()
    def validate_plot_spec(spec: dict[str, Any]) -> dict[str, Any]:
        """
        Function purpose:
            Validate and normalize a PlotSpec dictionary.

        Args:
            spec: Candidate PlotSpec dictionary.

        Outputs:
            Dictionary containing validation status and normalized spec.
        """
        return mcp_validate_plot_spec(spec)

    @server.tool()
    def render_plot(spec: dict[str, Any]) -> dict[str, Any]:
        """
        Function purpose:
            Render a PlotSpec dictionary to configured output files.

        Args:
            spec: PlotSpec dictionary.

        Outputs:
            Dictionary containing render status and generated output paths.
        """
        return mcp_render_plot(spec)

    return server


def main() -> None:
    """
    Function purpose:
        Run the PlotSpec MCP server over the default MCP transport.

    Args:
        None.

    Outputs:
        None. Starts the MCP server process.
    """
    server = build_mcp_server()
    server.run()


if __name__ == "__main__":
    main()
