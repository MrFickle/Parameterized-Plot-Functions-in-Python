"""Command-line interface for PlotSpec rendering."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .render import load_plot_spec_file, render_plot_file
from .specs import get_plot_schema, list_plot_types, validate_plot_spec


def main(argv: list[str] | None = None) -> int:
    """
    Function purpose:
        Run the `ppf` command-line interface.

    Args:
        argv: Optional argument list for tests; uses process arguments when None.

    Outputs:
        Integer process exit code.
    """
    # Build one parser with subcommands for PlotSpec operations.
    parser = argparse.ArgumentParser(description="Render parameterized plot specs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register `list-plots`.
    subparsers.add_parser("list-plots", help="List supported PlotSpec plot types.")

    # Register `schema <plot_type>`.
    schema_parser = subparsers.add_parser("schema", help="Print schema-like information for a plot type.")
    schema_parser.add_argument("plot_type", help="Supported plot type name.")

    # Register `validate <spec_file>`.
    validate_parser = subparsers.add_parser("validate", help="Validate a JSON/YAML PlotSpec file.")
    validate_parser.add_argument("spec_file", help="Path to spec file.")

    # Register `render <spec_file>`.
    render_parser = subparsers.add_parser("render", help="Render a JSON/YAML PlotSpec file.")
    render_parser.add_argument("spec_file", help="Path to spec file.")

    # Parse the provided command arguments.
    args = parser.parse_args(argv)

    # Dispatch to the selected command.
    if args.command == "list-plots":
        _print_json(list_plot_types())
        return 0
    if args.command == "schema":
        _print_json(get_plot_schema(args.plot_type))
        return 0
    if args.command == "validate":
        spec = load_plot_spec_file(args.spec_file)
        _print_json(validate_plot_spec(spec))
        return 0
    if args.command == "render":
        result = render_plot_file(args.spec_file)
        _print_json(result.to_dict())
        return 0

    # The argparse configuration should make this unreachable.
    parser.error(f"Unknown command: {args.command}")
    return 2


def _print_json(value: Any) -> None:
    """
    Function purpose:
        Print a JSON value with stable formatting.

    Args:
        value: JSON-serializable value to print.

    Outputs:
        None. Writes to standard output.
    """
    # Use sorted keys to make CLI output deterministic in tests.
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
