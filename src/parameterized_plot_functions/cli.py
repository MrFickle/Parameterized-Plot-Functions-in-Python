"""Command-line interface for PlotSpec rendering."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .render import load_plot_spec_file, render_plot_file
from .specs import get_all_plot_schemas, get_plot_schema, list_plot_types, validate_plot_spec


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

    # Register `schema [plot_type]`.
    schema_parser = subparsers.add_parser("schema", help="Print JSON Schema information for one plot type or all plot types.")
    schema_parser.add_argument("plot_type", nargs="?", help="Supported plot type name. Omit with --all.")
    schema_parser.add_argument("--all", action="store_true", help="Print schemas for every supported plot type.")

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
        if args.all:
            _print_json(get_all_plot_schemas())
            return 0
        if args.plot_type is None:
            parser.error("schema requires a plot_type unless --all is provided.")
            return 2
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
