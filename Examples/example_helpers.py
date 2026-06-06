"""Shared helpers for runnable plotting examples."""

from pathlib import Path
import sys

from matplotlib.figure import Figure

# Resolve local package imports when examples are run from a fresh checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parameterized_plot_functions import OutputConfig, SeriesStyle


def build_output_config(filename: str, return_fig: bool = False) -> OutputConfig:
    """
    Function purpose:
        Build a standard output configuration for example scripts.

    Args:
        filename: Base filename used for saved outputs.
        return_fig: Whether the plotting function should return its figure.

    Outputs:
        Output configuration that writes PNG and SVG files under `showcase_outputs`.
    """
    # Keep example outputs in the same folder used by the original examples.
    output_dir = Path("showcase_outputs")

    # Return the shared configuration used across individual example scripts.
    return OutputConfig(
        output_dir=str(output_dir),
        filename=filename,
        save_png=True,
        save_svg=True,
        save_pdf=True,
        dpi=300,
        return_fig=return_fig,
        transparent=False,
        metadata={"example": filename},
        save_metadata=True,
    )


def build_series_styles(keys: list[str]) -> dict[str, SeriesStyle]:
    """
    Function purpose:
        Build deterministic series styles for examples.

    Args:
        keys: Series keys that need visual styles.

    Outputs:
        Mapping from series key to reusable visual style.
    """
    # Use Matplotlib's built-in tab colors for stable, readable examples.
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    # Assign one style per key while cycling through the palette.
    return {
        key: SeriesStyle(color=colors[index % len(colors)], label=key, linewidth=2.5, marker="o", alpha=0.8)
        for index, key in enumerate(keys)
    }


def require_figure(fig: Figure | None, name: str) -> Figure:
    """
    Function purpose:
        Validate that a plotting function returned a figure.

    Args:
        fig: Figure returned by a plotting function.
        name: Human-readable figure name used in error messages.

    Outputs:
        The validated Matplotlib figure.
    """
    # Fail early when an example expects a returned figure but did not receive one.
    if fig is None:
        raise ValueError(f"{name} did not return a figure.")

    # Return the figure so layout examples can reuse it.
    return fig
