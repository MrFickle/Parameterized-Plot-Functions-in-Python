# Parameterized Plot Functions in Python

A modular, config-driven plotting library built on Matplotlib and Seaborn.

The current implementation lives in `src/parameterized_plot_functions`. Current runnable
examples live in `Examples/`. The `Legacy/` directory contains the old monolithic module
and old example scripts for reference.

## What It Provides

- Line, scatter, histogram, bar, dual-axis bar, and heatmap plotting helpers
- Shared dataclass configs for axis, figure, legend, series, annotations, reference lines, colorbars, and output
- Consistent Matplotlib styling across plot types
- Optional figure saving to PNG and SVG
- Optional return of Matplotlib `Figure` objects for tests, notebooks, or downstream composition
- Grid composition helpers for combining generated figures

## Project Layout

```text
.
+-- src/parameterized_plot_functions/   # Current refactored package
|   +-- plots/                          # Plot implementations
|   +-- configs.py                      # Public dataclass configs
|   +-- axes.py                         # Axis styling helpers
|   +-- annotations.py                  # Annotation helpers
|   +-- legends.py                      # Legend helpers
|   +-- layouts.py                      # Figure/grid helpers
|   +-- saving.py                       # Save/finalize behavior
+-- Examples/                           # Current runnable example scripts
+-- Legacy/                             # Old monolithic module and old examples
+-- tests/                              # Current behavior tests
```

## Installation

Requires Python 3.11+.

```bash
pip install -e .
```

Project dependencies are declared in `pyproject.toml`:

- `matplotlib`
- `numpy`
- `seaborn`
- `scikit-learn`

`requirements.txt` also includes `diptest`, but dip-test logic is currently commented out in
the refactored histogram implementation.

## Quick Start

```python
import numpy as np

from parameterized_plot_functions import (
    OutputConfig,
    SeriesStyle,
    plot_line,
)

x = np.arange(10)
y = x**2

fig = plot_line(
    x_series={"quadratic": x},
    y_series={"quadratic": y},
    xlabel="x",
    ylabel="y",
    title="Quadratic",
    series_styles={
        "quadratic": SeriesStyle(
            color="tab:blue",
            label="x^2",
            linewidth=2.5,
            marker="o",
        )
    },
    output_config=OutputConfig(return_fig=True),
)
```

By default, plot functions close the figure and return `None`. Use
`OutputConfig(return_fig=True)` when you need the `Figure` object.

## Saving Figures

```python
from parameterized_plot_functions import OutputConfig, plot_histogram

plot_histogram(
    data_series={"sample": data},
    bins=30,
    xlabel="Value",
    ylabel="Probability",
    title="Sample Distribution",
    output_config=OutputConfig(
        output_dir="figures",
        filename="sample_distribution",
        save_png=True,
        save_svg=True,
        dpi=300,
    ),
)
```

When both `output_dir` and `filename` are provided:

- PNG is written to `output_dir/filename.png`
- SVG is written to `output_dir/SVG/filename.svg`

## Public API

The package exports these plotting functions:

| Function | Purpose |
| --- | --- |
| `plot_line` | Multiple named x/y series, optional error bands or error bars, reference lines, annotations, minor ticks |
| `plot_scatter` | Multiple scatter series, optional linear regression fit, R2 labels, colorbar support |
| `plot_histogram` | Multiple distributions, configurable bins/stat, optional KDE, mean/std labels, reference lines |
| `plot_bar` | Single-axis bar plots with custom positions, widths, SEM error bars, annotations |
| `plot_dual_axis_bar` | Bar plots split across left/right y-axes |
| `plot_heatmap` | Seaborn heatmaps with optional annotations, colorbar, bounds, colormap, tick rotation |
| `create_empty_figure` | Create a blank Matplotlib figure |
| `draw_figures_grid` | Compose multiple figures into a grid-like combined figure |

The main config dataclasses are:

| Config | Purpose |
| --- | --- |
| `AxisStyle` | Labels, ticks, spines, log scales, tick visibility |
| `FigureStyle` | Figure size, title style, layout padding, Seaborn style, display behavior |
| `OutputConfig` | Output directory, filename, PNG/SVG flags, DPI, figure return behavior |
| `LegendStyle` | Legend visibility, position, columns, frame, font, handle spacing |
| `SeriesStyle` | Color, label, line/marker/bar styling, alpha, edge color |
| `TextStyle` | Annotation font size, color, weight, rotation |
| `AnnotationSpec` | Text annotation position and style |
| `ReferenceLineSpec` | Vertical/horizontal reference line style |
| `LineSpec` | Groups vertical and horizontal reference lines |
| `ColorbarConfig` | Scatter colorbar settings |

## Examples

Current example scripts mirror the old examples but use the refactored package API:

```bash
uv run python Examples/line_example.py
uv run python Examples/scatter_example.py
uv run python Examples/hist_example.py
uv run python Examples/bar_example.py
uv run python Examples/heatmap_example.py
uv run python Examples/layout_grid_example.py
```

Each script writes PNG and SVG outputs under `showcase_outputs/`.
`layout_grid_example.py` reuses the individual example figure creation functions and
passes their returned figures into `draw_figures_grid`.

### Scatter With Regression

```python
import numpy as np

from parameterized_plot_functions import OutputConfig, plot_scatter

x = np.arange(20)
y = 2 * x + np.random.normal(size=20)

fig = plot_scatter(
    x_series={"fit": x},
    y_series={"fit": y},
    xlabel="x",
    ylabel="y",
    title="Scatter Fit",
    do_linear_reg_fit=True,
    plot_r2_score=True,
    output_config=OutputConfig(return_fig=True),
)
```

### Bar Plot

```python
from parameterized_plot_functions import OutputConfig, SeriesStyle, plot_bar

fig = plot_bar(
    values={"A": 1.0, "B": 2.0},
    xlabel="Group",
    ylabel="Value",
    title="Grouped Values",
    x_positions={"A": 0, "B": 1},
    bar_widths={"A": 0.8, "B": 0.8},
    series_styles={
        "A": SeriesStyle(color="tab:blue", label="A"),
        "B": SeriesStyle(color="tab:orange", label="B"),
    },
    xticks=[0, 1],
    xtick_labels=["A", "B"],
    output_config=OutputConfig(return_fig=True),
)
```

### Heatmap

```python
import numpy as np

from parameterized_plot_functions import OutputConfig, plot_heatmap

fig = plot_heatmap(
    data=np.array([[1, 2], [3, 4]]),
    xlabel="Column",
    ylabel="Row",
    title="Matrix",
    annotate=True,
    colorbar=True,
    output_config=OutputConfig(return_fig=True),
)
```

## Testing

```bash
pytest
```

The current tests verify that each refactored plot function can create and return a
Matplotlib figure when called with `OutputConfig(return_fig=True)`.

## Development Notes

- Prefer imports from `parameterized_plot_functions`, resolved from `src/`.
- Keep new behavior in `src/parameterized_plot_functions`.
- Keep current examples in `Examples/`.
- Add or update tests under `tests/` when changing plot behavior.
- Treat `Legacy/` as historical reference only.
