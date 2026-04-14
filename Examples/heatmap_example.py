import matplotlib
matplotlib.use('Agg') 

import numpy as np
from parameterized_plot_functions import (
    FigureStyle, 
    OutputConfig, 
    AxisStyle,
    plot_heatmap
)

# Generate synthetic correlation-like data
data = np.random.rand(10, 10)
data = (data + data.T) / 2  # Make it symmetric
np.fill_diagonal(data, 1.0)

plot_heatmap(
    data=data,
    xlabel="Features Space X",
    ylabel="Features Space Y",
    title="High-Resolution Thermal Correlation Map",
    axis_style=AxisStyle(
        xlabel_size=20,
        ylabel_size=20,
        xtick_size=12,
        ytick_size=12
    ),
    figure_style=FigureStyle(
        figure_size=(14, 12),
        title_size=26,
        show_figure=False,
        tight_layout_pad=1.0
    ),
    output_config=OutputConfig(
        output_dir="showcase_outputs",
        filename="heatmap_comprehensive_example"
    ),
    annotate=True,
    colorbar=True,
    vmin=0,
    vmax=1,
    cmap="magma",
    rotate_ticks=True
)