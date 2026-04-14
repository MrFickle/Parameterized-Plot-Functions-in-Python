import matplotlib
matplotlib.use('Agg') 

import numpy as np
from parameterized_plot_functions import (
    FigureStyle, 
    OutputConfig, 
    SeriesStyle, 
    AxisStyle,
    LegendStyle,
    ColorbarConfig,
    LineSpec,
    ReferenceLineSpec,
    plot_scatter
)

# Clustered data
x1 = np.random.normal(2, 0.5, 100)
y1 = np.random.normal(2, 0.5, 100)
x2 = np.random.normal(5, 0.8, 150)
y2 = np.random.normal(5, 0.8, 150)

# Colors based on Y values for cluster 2
color_vals = {"Cluster 1": None, "Cluster 2": y2}

plot_scatter(
    x_series={"Cluster 1": x1, "Cluster 2": x2},
    y_series={"Cluster 1": y1, "Cluster 2": y2},
    xlabel="Dimension X1",
    ylabel="Dimension X2",
    title="Cluster Analysis with Regression & Color Density",
    series_styles={
        "Cluster 1": SeriesStyle(color="teal", marker="o", markersize=10, alpha=0.7, label="Stable Group"),
        "Cluster 2": SeriesStyle(color="darkred", marker="s", markersize=8, alpha=0.6, label="Dynamic Group", m_size_factor=1.5),
    },
    axis_style=AxisStyle(
        xlabel_size=22,
        ylabel_size=22,
        tick_length=8,
        pad_labels=12
    ),
    figure_style=FigureStyle(
        figure_size=(16, 12),
        title_size=28,
        use_seaborn=True,
        seaborn_style="white",
        show_figure=False
    ),
    legend_style=LegendStyle(
        loc="upper left",
        fontsize=16
    ),
    output_config=OutputConfig(
        output_dir="showcase_outputs",
        filename="scatter_comprehensive_example"
    ),
    do_linear_reg_fit=True,
    plot_r2_score=True,
    colorbar_config=ColorbarConfig(
        enabled=True,
        colormap="viridis",
        label="Density Gradient (Cluster 2)",
        location="right"
    ),
    color_values=color_vals,
    line_spec=LineSpec(
        vertical=[ReferenceLineSpec(value=3.5, color="black", linestyle="--", linewidth=1.5)]
    )
)