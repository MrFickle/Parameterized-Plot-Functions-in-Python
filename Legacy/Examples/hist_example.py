import matplotlib
matplotlib.use('Agg') 

import numpy as np
from parameterized_plot_functions import (
    FigureStyle, 
    OutputConfig, 
    SeriesStyle, 
    AxisStyle,
    LegendStyle,
    LineSpec,
    ReferenceLineSpec,
    AnnotationSpec,
    TextStyle,
    plot_histogram
)

# Generate overlapping distributions
dist1 = np.random.normal(0, 1, 1000)
dist2 = np.random.normal(2, 1.5, 1200)

plot_histogram(
    data_series={
        "Control": dist1,
        "Treatment": dist2
    },
    bins=40,
    xlabel="Observed Value",
    ylabel="Density / Probability",
    title="Distribution Analysis with Statistical Overlays",
    series_styles={
        "Control": SeriesStyle(color="steelblue", alpha=0.6, label="Control Group (N=1k)"),
        "Treatment": SeriesStyle(color="indianred", alpha=0.6, label="Treatment Group (N=1.2k)"),
    },
    axis_style=AxisStyle(
        xlabel_size=22,
        ylabel_size=22,
        remove_first_ytick=True,
        spine_width=3
    ),
    figure_style=FigureStyle(
        figure_size=(16, 10),
        use_seaborn=True,
        seaborn_style="ticks",
        show_figure=False
    ),
    legend_style=LegendStyle(
        loc="upper right",
        ncol=1,
        fontsize=16
    ),
    output_config=OutputConfig(
        output_dir="showcase_outputs",
        filename="hist_comprehensive_example"
    ),
    plot_mean=True,
    plot_std=True,
    plot_kde=True,
    hist_stat="density",
    xlims=(-5, 8),
    line_spec=LineSpec(
        vertical=[
            ReferenceLineSpec(value=0, color="blue", linestyle=":", linewidth=2),
            ReferenceLineSpec(value=2, color="red", linestyle="--", linewidth=2)
        ]
    ),
    annotations=[
        AnnotationSpec(text="Significant Overlap", xy=(0.3, 0.5), style=TextStyle(bold=True, color="purple"))
    ],
    extend_y_one_tick=True
)