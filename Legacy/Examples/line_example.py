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
    plot_line
)

# Time series data
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)
y1_err = np.random.rand(50) * 0.2
y2_err = np.random.rand(50) * 0.2

plot_line(
    x_series={"Sine": x, "Cosine": x},
    y_series={"Sine": y1, "Cosine": y2},
    xlabel="Time (s)",
    ylabel="Amplitude (Voltage)",
    title="Trigonometric Time-Series with Error Envelopes",
    series_styles={
        "Sine": SeriesStyle(color="blue", linewidth=4, label="Primary Phase", marker="o", markersize=6),
        "Cosine": SeriesStyle(color="orange", linewidth=4, linestyle="--", label="Secondary Phase", marker="x", markersize=8),
    },
    axis_style=AxisStyle(
        xlabel_size=20,
        ylabel_size=20,
        tick_width=2.5,
        spine_width=3,
        use_log_x=False
    ),
    figure_style=FigureStyle(
        figure_size=(14, 8),
        title_size=24,
        show_figure=False
    ),
    legend_style=LegendStyle(
        loc="lower left",
        ncol=2,
        fontsize=14,
        labelcolor="linecolor"
    ),
    output_config=OutputConfig(
        output_dir="showcase_outputs",
        filename="line_comprehensive_example"
    ),
    ylims=(-1.5, 1.5),
    yerr_series={"Sine": y1_err, "Cosine": y2_err},
    line_spec=LineSpec(
        horizontal=[ReferenceLineSpec(value=0, color="gray", alpha=0.5, linestyle="-")]
    ),
    rotate_xticks=False,
    plot_minor_ticks=True,
    extend_y_one_tick=False
)