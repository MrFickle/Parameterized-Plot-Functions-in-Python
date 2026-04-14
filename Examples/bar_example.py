import matplotlib
matplotlib.use('Agg') 

import numpy as np
from parameterized_plot_functions import (
    FigureStyle, 
    OutputConfig, 
    SeriesStyle, 
    AxisStyle, 
    LegendStyle,
    AnnotationSpec,
    TextStyle,
    plot_bar
)

# Define data
values = {"Group A": 4.5, "Group B": 7.2, "Group C": 3.1, "Group D": 8.0}
sem = {"Group A": 0.5, "Group B": 0.8, "Group C": 0.3, "Group D": 0.6}
x_pos = {"Group A": 1, "Group B": 2, "Group C": 3, "Group D": 4}
widths = {"Group A": 0.7, "Group B": 0.7, "Group C": 0.7, "Group D": 0.7}

# Configure complex styling
series_styles = {
    "Group A": SeriesStyle(color="#4C72B0", label="Discovery A", alpha=0.8, edgecolor="black", align="center"),
    "Group B": SeriesStyle(color="#55A868", label="Inquiry B", alpha=0.8, edgecolor="black", align="center"),
    "Group C": SeriesStyle(color="#C44E52", label="Control C", alpha=0.8, edgecolor="black", align="center"),
    "Group D": SeriesStyle(color="#8172B2", label="Validation D", alpha=0.8, edgecolor="black", align="center"),
}

# Create rich annotations
annotations = [
    AnnotationSpec(text="Highest Yield", xy=(4, 8.5), xycoords="data", style=TextStyle(bold=True, color="darkgreen")),
    AnnotationSpec(text="Baseline", xy=(0.05, 0.4), style=TextStyle(fontsize=12, rotation=45))
]

plot_bar(
    values=values,
    xlabel="Experimental Conditions",
    ylabel="Performance Metric (Units)",
    title="Comprehensive Bar Plot Showcase",
    x_positions=x_pos,
    bar_widths=widths,
    xticks=[1, 2, 3, 4],
    xtick_labels=["A", "B", "C", "D"],
    sem_values=sem,
    series_styles=series_styles,
    axis_style=AxisStyle(
        xlabel_size=24,
        ylabel_size=24,
        xtick_size=18,
        ytick_size=18,
        tick_width=3,
        tick_length=10,
        spine_width=4,
        remove_first_ytick=True
    ),
    figure_style=FigureStyle(
        figure_size=(16, 12),
        title_size=30,
        show_figure=False,
        use_seaborn=True,
        seaborn_style="whitegrid"
    ),
    legend_style=LegendStyle(
        enabled=True,
        loc="upper left",
        ncol=2,
        fontsize=16,
        frameon=True,
        bbox_to_anchor=(0.02, 0.98)
    ),
    output_config=OutputConfig(
        output_dir="showcase_outputs",
        filename="bar_comprehensive_example",
        save_svg=True,
        dpi=300
    ),
    rotate_xticks=True,
    plot_minor_ticks=True,
    extend_y_one_tick=True,
    annotations=annotations
)