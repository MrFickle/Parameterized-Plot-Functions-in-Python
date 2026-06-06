import numpy as np

from parameterized_plot_functions import (
    AnnotationSpec,
    LegendStyle,
    LineSpec,
    OutputConfig,
    ReferenceLineSpec,
    ShadedRegionSpec,
    create_subplots_figure,
    plot_area,
    plot_box,
    plot_contour,
    plot_correlation_heatmap,
    plot_grouped_bar,
    plot_heatmap,
    plot_hexbin,
    plot_pie,
    plot_stacked_bar,
    plot_timeline,
    plot_violin,
)


def return_fig_config():
    return OutputConfig(return_fig=True)


def test_distribution_plots_return_figures():
    data = {"a": np.arange(5), "b": np.arange(5) + 1}
    assert plot_box(
        data,
        "group",
        "value",
        "box",
        positions=[1, 3],
        tick_labels=["A", "B"],
        show_means=True,
        grid_axis="y",
        annotations=[AnnotationSpec("box", (0.1, 0.9))],
        legend_style=LegendStyle(enabled=False),
        output_config=return_fig_config(),
    ) is not None
    assert plot_violin(
        data,
        "group",
        "value",
        "violin",
        positions=[1, 3],
        tick_labels=["A", "B"],
        quantiles=[[0.5], [0.5]],
        grid_axis="y",
        annotations=[AnnotationSpec("violin", (0.1, 0.9))],
        legend_style=LegendStyle(enabled=False),
        output_config=return_fig_config(),
    ) is not None


def test_bar_extensions_return_figures():
    values = {"g1": {"a": 1.0, "b": 2.0}, "g2": {"a": 1.5, "b": 2.5}}
    assert plot_grouped_bar(
        values,
        "group",
        "value",
        "grouped",
        annotations=[AnnotationSpec("grouped", (0.1, 0.9))],
        legend_style=LegendStyle(enabled=False),
        output_config=return_fig_config(),
    ) is not None
    assert plot_stacked_bar(
        values,
        "group",
        "value",
        "stacked",
        annotations=[AnnotationSpec("stacked", (0.1, 0.9))],
        legend_style=LegendStyle(enabled=False),
        output_config=return_fig_config(),
    ) is not None


def test_pie_area_density_and_timeline_return_figures():
    x = np.linspace(0, 1, 20)
    y = np.sin(x)
    grid_x, grid_y = np.meshgrid(x, x)
    z = grid_x + grid_y

    assert plot_pie(
        {"a": 1, "b": 2},
        "pie",
        donut_width=0.4,
        explode=[0, 0.1],
        show_legend=True,
        legend_style=LegendStyle(enabled=True, loc="upper right"),
        annotations=[AnnotationSpec("pie", (0.1, 0.9))],
        output_config=return_fig_config(),
    ) is not None
    assert plot_area(
        x,
        {"a": y, "b": y + 1},
        "x",
        "y",
        "area",
        stacked=True,
        shaded_regions=[ShadedRegionSpec(0.2, 0.4)],
        line_spec=LineSpec(vertical=[ReferenceLineSpec(0.5)]),
        annotations=[AnnotationSpec("area", (0.1, 0.9))],
        output_config=return_fig_config(),
    ) is not None
    assert plot_hexbin(
        x,
        y,
        "x",
        "y",
        "hexbin",
        colorbar_label="count",
        bins="log",
        legend_style=LegendStyle(fontsize=8),
        annotations=[AnnotationSpec("hexbin", (0.1, 0.9))],
        output_config=return_fig_config(),
    ) is not None
    assert plot_contour(
        grid_x,
        grid_y,
        z,
        "x",
        "y",
        "contour",
        colorbar_label="z",
        vmin=0,
        vmax=2,
        legend_style=LegendStyle(fontsize=8),
        annotations=[AnnotationSpec("contour", (0.1, 0.9))],
        output_config=return_fig_config(),
    ) is not None
    assert plot_timeline(
        {"phase": [1, 2, 3]},
        "time",
        "timeline",
        labels={"phase": ["a", "b", "c"]},
        line_spec=LineSpec(vertical=[ReferenceLineSpec(2)]),
        legend_style=LegendStyle(enabled=False),
        annotations=[AnnotationSpec("timeline", (0.1, 0.9))],
        output_config=return_fig_config(),
    ) is not None


def test_correlation_heatmap_returns_figure():
    data = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 7], [4, 5, 9]])
    assert plot_heatmap(
        data,
        "x",
        "y",
        "heatmap",
        colorbar=True,
        legend_style=LegendStyle(fontsize=8),
        annotations=[AnnotationSpec("heatmap", (0.1, 0.9))],
        output_config=return_fig_config(),
    ) is not None
    assert plot_correlation_heatmap(
        data,
        legend_style=LegendStyle(fontsize=8),
        annotations=[AnnotationSpec("corr", (0.1, 0.9))],
        output_config=return_fig_config(),
    ) is not None


def test_subplots_helper_returns_figure():
    def draw_line(ax):
        ax.plot([0, 1], [0, 1])

    assert create_subplots_figure([draw_line], 1, 1, title="subplots", output_config=return_fig_config()) is not None
