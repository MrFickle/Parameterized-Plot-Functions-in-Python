import numpy as np

from parameterized_plot_functions import OutputConfig, ShadedRegionSpec, plot_bar, plot_histogram, plot_line, plot_scatter


def test_output_config_can_save_pdf_while_returning_figure(tmp_path):
    fig = plot_line(
        x_series={"a": np.arange(3)},
        y_series={"a": np.arange(3)},
        xlabel="x",
        ylabel="y",
        title="line",
        shaded_regions=[ShadedRegionSpec(0.5, 1.5)],
        output_config=OutputConfig(
            output_dir=str(tmp_path),
            filename="line",
            save_png=False,
            save_svg=False,
            save_pdf=True,
            return_fig=True,
            transparent=True,
            metadata={"kind": "line"},
            save_metadata=True,
        ),
    )

    assert fig is not None
    assert (tmp_path / "line.pdf").exists()
    assert (tmp_path / "line.json").exists()


def test_existing_plot_extensions_return_figures():
    x = np.arange(5)
    y = x**2
    assert plot_scatter(
        x_series={"a": x},
        y_series={"a": y},
        xlabel="x",
        ylabel="y",
        title="scatter",
        polynomial_degree=2,
        point_labels={"a": [str(i) for i in x]},
        output_config=OutputConfig(return_fig=True),
    ) is not None

    assert plot_histogram(
        data_series={"a": y},
        bins=3,
        xlabel="x",
        ylabel="p",
        title="hist",
        cumulative=True,
        fitted_distribution="normal",
        percentile_markers=[50],
        output_config=OutputConfig(return_fig=True),
    ) is not None

    assert plot_bar(
        values={"a": 1, "b": 2},
        xlabel="x",
        ylabel="y",
        title="bar",
        x_positions={"a": 0, "b": 1},
        bar_widths={"a": 0.5, "b": 0.5},
        value_labels=True,
        output_config=OutputConfig(return_fig=True),
    ) is not None
