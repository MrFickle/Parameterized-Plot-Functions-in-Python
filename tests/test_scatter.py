import numpy as np

from parameterized_plot_functions import plot_scatter, OutputConfig


def test_plot_scatter_returns_figure():
    x = np.arange(10)
    y = x + 1

    fig = plot_scatter(
        x_series={"a": x},
        y_series={"a": y},
        xlabel="x",
        ylabel="y",
        title="scatter",
        output_config=OutputConfig(return_fig=True),
    )

    assert fig is not None