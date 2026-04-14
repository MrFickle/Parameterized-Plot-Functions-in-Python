import numpy as np

from parameterized_plot_functions import plot_line, OutputConfig


def test_plot_line_returns_figure():
    x = np.arange(10)
    y = x ** 2

    fig = plot_line(
        x_series={"a": x},
        y_series={"a": y},
        xlabel="x",
        ylabel="y",
        title="line",
        output_config=OutputConfig(return_fig=True),
    )

    assert fig is not None