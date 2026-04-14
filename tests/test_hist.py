import numpy as np

from parameterized_plot_functions import plot_histogram, OutputConfig


def test_plot_histogram_returns_figure():
    data = np.random.randn(200)

    fig = plot_histogram(
        data_series={"dist": data},
        bins=20,
        xlabel="x",
        ylabel="freq",
        title="hist",
        output_config=OutputConfig(return_fig=True),
    )

    assert fig is not None