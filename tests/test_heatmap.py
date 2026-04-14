import numpy as np

from parameterized_plot_functions import plot_heatmap, OutputConfig


def test_plot_heatmap_returns_figure():
    data = np.array([[1, 2], [3, 4]])

    fig = plot_heatmap(
        data=data,
        xlabel="x",
        ylabel="y",
        title="heatmap",
        output_config=OutputConfig(return_fig=True),
    )

    assert fig is not None