from parameterized_plot_functions import plot_bar, OutputConfig


def test_plot_bar_returns_figure():
    fig = plot_bar(
        values={"a": 1.0, "b": 2.0},
        xlabel="x",
        ylabel="y",
        title="bar",
        x_positions={"a": 0, "b": 1},
        bar_widths={"a": 0.8, "b": 0.8},
        output_config=OutputConfig(return_fig=True),
    )

    assert fig is not None