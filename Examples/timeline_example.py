"""Example script for timeline/event plots."""

from matplotlib.figure import Figure

from example_helpers import build_output_config, build_series_styles
from parameterized_plot_functions import AnnotationSpec, AxisStyle, FigureStyle, LegendStyle, LineSpec, ReferenceLineSpec, TextStyle, plot_timeline


def create_timeline_figure(return_fig: bool = False) -> Figure | None:
    """
    Function purpose:
        Create a timeline plot with grouped lanes and labeled events.

    Args:
        return_fig: Whether to return the Matplotlib figure.

    Outputs:
        The figure when `return_fig` is true, otherwise None.
    """
    # Store project events as lane -> event times.
    events = {"Design": [1, 3, 5], "Build": [2, 4, 6], "Validate": [5, 7, 8]}

    # Store event labels in the same order as the event times.
    labels = {
        "Design": ["start", "review", "handoff"],
        "Build": ["api", "plots", "docs"],
        "Validate": ["tests", "examples", "release"],
    }

    # Create a global annotation for the release marker.
    annotations = [
        AnnotationSpec(
            text="Release window",
            xy=(6.1, 2.35),
            xycoords="data",
            style=TextStyle(fontsize=13, color="tab:red", bold=True),
        )
    ]

    # Render event lanes with a vertical reference marker.
    return plot_timeline(
        events=events,
        xlabel="Week",
        title="Timeline Plot Example",
        axis_style=AxisStyle(
            xlabel_size=22,
            ylabel_size=22,
            xtick_size=16,
            ytick_size=16,
            tick_width=2.0,
            tick_length=8.0,
            spine_width=2.0,
            pad_labels=10.0,
            pad_ticks=6.0,
        ),
        figure_style=FigureStyle(
            figure_size=(13, 7),
            title_size=24,
            title_weight="bold",
            tight_layout_pad=0.8,
            show_figure=False,
            use_seaborn=True,
            seaborn_style="ticks",
            seaborn_font_scale=1.2,
            theme="publication",
        ),
        legend_style=LegendStyle(
            enabled=True,
            loc="upper left",
            ncol=3,
            frameon=True,
            fontsize=13,
            handletextpad=0.6,
            handlelength=1.2,
            bbox_to_anchor=(0.02, 0.98),
        ),
        output_config=build_output_config("timeline_example", return_fig=return_fig),
        series_styles=build_series_styles(["Design", "Build", "Validate"]),
        labels=labels,
        lane_labels=["Design phase", "Build phase", "Validation phase"],
        marker_size=110,
        draw_lane_lines=True,
        label_offset=0.12,
        xlims=(0, 9),
        xticks=list(range(0, 10)),
        xtick_labels=[str(i) for i in range(0, 10)],
        line_spec=LineSpec(vertical=[ReferenceLineSpec(6, color="tab:red")]),
        annotations=annotations,
    )


def main() -> None:
    """
    Function purpose:
        Run the timeline plot example.

    Args:
        None.

    Outputs:
        None. Plot files are written to `showcase_outputs`.
    """
    # Generate and save the example figure.
    create_timeline_figure()


if __name__ == "__main__":
    main()
