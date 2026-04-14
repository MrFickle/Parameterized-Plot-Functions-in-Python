import os

from matplotlib.figure import Figure

from .configs import FigureStyle, OutputConfig


def save_figure(fig: Figure, output_config: OutputConfig) -> None:
    if not output_config.output_dir or not output_config.filename:
        return

    os.makedirs(output_config.output_dir, exist_ok=True)

    if output_config.save_png:
        png_path = os.path.join(output_config.output_dir, f"{output_config.filename}.png")
        fig.savefig(png_path, dpi=output_config.dpi, bbox_inches="tight")

    if output_config.save_svg:
        svg_dir = os.path.join(output_config.output_dir, "SVG")
        os.makedirs(svg_dir, exist_ok=True)
        svg_path = os.path.join(svg_dir, f"{output_config.filename}.svg")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")


def finalize_figure(
    fig: Figure,
    ax,
    title: str,
    figure_style: FigureStyle,
    output_config: OutputConfig,
):
    ax.set_title(
        title,
        fontsize=figure_style.title_size,
        fontweight=figure_style.title_weight,
    )
    fig.tight_layout(pad=figure_style.tight_layout_pad)

    if not output_config.return_fig:
        save_figure(fig, output_config)

    if figure_style.show_figure:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        import matplotlib.pyplot as plt
        plt.close(fig)

    return fig if output_config.return_fig else None