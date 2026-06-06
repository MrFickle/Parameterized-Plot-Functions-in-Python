import runpy
import sys
from pathlib import Path


def test_new_plot_type_examples_run():
    examples_dir = str(Path("Examples").resolve())
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    example_paths = [
        "Examples/grouped_bar_example.py",
        "Examples/stacked_bar_example.py",
        "Examples/dual_axis_bar_example.py",
        "Examples/correlation_heatmap_example.py",
        "Examples/box_example.py",
        "Examples/violin_example.py",
        "Examples/pie_example.py",
        "Examples/area_example.py",
        "Examples/hexbin_example.py",
        "Examples/contour_example.py",
        "Examples/timeline_example.py",
        "Examples/subplots_example.py",
    ]

    for example_path in example_paths:
        runpy.run_path(example_path, run_name="__main__")
