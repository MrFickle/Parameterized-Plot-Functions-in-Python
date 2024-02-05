# Data Visualization Utilities

This repository contains a set of Python scripts designed to facilitate and streamline the creation of various types of data visualizations. These utilities leverage popular libraries such as Matplotlib, Seaborn, NumPy, and Scikit-learn to produce customizable and informative plots.

## Scripts

### `parameterized_plot_functions.py`

This script serves as the core of the visualization toolkit, offering a suite of functions dedicated to plotting data. It includes capabilities for generating histograms, scatter plots, line plots, and bar plots with extensive customization options for colors, labels, legends, and more.

### Usage Examples

The repository includes example scripts in the /Examples folder, demonstrating how to utilize the `plot_functions_current_upload.py` functions for different types of plots:

- **`barplot_example.py`**: Illustrates how to create a bar plot demonstrating all customizations available.
- **`histogram_example.py`**: Shows how to generate a histogram demonstrating all customizations available.
- **`lineplot_example.py`**: Demonstrates the creation of a line plot demonstrating all customizations available.
- **`scatterplot_example.py`**: Provides an example of a scatter plot demonstrating all customizations available.

## General Features

- **Custom Legends**: Detailed control over legend appearance and placement.
- **Guiding Lines**: Easy to plot vertical and horizontal lines for guidance purposes.
- **Color Mapping**: Flexible color assignments to differentiate between data groups.
- **Error Bars**: Options to include standard deviation or standard error in plots.
- **Statistical Annotations**: Ability to display mean, standard deviation, and R-squared values.
- **Layout Customization**: Extensive customization of plot layout, including axis labels, tick marks, and figure sizing.

## Dependencies

To use these scripts, you'll need Python installed on your system along with the following libraries:

- Matplotlib
- Seaborn
- NumPy
- Scikit-learn
- diptest (for dip test functionality)

You can install these dependencies using `pip`:

```sh
pip install matplotlib seaborn numpy scikit-learn diptest
```

## Getting Started

To get started with these plotting utilities, clone this repository to your local machine:

```sh
git clone https://github.com/MrFickle/Parameterized-Plot-Functions-in-Python.git
```

## Contributing
Contributions to improve or extend the functionality of these plotting utilities are welcome. Please feel free to fork 
the repository, make your changes, and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE)