import parameterized_plot_functions as plotfuns
import numpy as np
from importlib import reload


# Define data distributions for X and Y coordinates with random values, scaled differently for demonstration
data_dictionary_x = {'Dist1': np.random.rand(100), 'Dist2': np.random.rand(200)}
data_dictionary_y = {'Dist1': 4*np.random.rand(100), 'Dist2': 4*np.random.rand(200)}
# Set labels for the X and Y axes
xlabel = 'X'
ylabel = 'Y'
# Title of the figure
figure_title = 'Random Distributions'
# Output directory and filename for saving the plot
data_path_out = r'D:\Downloads\Temp2'
filename = 'test_scatter'
# Set limits for the X and Y axes to define the plot boundaries
xlims = [0, 1.1]
ylims = [0, 4.4]
# Dictionary mapping distribution names to colors for plot elements
color_dictionary = {'Dist1': 'red', 'Dist2': 'blue'}
# Dictionary specifying the opacity level for plot elements of each distribution
alpha = {'Dist1': 1, 'Dist2': 1}
# Legend names mapping for each distribution with a description of data points
legend_dictionary = {'Dist1': '100 points', 'Dist2': '200 points'}
# Font size settings for various plot elements
font_size = 50
label_size = font_size/1.125  # Calculate label size based on the font size
# Marker styles and sizes for each distribution in the scatter plot
m_dict = {'Dist1': 'o', 'Dist2': 'x'}
ms_dict = {'Dist1': 15, 'Dist2': 30}
# Figure size and custom tick values and labels for both axes
figure_size = (25, 25)
xticks_values = np.arange(0, 1.1, 0.2)
yticks_values = np.arange(0, 4.1, 0.4)
xticks_labels = np.round(xticks_values, 1).astype(str)
yticks_labels = np.round(yticks_values, 1).astype(str)
# Location for the legend and flag to plot R^2 score
legend_loc = 'upper left'
plot_r2_score = True
# Saving options and marker size scaling factor
save_svg = True
m_size_factor = 4
# Flags for log scale, colorbar addition, and colorbar customization
use_log_scale = False
add_colorbar = False
colorbar_colormap = 'viridis'
color_values = {'Dist1': data_dictionary_y['Dist1'], 'Dist2': data_dictionary_y['Dist2']}
colorbar_label = 'Colorbar'
colorbar_loc = 'right'
colorbar_ticks = np.arange(0, 1.1, 0.2)
colorbar_tick_labels = np.array(['0', '1/5', '2/5', '3/5', '4/5', '1'])
colorbar_orientation = 'vertical'
# Annotations to be placed on the plot, including their properties
place_text = ['Note1', 'Note2']
place_text_loc = [[0.3, 0.5], [0.4, 0.8]]
make_text_bold = True
place_text_font_size = [50, 50]
rotate_place_text = [True, False]
place_text_color = ['purple', 'orange']
# Padding for layout adjustment and properties for horizontal and vertical lines
pad = 0.25
plot_horizontal_line = True
horizontal_line_y = [0.5]
horizontal_line_color = ['green']
horizontal_line_ls = ['-.']
horizontal_line_lw = [4]
do_linear_reg_fit = True
linear_reg_fit_lw = 5
linear_reg_fit_ls = 'solid'
plot_vertical_line = True
vertical_line_x = [0.5]
vertical_line_color = ['black']
vertical_line_ls = [':']
vertical_line_lw = [4]
# Legend, spine, marker, and tick properties
ncol = 1
spine_width = 5
handletextpad = 0.5
m_linewidth = 1
tick_width = 4
tick_length = 10
# Flag to control the return behavior of the function (return figure object vs. saving)
return_fig_instead_of_save = False
# Invocation of the parameterized scatter plot function with the specified settings
plotfuns.parameterized_scatterplot(data_dictionary_x, data_dictionary_y, xlabel, ylabel, figure_title,
                                   data_path_out, filename, xlims=xlims, ylims=ylims, color_dictionary=color_dictionary,
                                   alpha=alpha, legend_dictionary=legend_dictionary, font_size=font_size, label_size=label_size,
                                   m_dict=m_dict, ms_dict=ms_dict, figure_size=figure_size, xticks_values=xticks_values, yticks_values=yticks_values,
                                   xticks_labels=xticks_labels, yticks_labels=yticks_labels, legend_loc=legend_loc,
                                   plot_r2_score=plot_r2_score, save_svg=save_svg, m_size_factor=m_size_factor, use_log_scale=use_log_scale,
                                   add_colorbar=add_colorbar, colorbar_colormap=colorbar_colormap, color_values=color_values,
                                   colorbar_label=colorbar_label, colorbar_loc=colorbar_loc, colorbar_ticks=colorbar_ticks,
                                   colorbar_tick_labels=colorbar_tick_labels, colorbar_orientation=colorbar_orientation, place_text=place_text,
                                   place_text_loc=place_text_loc, place_text_font_size=place_text_font_size, make_text_bold=make_text_bold,
                                   rotate_place_text=rotate_place_text, place_text_color=place_text_color, pad=pad, plot_horizontal_line=plot_horizontal_line,
                                   horizontal_line_y=horizontal_line_y, horizontal_line_color=horizontal_line_color, horizontal_line_ls=horizontal_line_ls,
                                   horizontal_line_lw=horizontal_line_lw, do_linear_reg_fit=do_linear_reg_fit, linear_reg_fit_lw=linear_reg_fit_lw,
                                   linear_reg_fit_ls=linear_reg_fit_ls, plot_vertical_line=plot_vertical_line, vertical_line_x=vertical_line_x,
                                   vertical_line_color=vertical_line_color, vertical_line_ls=vertical_line_ls, vertical_line_lw=vertical_line_lw,
                                   spine_width=spine_width, ncol=ncol, handletextpad=handletextpad, m_linewidth=m_linewidth, tick_width=tick_width,
                                   tick_length=tick_length, return_fig_instead_of_save=False)