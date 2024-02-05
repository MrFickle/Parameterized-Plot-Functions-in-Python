import parameterized_plot_functions as plotfuns
import numpy as np
from importlib import reload


# Define data distributions for X and Y values of the lines
data_dictionary_x = {'Dist1': np.arange(0, 1.1, 0.1), 'Dist2': np.arange(0, 1.1, 0.1)}
data_dictionary_y = {'Dist1': np.arange(0, 2.1, 0.2), 'Dist2': np.arange(0, 4.1, 0.4)}
# Define axis labels and the figure title
xlabel = 'X'
ylabel = 'Y'
figure_title = 'Random Distributions'
# Set the output path and filename for the plot
data_path_out = r'D:\Downloads\Temp2'
filename = 'test_line'
# Provide a dictionary for line colors and legend names
color_dictionary = {'Dist1': 'red', 'Dist2': 'blue'}
legend_dictionary = {'Dist1': 'Line 1', 'Dist2': 'Line 2'}
# Specify line widths and styles for each distribution
lw_dict = {'Dist1': 5, 'Dist2': 5}
ls_dict = {'Dist1': 'solid', 'Dist2': 'dashed'}
# If needed, define a dictionary for custom font sizes for plot elements (here set to None as an example)
fs_dict = None
# Set marker styles and sizes for each distribution
m_dict = {'Dist1': 'o', 'Dist2': 'x'}
ms_dict = {'Dist1': 15, 'Dist2': 30}
# Define the figure size
figure_size = (25, 25)
# Set the values and labels for x and y ticks
xticks_values = np.arange(0, 1.1, 0.2)
yticks_values = np.arange(0, 4.1, 0.4)
xticks_labels = np.array(['0', '1/5', '2/5', '3/5', '4/5', '1'])
yticks_labels = np.round(yticks_values, 1).astype(str)
# Choose the location for the plot legend
legend_loc = 'upper left'
# Provide dictionaries for Y errors for each line if available
data_dictionary_yerr = {'Dist1': np.random.rand(11), 'Dist2': np.random.rand(11)}
# Set the limits for the x and y axes
xlims = [0, 1.1]
ylims = [0, 4.4]
# Choose whether to use standard error of the mean or error bars
use_sem_instead_of_errorbar = False
# Set the alpha transparency for each line
alpha_dict = {'Dist1': 1, 'Dist2': 1}
# Define the histogram statistic to use (e.g., 'probability', 'frequency', etc.)
hist_stat = 'probability'
# Choose whether to save the plot as an SVG
save_svg = True
# Define if lines should be shown in the legend
show_lines = True
# Add text annotations to the plot
place_text = ['Note1', 'Note2']
place_text_loc = [[0.3, 0.5], [0.4, 0.8]]
make_text_bold = True
place_text_font_size = [50, 50]
rotate_place_text = [True, False]
place_text_color = ['purple', 'orange']
# Define if logarithmic scale is to be used for axes
use_log_scale = False
# Define if masking should be used to only plot points with finite Y values
use_mask = True
# Set the width and length of axis ticks
tick_width = 4
tick_length = 10
# Define error bar cap size, line width, and cap thickness
yerr_capsize = 20
yerr_elinewidth = 5
capthick = 5
# Set the width of the axis spines
spine_width = 5
# Define padding used with tight layout
pad = 0.25
# Choose whether to use seaborn styling for the plot
use_sns = True
# Define whether to plot vertical lines on the plot
plot_vertical_line = True
vertical_line_x = [0.5]
vertical_line_color = ['black']
vertical_line_ls = [':']
vertical_line_lw = [4]
# Define whether to plot horizontal lines on the plot
plot_horizontal_line = True
horizontal_line_y = [0.5]
horizontal_line_color = ['green']
horizontal_line_ls = ['-.']
horizontal_line_lw = [4]
# Define whether to use a colorbar
add_colorbar = False
colorbar_colormap = 'viridis'
colorbar_label = 'Colorbar'
colorbar_loc = 'right'
colorbar_ticks = np.arange(0, 1.1, 0.2)
colorbar_tick_labels = np.array(['0', '1/5', '2/5', '3/5', '4/5', '1'])
colorbar_orientation = 'vertical'
# Define the line length in the legend
legend_line_length = 1
# Define whether to color each error bar with the color of its corresponding line
use_line_color_for_error = False
# Define whether to pad ticks and labels
pad_ticks = True
pad_labels = True
# Define number of columns in the legend
ncol = 2
# Define pad between line and text in legend
handletextpad = 0.5
# Define linewidth factor of legend line
legend_lw_multiplier = 1
# Font size
font_size = 50
# Define whether to color each legend text with the color of its corresponding line
use_line_color_for_legends = True
# Define whether to save the figure or return it as an object
return_fig_instead_of_save = False
# Invocation of the parametrized plotting function with all the specified parameters
plotfuns.plot_parameterized_lineplot(data_dictionary_x, data_dictionary_y, xlabel, ylabel, figure_title,
                                     data_path_out, filename, color_dictionary=color_dictionary,
                                     legend_dictionary=legend_dictionary, lw_dict=lw_dict, ls_dict=ls_dict, 
                                     fs_dict=fs_dict, m_dict=m_dict, ms_dict=ms_dict, figure_size=figure_size, 
                                     xticks_values=xticks_values, yticks_values=yticks_values, xticks_labels=xticks_labels,
                                     yticks_labels=yticks_labels, legend_loc=legend_loc, data_dictionary_yerr=data_dictionary_yerr, 
                                     xlims=xlims, ylims=ylims, use_sem_instead_of_errorbar=use_sem_instead_of_errorbar, 
                                     alpha_dict=alpha_dict, save_svg=save_svg, show_lines=show_lines, 
                                     place_text=place_text, place_text_loc=place_text_loc, make_text_bold=make_text_bold,
                                     place_text_font_size=place_text_font_size, rotate_place_text=rotate_place_text, 
                                     place_text_color=place_text_color, use_log_scale=use_log_scale, use_mask=use_mask, 
                                     tick_width=tick_width, tick_length=tick_length,
                                     yerr_capsize=yerr_capsize, yerr_elinewidth=yerr_elinewidth, capthick=capthick, 
                                     spine_width=spine_width, pad=pad,
                                     use_sns=use_sns, plot_vertical_line=plot_vertical_line, 
                                     vertical_line_x=vertical_line_x, vertical_line_color=vertical_line_color,
                                     vertical_line_ls=vertical_line_ls, vertical_line_lw=vertical_line_lw,
                                     plot_horizontal_line=plot_horizontal_line,
                                     horizontal_line_y=horizontal_line_y, horizontal_line_color=horizontal_line_color,
                                     horizontal_line_ls=horizontal_line_ls,
                                     horizontal_line_lw=horizontal_line_lw, add_colorbar=add_colorbar,
                                     colorbar_colormap=colorbar_colormap, colorbar_label=colorbar_label,
                                     colorbar_loc=colorbar_loc, colorbar_ticks=colorbar_ticks, colorbar_tick_labels=colorbar_tick_labels,
                                     colorbar_orientation=colorbar_orientation, legend_line_length=legend_line_length,
                                     use_line_color_for_error=use_line_color_for_error,
                                     pad_ticks=pad_ticks, pad_labels=pad_labels, ncol=ncol, handletextpad=handletextpad,
                                     bbox_to_anchor=None,
                                     font_size=font_size, legend_lw_multiplier=legend_lw_multiplier, use_line_color_for_legends=use_line_color_for_legends,
                                     return_fig_instead_of_save=False)