import plot_functions_reformed as plotfuns
import numpy as np
from importlib import reload


# Define data distributions with the height of each bar
data_dictionary = {'Bar1': 4, 'Bar2': 5, 'Bar3': 2, 'Bar4': 8}
# Set labels and title for the plot
xlabel = 'Value'
ylabel = 'Height'
figure_title = 'Random Bars'
# Define output path and filename for saving the plot
data_path_out = r'D:\Downloads\Temp2'
filename = 'test_hist'
# Specify colors for each bar and legend entries
color_dict_bars = {'Bar1': 'blue', 'Bar2': 'red', 'Bar3': 'green', 'Bar4': 'pink'}
color_dict_legends = {'Legend1': 'purple', 'Legend2': '#EEEEEE'}
# Define the position (bins) and width for each bar
bins = {'Bar1': 1, 'Bar2': 2, 'Bar3': 3, 'Bar4': 4}
bar_width = {'Bar1': 1, 'Bar2': 1, 'Bar3': 1, 'Bar4': 1}
# Set the ticks for the x-axis and their labels
xticks = np.arange(0, 5, 1)
xticks_labels = xticks.astype(str)
# Define the legends for the plot
legends = ['Legend1', 'Legend2']
# Set the ticks for the y-axis and their labels
yticks = np.arange(0, 10, 2)
yticks_labels = yticks.astype(str)
# Alignment of bars on the x-axis (center or edge)
align = 'center'
# Opacity of the bars
alpha = 1
# Edge color of the bars, if any
edgecolor = None
# Whether to rotate the x-axis tick labels
rotate_xticks = False
# Standard error of the mean for the bars, if provided
sem_data_dictionary = {'Bar1': 1, 'Bar2': 2, 'Bar3': 0.5, 'Bar4': 0.5}
# Whether to remove the first tick on the y-axis
remove_first_y = False
# Limits for the x-axis and y-axis
xlims = [0, 5]
ylims = [0, 10]
# Size of the figure as (width, height)
figure_size = (25, 25)
# Location of the legend within the plot
legend_loc = 'upper left'
# Number of columns in the legend
ncol = 1
# Font size for text in the plot
font_size = 50
# Font size for tick labels on the axes
label_size = font_size/1.125
# Style parameters for error bars
capsize = 5
elinewidth = 2
capthick = 2
# Whether to plot minor tick marks on the axes
plot_minor_ticks = True
# Width of the plot spines
spine_width = 7
# Font sizes for tick labels, if different from label_size
yticks_font_size = label_size
xticks_font_size = label_size
# Whether to extend the y-axis to include one extra tick beyond the data range
set_ylims_one_extra_tick = True
# Padding used in layout adjustment
pad = 0.25
# Whether to save the plot as SVG format in addition to PNG
save_svg = True
# Text to annotate on the plot and their respective locations
place_text = ['Note1', 'Note2']
place_text_loc = [[0.5, 0.5], [0.8, 0.8]]
# Colors for the annotated text
place_text_color = ['purple', 'orange']
# Font size for annotated text
place_text_font_size = [font_size, font_size]
# Whether to rotate annotated text and make it bold
rotate_place_text = [True, False]
make_text_bold = True
# Whether to apply seaborn style to the plot
use_sns = True
# Whether to remove tick lines from the x-axis
disable_xtick_edges = True
# Location for the legend's bounding box anchor, if specified
bbox_to_anchor = None
# Whether to make the legend patches (color squares) visible
make_legend_patch_visible = True
# Whether to match the legend text color to the line color
legend_linecolor = True
# Padding between the legend text and its corresponding patch (color square)
handletextpad = 0.5
# Width and length of the ticks on both axes
tick_width = 1
tick_length = 5
# Define whether to save the figure or return it as an object
return_fig_instead_of_save = False
# Invocation of the parametrized plotting function with all the specified parameters
plotfuns.plot_parameterized_barplot(data_dictionary, xlabel, ylabel, figure_title, data_path_out, filename,
                               color_dict_bars, color_dict_legends, bins, bar_width,
                               xticks, xticks_labels, legends, yticks=yticks, yticks_labels=yticks_labels,
                               align=align, alpha=1.0, edgecolor=edgecolor, rotate_xticks=rotate_xticks,
                               sem_data_dictionary=sem_data_dictionary,
                               remove_first_y=remove_first_y, xlims=xlims, ylims=ylims, figure_size=figure_size,
                               legend_loc=legend_loc, ncol=ncol, font_size=font_size, label_size=label_size,
                               capsize=capsize, elinewidth=elinewidth, capthick=capthick, plot_minor_ticks=plot_minor_ticks,
                               spine_width=spine_width, yticks_font_size=yticks_font_size, xticks_font_size=xticks_font_size,
                               set_ylims_one_extra_tick=set_ylims_one_extra_tick, pad=pad, save_svg=save_svg, place_text=place_text,
                               place_text_loc=place_text_loc, place_text_color=place_text_color, place_text_font_size=place_text_font_size,
                               rotate_place_text=rotate_place_text, make_text_bold=make_text_bold, use_sns=use_sns,
                               disable_xtick_edges=disable_xtick_edges, bbox_to_anchor=bbox_to_anchor, make_legend_patch_visible=make_legend_patch_visible,
                               handletextpad=handletextpad, tick_width=tick_width, tick_length=tick_length,
                               return_fig_instead_of_save=return_fig_instead_of_save)