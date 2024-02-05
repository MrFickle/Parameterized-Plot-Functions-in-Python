import parameterized_plot_functions as plotfuns
import numpy as np
from importlib import reload


# Define data distributions with two different distributions, each generated using numpy's random.rand function with 100 and 200 values respectively
data_dictionary = {'Dist1': np.random.rand(100), 'Dist2': np.random.rand(200)}
# Define histogram bins with a range from 0 to 1, with a step of 0.05
bins = np.arange(0, 1.05, 0.05)
# Labels for the x and y axes
xlabel = 'Value'
ylabel = 'Probability'
# Title of the figure
figure_title = 'Random Distributions'
# Set the output path and filename for the plot
data_path_out = r'D:\Downloads\Temp2'
filename = 'test_hist'
# Dictionary mapping distribution names to their respective colors for plotting
color_dict = {'Dist1': 'red', 'Dist2': 'blue'}
# Dictionary specifying the opacity for each distribution plot
alpha = {'Dist1': 0.8, 'Dist2': 0.8}
# Specifies that the histogram statistic to be plotted is 'probability'
hist_stat = 'probability'
# Limits for the x and y axes, setting the plot boundaries
xlims = [0, 1]
ylims = [0, 1]
# Ticks to be shown on the x and y axes
xticks = np.arange(0, 1.1, 0.2)
yticks = np.arange(0, 0.21, 0.05)
# A list containing x-axis values where vertical threshold lines should be plotted
plot_vertical_th_line = [0.5]
# Legend information, mapping distribution names to descriptions of the number of values
legend_dictionary = {'Dist1': '100 values', 'Dist2': '200 values'}
# Flags to control the display of the legend, KDE plot, dip test, mean, and standard deviation
plot_legend = True
legend_loc = 'upper left'
ncol = 1
plot_kde = True
perform_dip_test = True
plot_mean = True
plot_std = True
# Rounding specifications for the mean and standard deviation when displayed
mean_round_decimal = 1
std_round_decimal = 3
# Specifications for the figure size and font sizes used in the plot
figure_size = (25, 25)
font_size = 50
label_size = font_size/1.125  # Calculate label size as a function of font size
# Annotations to be placed on the plot, including their locations, colors, font sizes, and rotation
place_text = ['Note1', 'Note2']
place_text_loc = [[0.5, 0.5], [0.8, 0.8]]
place_text_color = ['purple', 'orange']
text_font_size = [font_size, font_size]
rotate_place_text = [True, False]
make_text_bold = True
# Padding specifications for layout adjustments
pad = 0.25
handletextpad = 0.5
# Flags and parameters related to saving the plot, including format and file naming
save_svg = True
set_ylims_one_extra_tick = True
remove_first_y = True
return_fig_instead_of_save = False
# Invocation of the parametrized plotting function with all the specified parameters
plotfuns.plot_parameterized_hist(data_dictionary, bins, xlabel, ylabel, figure_title, data_path_out, filename,
                                 color_dict, alpha, hist_stat=hist_stat, xlims=xlims, ylims=ylims, xticks=xticks, yticks=yticks,
                                 plot_vertical_th_line=plot_vertical_th_line, legend_dictionary=legend_dictionary,
                                 plot_legend=plot_legend, legend_loc=legend_loc, ncol=ncol, plot_kde=plot_kde,
                                 perform_dip_test=perform_dip_test, plot_mean=plot_mean, plot_std=plot_std,
                                 mean_round_decimal=mean_round_decimal, std_round_decimal=std_round_decimal,
                                 figure_size=figure_size, font_size=font_size, label_size=label_size, place_text=place_text,
                                 place_text_loc=place_text_loc, place_text_color=place_text_color, place_text_font_size=text_font_size,
                                 rotate_place_text=rotate_place_text, make_text_bold=make_text_bold, pad=pad,
                                 handletextpad=handletextpad, save_svg=save_svg, set_ylims_one_extra_tick=set_ylims_one_extra_tick,
                                 remove_first_y=remove_first_y, return_fig_instead_of_save=return_fig_instead_of_save)