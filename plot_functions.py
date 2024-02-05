"""
This script contains functions that are used only for plotting data.
"""

# Modules
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from sklearn.linear_model import LinearRegression
from mpl_axes_aligner import align
from sklearn.metrics import r2_score
import diptest
import matplotlib.gridspec as gridspec

plt.rcParams["font.family"] = "Arial"


'''
This is a parameterized function for histograms. You can ignore the arguments with default values if you want and only pass the ones with no default values. 
Make sure you read the description of each argument to see what it does.
1) data_dictionary: A dictionary that contains distributions for which you want their histograms in the form {'Dist1': numpy.array, 'Dist2': numpy.array, etc}
2) xlabel: The label on the xaxis
3) ylabel: The label on the yaxis
4) figure_title: The title of the figure
5) color_dict: A dictionary that contains the color of each distribution you want to plot in the form {'Dist1': 'color1', 'Dist2': 'color2', etc}
6) bins: A numpy.array containing the bins to use (e.g. np.arange(0, 1.1, 0.1) for dists in [0, 1])
7) alpha: A dictionary that with the alpha / opacity value (in [0, 1]) for each distribution you want to plot in the form {'Dist1': value1, 'Dist2': value2, etc}
8) data_path_out: The absolute path of the folder at which you want to store the figure (without the name of the file). That path must exist so you can create it before the call to this function if it 
doesn't. You could also add that part inside this function if you want.
9) filename: The name of the file where the plot will be stored in the form '\Somename' (you don't need to add a file extension)
10) remove_first_y: True/False flag which removes the first tick on the y axis (this is used in cases where the x and the y axis have the same starting tick and you don't want them to be overlapping)
11) xlims: A list of min and max values to be shown for the xaxis [min_value, max_value]
12) ylims: A list of min and max values to be shown for the yaxis [min_value, max_value]
13) figure_size: A tuple for the size of the image (width, height)
14) legend_loc: The location of the legend
15) ncol: The number of columns used to show the legend
16) font_size: The font size of the xlabel, ylabel, title, and the legend. This param is also used in order to adjust the patches of the legend. Could use a better practice but w/e :P
17) label_size: The font size of the xticks, yticks
18) hue: I believe this doesn't really work. You can test it yourself.
19) save_svg: Whether you want to save the image in an '.svg' format. The image is automatically saved as a '.png' either way.
20) hist_stat: The type of yaxis you want for the histogram (e.g. 'probability', 'frequency', etc). Check the sns.histplot documentation for options.
21) xticks: A numpy.array or list of values to show as ticks in the xaxis.
22) yticks: A numpy.array or list of values to show as ticks in the yaxis.
23) plot_vertical_th_line: A list of thresholds for the xaxis to plot vertical lines (e.g. [th1, th2, th3, etc]. You could add parameters for the vertical line to be more modular.
24) set_ylims_one_extra_tick: True/False flag that expands the yaxis limits for one extra tick.
25) plot_legend: True/False flag on whether you want to plot the legend.
26) legend_dictionary: A dictionary that contains the legend name of each distribution you want to plot in the form {'Dist1': 'Legend1', 'Dist2': 'Legend2', etc}
27) pad: The padding used for the image with tight layout
28) place_text: A list containing strings of texts you want to annotate in the image (e.g. ['text1', 'text2', etc]
29) place_text_loc: A list containing a [x, y] list for the location of each annotated text (e.g. [[x1, y1], [x2, y2], etc])
30) text_font_size: A list containing the font size of each annotated text (I usually set this either to font_size or label_size, e.g. [font_size, font_size, etc])
31) make_text_bold: True/False flag on whether you want the annotated text to be bold or not (you could change this to apply to each individual text)
32) place_text_color: A list containing the color of each annotated text (['color1', 'color2', etc]) 
33) handletextpad: The padding between the legend text and its patch
34) plot_kde: True/False flag on whether you want to plot a kernel density estimation function for each histogram
35) perform_dip_test: True/False flag on whether you want to perform the Hartigans' dip test for each histogram to see whether it is a multimodal distribution
36) plot_mean: True/False flag on whether you want to plot the mean of each distribution in the legend
37) plot_std: True/False flag on whether you want to plot the standard deviation of each distribution in the legend
38) mean_round_decimal: The decimal point at which to round the mean of each distribution
39) std_round decimal: The decimal point at which to round the std of each distribution
40) use_italics_for_stats: True/False flag on whether you want to use italic fonts for the mean and std statistics if plotted in the legend
41) return_fig_instead_of_save: True/False flag, which if False the plot is stored in an image, whereas if it's True the image is returned as an object.
'''
# Create a parametrized function for histograms
def plot_parameterized_hist(data_dictionary, xlabel, ylabel, figure_title, color_dict, bins,
                            alpha, data_path_out, filename, remove_first_y=False, xlims=None, ylims=None,
                            figure_size=(25, 25), legend_loc='upper right', ncol=2, font_size=115, label_size=100,
                            hue=False, save_svg=True, hist_stat='probability', xticks=None, yticks=None, plot_vertical_th_line=None,
                            set_ylims_one_extra_tick=False, plot_legend=True, legend_dictionary=None, pad=0.25, place_text=None, place_text_loc=None,
                            text_font_size=None, make_text_bold=False, place_text_color=None, handletextpad=1, plot_kde=False, perform_dip_test=False,
                            plot_mean=False, plot_std=False, mean_round_decimal=2, std_round_decimal=2, use_italics_for_stats=False, return_fig_instead_of_save=False):
    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()

    fig, ax = plt.subplots(1)
    fig.set_size_inches(figure_size)
    sns.set(font_scale=5.0, style='ticks')
    # plt.rcParams["font.weight"] = "bold"
    # Get legends aka data dictionary keys
    legends = [key for key in data_dictionary.keys()]

    # # Max, min x
    # min_x = min([min(data_dictionary[i]) for i in data_dictionary.keys()])
    # max_x = max([max(data_dictionary[i]) for i in data_dictionary.keys()])
    # # X range
    # #x_range = max_x - min_x
    # #binwidth = x_range / 20
    # patches
    patches = []
    for i in range(len(legends)):
        data_dist = data_dictionary[legends[i]]

        if not hue:
            if plot_kde:
                sns.histplot(data=data_dist, ax=ax, color=color_dict[legends[i]], stat=hist_stat, bins=bins,
                             alpha=alpha[legends[i]], edgecolor='none', kde=plot_kde, line_kws={'alpha': 1, 'linewidth': 7})
            else:
                sns.histplot(data=data_dist, ax=ax, color=color_dict[legends[i]], stat=hist_stat, bins=bins,
                             alpha=alpha[legends[i]], edgecolor='none')

        legend_suffix = ''

        if perform_dip_test:
            dip, pval = diptest.diptest(data_dist)
            legend_suffix += f', Dip test pvalue: {round(pval, 3)}'

        if plot_mean:
            data_mean = round(np.mean(data_dist), mean_round_decimal)
            if use_italics_for_stats:
                legend_suffix += f'\n$\it μ: {data_mean}$'
                # temp = f' $\it μ: {data_mean}$'
                # legend_suffix += f'$\\bf{temp}$'
            else:
                legend_suffix += f', μ: {data_mean}'

        if plot_std:
            data_std = round(np.std(data_dist, ddof=1), std_round_decimal)
            if use_italics_for_stats:
                legend_suffix += f'\n$\it σ: {data_std}$'
                # temp = f' $\it σ: {data_std}$'
                # legend_suffix += f'$\\bf{temp}$'
            else:
                legend_suffix += f', σ: {data_std}'


        if legend_dictionary is not None:
            if legend_dictionary[legends[i]] != '':
                patches.append(mpatches.Patch(color=color_dict[legends[i]], label=legend_dictionary[legends[i]] + legend_suffix))
        else:
            patches.append(mpatches.Patch(color=color_dict[legends[i]], label=legends[i] + legend_suffix))

        # if plot_kde:
        #     sns.kdeplot(data=data_dist, color='black')

    if hue:
        all_dists = pd.DataFrame(data_dictionary)
        all_dists = pd.melt(all_dists, value_vars=legends, var_name='melt')
        sns.histplot(data=all_dists, ax=ax, color=color_dict, stat=hist_stat, bins=bins,
                     alpha=alpha, edgecolor='none', hue='melt', multiple='dodge')

    if plot_vertical_th_line is not None:
        for th in plot_vertical_th_line:
            plt.axvline(x=th, color='black', linewidth=5, linestyle='dashed')

    # Max, min y
    # min_y, max_y = ax.get_ylim()

    ax.tick_params(which='major', axis='x', direction='out', pad=20)
    ax.tick_params(which='major', axis='y', direction='out', pad=20)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticks is not None:
        ax.set_xticks(xticks)

    # ax.set_xlim([min_x, max_x])
    # ax.set_ylim([min_y, max_y])
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=20, fontsize=font_size)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=20, fontsize=font_size)
    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)
    ax.tick_params(axis='x', pad=15)

    if set_ylims_one_extra_tick:
        yticks_default = ax.get_yticks()
        y_tick_step = yticks_default[1] - yticks_default[0]
        ax.set_ylim([yticks_default[0], yticks_default[-1] + y_tick_step])

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontweight('bold') for label in labels]

    if xlims:
        ax.set_xlim([xlims[0], xlims[1]])
    if ylims:
        ax.set_ylim([ylims[0], ylims[1]])
    # plt.locator_params(axis='x', nbins=3)
    # plt.locator_params(axis='y', nbins=4)
    #plt.yticks(rotation=90)
    if plot_legend:
        legend = ax.legend(handles=patches, handlelength=1, frameon=False, loc=legend_loc, ncol=ncol, handletextpad=handletextpad)
        for patch in legend.get_patches():
            patch.set_height(font_size - 30)
            patch.set_width(font_size - 30)
            patch.set_x(-font_size/4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if remove_first_y:
        ax.yaxis.get_major_ticks()[0].set_visible(False)
    [i.set_linewidth(7) for i in ax.spines.values()]
    ax.xaxis.set_tick_params(width=5, length=20, direction='out')
    ax.yaxis.set_tick_params(width=5, length=20, direction='out')
    if plot_legend:
        plt.setp(ax.get_legend().get_texts(), fontsize=font_size, fontweight='bold')
    plt.title(figure_title, fontsize=font_size, fontweight='bold')

    if place_text is not None:
        if type(place_text) == list:
            for i in range(len(place_text)):
                ax.annotate(place_text[i], xy=place_text_loc[i], xycoords='axes fraction',
                            fontsize=text_font_size[i], fontweight=f'{"bold" if make_text_bold else "normal"}',
                            color=f'{place_text_color[i] if place_text_color[i] is not None else "black"}')

        elif place_text:
            ax.annotate(place_text, xy=place_text_loc, xycoords='axes fraction',
                        fontsize=text_font_size, fontweight=f'{"bold" if make_text_bold else "normal"}',
                        color=f'{place_text_color if place_text_color is not None else "black"}')

    fig.tight_layout(pad=pad)

    if return_fig_instead_of_save:
        return fig
    else:
        # Save figure as png
        data_path = data_path_out + filename + '.png'
        plt.savefig(data_path)
        # Save as svg
        if save_svg:
            data_path = data_path_out + f'\SVG'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            data_path = data_path + filename + '.svg'

            plt.savefig(data_path, format='svg')

    plt.close()


'''
This is a parameterized function for line plots. You can ignore the arguments with default values if you want and only pass the ones with no default values. 
Make sure you read the description of each argument to see what it does.
1) data_dictionary_x: A dictionary that contains the x values in a numpy.array or a list of each line you want to plot {'Line1': xvalues1, 'Line2': xvalues2, etc}
2) data_dictionary_y: A dictionary that contains the y values in a numpy.array or a list of each line you want to plot {'Line1': yvalues1, 'Line2': yvalues2, etc}
2) xlabel: The label on the xaxis
3) ylabel: The label on the yaxis
4) figure_title: The title of the figure
5) data_path_out: The absolute path of the folder at which you want to store the figure (without the name of the file). That path must exist so you can create it before the call to this function if it 
doesn't. You could also add that part inside this function if you want.
6) filename: The name of the file where the plot will be stored in the form '\Somename' (you don't need to add a file extension)
7) color_dictionary: A dictionary that contains the color of each line you want to plot in the form {'Line1': 'color1', 'Line2': 'color2', etc}
8) legend_dictionary: A dictionary that contains the legend name of each line you want to plot in the form {'Line1': 'Legend1', 'Line2': 'Legend2', etc}
9) lw_dict: A dictionary that contains the linewidth of each line you want to plot in the form {'Line1': lw1, 'Line2': lw2, etc}
10) ls_dict: A dictionary that contains the linestyle of each line you want to plot in the form {'Line1': ls1, 'Line2': ls2, etc}
11) fs_dict: A dictionary that contains the font size of each text in the form {'xlabel': font_size1, 'ylabel': font_size2, etc} for the title, xlabel, ylabel, xticks, yticks, and legend
12) m_dict: A dictionary that contains the marker style for the points in each line you want to plot in the form {'Line1': 'style1', 'Line2': 'style2', etc}
13) ms_dict: A dictionary that contains the marker size for the points in each line you want to plot in the form {'Line1': 'size1', 'Line2': 'size2', etc}
14) figure_size: A tuple for the size of the image (width, height)
15) xticks_values: A numpy.array or list of values to show as ticks in the xaxis.
16) yticks_values: A numpy.array or list of values to show as ticks in the yaxis.
17) xticks_labels: A numpy.array or list of labels to overwrite the ticks in the xaxis.
18) yticks_labels: A numpy.array or list of labels to overwrite the ticks in the xaxis.
19) legend_loc: The location of the legend
20) data_dictionary_yerr: A dictionary that contains the standard error of the mean values in a numpy.array or a list for each point of each line you want to plot {'Line1': yerrvalues1, 'Line2': yerrvalues2, etc}
21) xlims: A list of min and max values to be shown for the xaxis [min_value, max_value]
22) ylims: A list of min and max values to be shown for the yaxis [min_value, max_value]
23) use_sem_instead_of_errorbar: True/False flag that shows the standard error of the mean values as a shaded area when true and as error bars when false
24) alpha_dict: A dictionary that with the alpha / opacity value (in [0, 1]) for each line you want to plot in the form {'Line1': value1, 'Line2': value2, etc}
25) save_svg: Whether you want to save the image in an '.svg' format. The image is automatically saved as a '.png' either way.
26) show_lines: True/False flag on whether you want to show the lines in the legend or not
27) place_text: A list containing strings of texts you want to annotate in the image (e.g. ['text1', 'text2', etc]
28) place_text_loc: A list containing a [x, y] list for the location of each annotated text (e.g. [[x1, y1], [x2, y2], etc])
29) make_text_bold: True/False flag on whether you want the annotated text to be bold or not (you could change this to apply to each individual text)
30) place_text_font_size: A list containing the font size of each annotated text (I usually set this either to font_size or label_size, e.g. [font_size, font_size, etc])
31) rotate_place_text: True/False flag on whether you want to write the annotated text vertically or horizontally
32) place_text_color: A list containing the color of each annotated text (['color1', 'color2', etc]) 
33) null_dict: I should probably remove this becasuse I am not sure it actually does anything useful
34) null_color: I should probably remove this becasuse I am not sure it actually does anything useful
35) use_log_scale: True/False flag on whether you want both axis to be in log scale
36) use_mask: True/False flag on whether you want to plot only the points with finite y values (useful if you have nan or empty values)
37) tick_width: The width of the ticks in the x and y axis.
38) tick_length: The length of the ticks in the x and y axis
39) yerr_capsize: The width of the cap in the error bar
40) yerr_elinewidth: The linewidth of the error bar line
41) capthick: The thickness of the cap in the error bar
42) spine_width: The width of the axis lines
43) pad: The padding used for the image with tight layout
44) use_sns: True/False flag on whether to use the seaborn library for setting up some params
45) plot_vertical_line: True/False flag on whether you want to plot a vertical line
46) vertical_line_x: The value on the x axis where you want the vertical line to be
47) vertical_line_color: The color of the vertical line
48) vertical_line_ls: The linestyle of the vertical line
49) vertical_line_lw: The linewidth of the vertical line
50) plot_horizontal_line: True/False flag on whether you want to plot a horizontal line
51) horizonal_line_y: The value on the y axis where you want the horizontal line to be
52) horizontal_line_color: The color of the horizontal line
53) horizontal_line_ls: The linestyle of the horizontal line
54) horizontal_line_lw: The linewidth of the horizontal line
55) add_colorbar: True/False flag on whether you want to add a colorbar
56) colorbar_colormap: The colormap you want to use for your colorbar
57) colorbar_label: The label for the colorbar
58) colorbar_loc: The location of the colorbar
59) colorbar_ticks: A numpy.array or list of values to show as ticks in the colorbar.
60) colorbar_tick_labels: A numpy.array or list of labels to overwrite the ticks in the colorbar.
61) colorbar_orientation: 'vertical'/'horizontal' depending on how you want the colorbar to be
62) legend_line_length: The length of the lines in the legend
63) use_line_color_for_error: Whether you want to use as a color for the error bars the same color as the one their corresponding line has.
64) pad_ticks: True/False flag on whether you want to pad the ticks in the x and y axis.
65) pad_labels: True/False flag on whether you want to pad the labels in the x and y axis. 
66) ncol: The number of columns used to show the legend
67) handletextpad: The padding between the legend text and its line
68) bbox_to_anchor: The location of the bounding box corner to use to frame the legend
69) font_size: The font size of the xlabel, ylabel, title, and the legend. The font size of the ticks is defined as font_size/1.125
70) legend_lw_multiplier: A param to set the linewidth of the line in the legend
71) use_line_color_for_legends: True/False flag on whether you want the text of the legends to be the same as their corresponding line
72) return_fig_instead_of_save: True/False flag, which if False the plot is stored in an image, whereas if it's True the image is returned as an object.
'''
# Create a fully parametrized function that plots single or multiple [x, y] distributions of points with the settings
# desired and stores the figure at given path.
def plot_parameterized_lineplot(data_dictionary_x, data_dictionary_y, xlabel, ylabel, figure_title, data_path_out, filename,
                           color_dictionary=None, legend_dictionary=None, lw_dict=None, ls_dict=None, fs_dict=None,
                           m_dict=None, ms_dict=None, figure_size=None, xticks_values=None, yticks_values=None,
                           xticks_labels=None, yticks_labels=None, legend_loc="best", data_dictionary_yerr=None, xlims=None, ylims=None,
                           use_sem_instead_of_errorbar=True, alpha_dict=None, save_svg=True, show_lines=True, place_text=None, place_text_loc=None, make_text_bold=True, place_text_font_size=None,
                           rotate_place_text=False, place_text_color=None, null_dict=None, null_color='grey', use_log_scale=False, use_mask=True, tick_width=4, tick_length=16, yerr_capsize=20, yerr_elinewidth=5, capthick=3,
                           spine_width=7, pad=0.25, use_sns=True, plot_vertical_line=False, vertical_line_x=0, vertical_line_color='black', vertical_line_ls='dashed',
                           vertical_line_lw=4, plot_horizontal_line=False, horizontal_line_y=0, horizontal_line_color='black', horizontal_line_ls='dashed',
                           horizontal_line_lw=4, add_colorbar=False, colorbar_colormap=None, colorbar_label=None, colorbar_loc='right', colorbar_ticks=None, colorbar_tick_labels=None,
                           colorbar_orientation='vertical', legend_line_length=1, use_line_color_for_error=False, pad_ticks=True, pad_labels=True,
                           ncol=1, handletextpad=1, bbox_to_anchor=None, font_size=90, legend_lw_multiplier=3, use_line_color_for_legends=False,
                           return_fig_instead_of_save=False):
    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()

    # Setup styles
    if use_sns:
        sns.set(font_scale=5.0, style='ticks')
        plt.rcParams["font.weight"] = "bold"

    # Get the keys of the dictionaries
    dict_keys = [j for j in data_dictionary_x.keys()]

    # If any of the arguments are None then set their values for all keys to default
    # Color Default = Blue
    if color_dictionary is None:
        color_dictionary = {}
        for key in dict_keys:
            color_dictionary[key] = 'blue'

    # Linewidth Default = 2
    if lw_dict is None:
        lw_dict = {}
        for key in dict_keys:
            lw_dict[key] = 7

    # Linestyle Default = 'solid'
    if ls_dict is None:
        ls_dict = {}
        for key in dict_keys:
            ls_dict[key] = 'solid'

    # Font size Default = 16
    if fs_dict is None:
        fs_dict = {}
        fs_dict['title'] = font_size
        fs_dict['xlabel'] = font_size
        fs_dict['ylabel'] = font_size
        fs_dict['xticks'] = font_size / 1.125
        fs_dict['yticks'] = font_size / 1.125
        fs_dict['legend'] = font_size

    # Marker Default = None
    if m_dict is None:
        m_dict = {}
        for key in dict_keys:
            m_dict[key] = None

    # Marker Size Default = None
    if ms_dict is None:
        ms_dict = {}
        for key in dict_keys:
            ms_dict[key] = None

    if alpha_dict is None:
        alpha_dict = {}
        for key in dict_keys:
            alpha_dict[key] = 1

    # Initialize plot
    if figure_size:
        fig, ax = plt.subplots(1)
        fig.set_size_inches(figure_size)
    # Figure Size Default = (12, 8)
    else:
        fig, ax = plt.subplots(figsize=(25, 25))
    # For every key in the dictionary plot its points with the respective settings
    for key in dict_keys:
        if not isinstance(None, type(data_dictionary_x[key])):
            if data_dictionary_yerr:
                if use_sem_instead_of_errorbar:
                    std_mean_obs_upper = data_dictionary_y[key] + data_dictionary_yerr[key]
                    std_mean_obs_lower = data_dictionary_y[key] - data_dictionary_yerr[key]

                    # Plot upper error
                    plt.plot(data_dictionary_x[key], std_mean_obs_upper, linestyle='None', color=color_dictionary[key], alpha=0.3)
                    # Plot lower error
                    plt.plot(data_dictionary_x[key], std_mean_obs_lower, linestyle='None', color=color_dictionary[key], alpha=0.3)
                    # Fill the in betweens
                    plt.fill_between(data_dictionary_x[key], std_mean_obs_upper, std_mean_obs_lower, color=color_dictionary[key],
                                     alpha=0.3)
                else:
                    if use_line_color_for_error:
                        ax.errorbar(data_dictionary_x[key], data_dictionary_y[key], yerr=data_dictionary_yerr[key], ecolor=color_dictionary[key], elinewidth=yerr_elinewidth, capsize=yerr_capsize,
                                    color=color_dictionary[key], linewidth=lw_dict[key], linestyle=ls_dict[key], marker=m_dict[key], markersize=ms_dict[key], capthick=capthick)
                    else:
                        ax.errorbar(data_dictionary_x[key], data_dictionary_y[key], yerr=data_dictionary_yerr[key], ecolor='black', elinewidth=yerr_elinewidth, capsize=yerr_capsize,
                                    color=color_dictionary[key], linewidth=lw_dict[key], linestyle=ls_dict[key], marker=m_dict[key], markersize=ms_dict[key], capthick=capthick)
            if use_mask:
                mask = np.isfinite(data_dictionary_y[key])

                if null_dict:
                    if null_dict[key]:
                        plt.plot(data_dictionary_x[key][mask], data_dictionary_y[key][mask], color=null_color, linewidth=lw_dict[key],
                                 linestyle=ls_dict[key], marker=m_dict[key], markersize=ms_dict[key], alpha=alpha_dict[key])
                else:
                    plt.plot(data_dictionary_x[key][mask], data_dictionary_y[key][mask], color=color_dictionary[key], linewidth=lw_dict[key],
                             linestyle=ls_dict[key], marker=m_dict[key], markersize=ms_dict[key], alpha=alpha_dict[key])
            else:
                if null_dict:
                    if null_dict[key]:
                        plt.plot(data_dictionary_x[key], data_dictionary_y[key], color=null_color, linewidth=lw_dict[key],
                                 linestyle=ls_dict[key], marker=m_dict[key], markersize=ms_dict[key], alpha=alpha_dict[key])
                else:
                    plt.plot(data_dictionary_x[key], data_dictionary_y[key], color=color_dictionary[key], linewidth=lw_dict[key],
                             linestyle=ls_dict[key], marker=m_dict[key], markersize=ms_dict[key], alpha=alpha_dict[key])

    # Set title, xlabel, ylabel, xticks, yticks
    ax.set_title(figure_title, fontsize=fs_dict['title'], fontweight='bold')
    if pad_labels:
        ax.set_xlabel(xlabel, fontsize=fs_dict['xlabel'], labelpad=fs_dict['xlabel'] / 5, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=fs_dict['ylabel'], labelpad=fs_dict['ylabel'] / 5, fontweight='bold')
    else:
        ax.set_xlabel(xlabel, fontsize=fs_dict['xlabel'], fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=fs_dict['ylabel'], fontweight='bold')
    if pad_ticks:
        ax.tick_params(which='major', axis='x', direction='out', pad=15)
        ax.tick_params(which='major', axis='y', direction='out', pad=20)
    [i.set_linewidth(spine_width) for i in ax.spines.values()]
    ax.xaxis.set_tick_params(width=tick_width, length=tick_length, direction='out')
    ax.yaxis.set_tick_params(width=tick_width, length=tick_length, direction='out')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Set xticks if given, otherwise use default
    if not isinstance(None, type(xticks_values)):
        ax.set_xticks(xticks_values)
    # Set yticks if given, otherwise use default
    if not isinstance(None, type(yticks_values)):
        ax.set_yticks(yticks_values)

    # Plot xticklabels if given
    if not isinstance(None, type(xticks_labels)):
        ax.set_xticklabels(labels=xticks_labels, fontsize=fs_dict['xticks'], fontweight='bold')
    # Plot yticklabels if given
    if not isinstance(None, type(yticks_labels)):
        ax.set_yticklabels(labels=yticks_labels, fontsize=fs_dict['yticks'], fontweight='bold')

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)

    if use_log_scale:
        plt.yscale('log')
        plt.xscale('log')

    ax.tick_params(axis='x', labelsize=fs_dict['xticks'])
    ax.tick_params(axis='y', labelsize=fs_dict['yticks'])
    # Plot legends if given
    if legend_dictionary:
        legend_handles = list()
        labels_used = list()
        for key in legend_dictionary.keys():
            if legend_dictionary[key] not in labels_used and legend_dictionary[key] != '':
                labels_used.append(legend_dictionary[key])
                if null_dict:
                    if null_dict[key]:
                        legend_handles.append(mlines.Line2D([], [], linestyle=ls_dict[key], color=null_color,
                                                            label=legend_dictionary[key], linewidth=legend_lw_multiplier * lw_dict[key],
                                                            marker=m_dict[key], markersize=ms_dict[key], visible=show_lines))
                else:
                    legend_handles.append(mlines.Line2D([], [], linestyle=ls_dict[key], color=color_dictionary[key],
                                                        label=legend_dictionary[key], linewidth=legend_lw_multiplier * lw_dict[key],
                                                        marker=m_dict[key], markersize=ms_dict[key], visible=show_lines))

        if bbox_to_anchor is not None:
            if show_lines:
                ax.legend(handles=legend_handles, loc=legend_loc, frameon=False, prop={'size': fs_dict['legend']}, handlelength=legend_line_length, ncol=ncol, handletextpad=handletextpad,
                          bbox_to_anchor=bbox_to_anchor, labelcolor=f'{"linecolor" if use_line_color_for_legends else "None"}')
            else:
                ax.legend(handles=legend_handles, loc=legend_loc, frameon=False, prop={'size': fs_dict['legend']}, labelcolor='linecolor', handlelength=legend_line_length, ncol=ncol,
                          bbox_to_anchor=bbox_to_anchor)
        else:
            if show_lines:
                ax.legend(handles=legend_handles, loc=legend_loc, frameon=False, prop={'size': fs_dict['legend']}, handlelength=legend_line_length, ncol=ncol, handletextpad=handletextpad,
                          labelcolor=f'{"linecolor" if use_line_color_for_legends else "None"}')
            else:
                ax.legend(handles=legend_handles, loc=legend_loc, frameon=False, prop={'size': fs_dict['legend']}, labelcolor='linecolor', handlelength=legend_line_length, ncol=ncol)

    # If there is text to place
    if place_text is not None:
        if type(place_text) == list:
            if place_text_color is None:
                place_text_color = [None for j in range(len(place_text))]
            for i in range(len(place_text)):
                ax.annotate(place_text[i], xy=place_text_loc[i], xycoords='axes fraction', color=place_text_color[i],
                            fontsize=place_text_font_size[i], fontweight=f'{"bold" if make_text_bold else "normal"}', rotation=f'{"vertical" if rotate_place_text[i] else "horizontal"}')

        elif place_text:
            if place_text_color is None:
                place_text_color = None
            ax.annotate(place_text, xy=place_text_loc, xycoords='axes fraction', color=place_text_color,
                        fontsize=place_text_font_size, fontweight=f'{"bold" if make_text_bold else "normal"}', rotation=f'{"vertical" if rotate_place_text else "horizontal"}')

    if plot_vertical_line:
        plt.axvline(x=vertical_line_x, color=vertical_line_color, linewidth=vertical_line_lw, linestyle=vertical_line_ls)

    if plot_horizontal_line:
        plt.axhline(y=horizontal_line_y, color=horizontal_line_color, linewidth=horizontal_line_lw, linestyle=horizontal_line_ls)

    if add_colorbar:
        cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=colorbar_colormap), ax=ax, location=colorbar_loc, label=colorbar_label, ticks=colorbar_ticks, orientation=colorbar_orientation)
        cbar.set_ticklabels(colorbar_tick_labels, fontweight='bold', fontsize=fs_dict['legend'])
        cbar.set_label(colorbar_label, fontweight='bold', fontsize=fs_dict['legend'])

    fig.tight_layout(pad=pad)

    if return_fig_instead_of_save:
        return fig

    else:
        # Save the figure
        fig.savefig(data_path_out + filename + '.png')
        if save_svg:
            data_path_svg = data_path_out + r'\SVG'
            if not os.path.exists(data_path_svg):
                os.makedirs(data_path_svg)
            fig.savefig(data_path_svg + filename + '.svg', format='svg')

    plt.close()



'''
This is a parameterized function for scatter plots. You can ignore the arguments with default values if you want and only pass the ones with no default values. 
Make sure you read the description of each argument to see what it does.
1) data_dictionary_x: A dictionary that contains the x values in a numpy.array or a list of each distribution you want to plot {'Dist1': xvalues1, 'Dist2': xvalues2, etc}
2) data_dictionary_y: A dictionary that contains the y values in a numpy.array or a list of each distribution you want to plot {'Dist1': yvalues1, 'Dist2': yvalues2, etc}
2) xlabel: The label on the xaxis
3) ylabel: The label on the yaxis
4) alpha: A dictionary that defines the alpha / opacity value (in [0, 1]) for each distribution you want to plot in the form {'Dist1': value1, 'Dist2': value2, etc}
5) figure_title: The title of the figure
6) data_path_out: The absolute path of the folder at which you want to store the figure (without the name of the file). That path must exist so you can create it before the call to this function if it 
doesn't. You could also add that part inside this function if you want.
7) filename: The name of the file where the plot will be stored in the form '\Somename' (you don't need to add a file extension)
8) xlims: A list of min and max values to be shown for the xaxis [min_value, max_value]
9) ylims: A list of min and max values to be shown for the yaxis [min_value, max_value]
10) color_dictionary: A dictionary that contains the color of each distribution you want to plot in the form {'Dist1': 'color1', 'Dist2': 'color2', etc}
11) legend_dictionary: A dictionary that contains the legend name of each distribution you want to plot in the form {'Dist1': 'Legend1', 'Dist2': 'Legend2', etc}
12) font_size: The font size of the xlabel, ylabel, title, and the legend. This param is also used in order to adjust the patches of the legend. Could use a better practice but w/e :P
13) label_size: The font size of the xticks, yticks
14) m_dict: A dictionary that contains the marker style for the points in each distribution you want to plot in the form {'Dist1': 'style1', 'Dist2': 'style2', etc}
15) ms_dict: A dictionary that contains the marker size for the points in each distribution you want to plot in the form {'Dist1': 'size1', 'Dist2': 'size2', etc}
16) figure_size: A tuple for the size of the image (width, height)
17) xticks_values: A numpy.array or list of values to show as ticks in the xaxis.
18) yticks_values: A numpy.array or list of values to show as ticks in the yaxis.
19) xticks_labels: A numpy.array or list of labels to overwrite the ticks in the xaxis.
20) yticks_labels: A numpy.array or list of labels to overwrite the ticks in the xaxis.
21) legend_loc: The location of the legend
22) plot_linear_line: True/False flag on whether you want to plot the y=x line
23) plot_r2_score: True/False flag on whether you want to annotate as text the R2 score for each distribution from a linear fit. Must set do_linear_reg_fit to True.
24) save_svg: True/False flag on whether you want to save the image in an '.svg' format. The image is automatically saved as a '.png' either way.
25) m_size_factor: A value to change the size of the markers for all points (it multiplies the value from ms_dict)
26) use_log_scale: True/False flag on whether you want both axis to be in log scale
27) colormap: A color palette to use as a color map
28) color_values: ??
29) colorbar_label: The label for the colorbar if you are using a colormap
30) place_text: A list containing strings of texts you want to annotate in the image (e.g. ['text1', 'text2', etc]
31) place_text_loc: A list containing a [x, y] list for the location of each annotated text (e.g. [[x1, y1], [x2, y2], etc])
32) place_text_font_size: A list containing the font size of each annotated text (I usually set this either to font_size or label_size (e.g. [font_size, font_size, etc])
33) make_text_bold: True/False flag on whether you want the annotated text to be bold or not (you could change this to apply to each individual text)
34) rotate_place_text: True/False flag on whether you want to write the annotated text vertically or horizontally
35) place_text_color: A list containing the color of each annotated text (e.g. ['color1', 'color2', etc]) 
36) pad: The padding used for the image with tight layout
37) plot_horizontal_line: True/False flag on whether you want to plot a horizontal line
38) horizonal_line_y: A list that contains values in the y axis where you want the horizontal lines to be (e.g. [5, -1, etc]). If you want 1 line just use 1 value in a list.
39) horizontal_line_color: A list that contains the color of each horizontal line (e.g. ['color1', 'color2', etc]) 
40) horizontal_line_ls: A list that contains the linestyle of each horizontal line (e.g. ['solid', 'dashed', etc]) 
41) horizontal_line_lw: A list that contains the linewidth of each horizontal line (e.g. [2, 5, etc]) 
42) do_linear_reg_fit: True/False flag on whether you want to perform a linear regression fitting on each distribution and plot its line
43) linear_reg_fit_lw: The linewidth of the linear regression fitted line for each distribution
44) plot_vertical_line: True/False flag on whether you want to plot a vertical line
45) vertical_line_x: A list that contains values in the x axis where you want the vertical lines to be (e.g. [5, -1, etc]). If you want 1 line just use 1 value in a list.
46) vertical_line_color: A list that contains the color of each vertical line (e.g. ['color1', 'color2', etc]) 
47) vertical_line_ls: A list that contains the linestyle of each vertical line (e.g. ['solid', 'dashed', etc]) 
48) vertical_line_lw: A list that contains the linewidth of each vertical line (e.g. [2, 5, etc]) 
49) spine_width: The width of the axis lines
50) plot_y_equal_x_line: True/False flag on whether you want to plot the y=x line (review what this does and why it differs from plot_linear_line
51) ncol: The number of columns used to show the legend
52) handletextpad: The padding between the legend text and its patch
53) labelspacing: The spacing between the legend texts 
54) columnspacing: The spacing between the legend columns
55) m_linewidth: The linewidth of a line passing the points in each distribution (basically lineplot)
56) tick_width: The width of the ticks in the x and y axis.
57) tick_length: The length of the ticks in the x and y axis
58) return_fig_instead_of_save: True/False flag, which if False the plot is stored in an image, whereas if it's True the image is returned as an object.
'''
# Create a parameterized scatterplot function
def parameterized_scatterplot(data_dictionary_x, data_dictionary_y, xlabel, ylabel, alpha, figure_title, data_path_out,
                              filename, xlims=None, ylims=None, color_dictionary=None, legend_dictionary=None, font_size=90, label_size=90/1.125,
                              m_dict=None, ms_dict=None, figure_size=(25, 25), xticks_values=None, yticks_values=None,
                              xticks_labels=None, yticks_labels=None, legend_loc="best", plot_linear_line=False, plot_r2_score=False,
                              save_svg=True, m_size_factor=4, use_log_scale=False, colormap=None, color_values=None, colorbar_label=None,
                              place_text=None, place_text_loc=None, place_text_font_size=None, make_text_bold=None, rotate_place_text=None,
                              place_text_color=None, pad=0.25, plot_horizontal_line=False, horizontal_line_y=[0],
                              horizontal_line_color=['black'], horizontal_line_ls=['dashed'], horizontal_line_lw=[4], do_linear_reg_fit=False, linear_reg_fit_lw=5, plot_vertical_line=False,
                              vertical_line_x=[0], vertical_line_color=['black'], vertical_line_ls=['dashed'], vertical_line_lw=[4], spine_width=7, plot_y_equal_x_line=False, ncol=1, handletextpad=0.8,
                              labelspacing=0.5, columnspacing=2, m_linewidth=1, tick_width=4, tick_length=16, return_fig_instead_of_save=False):
    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()
    # Font size = 16

    fig, ax = plt.subplots(1)
    fig.set_size_inches(figure_size)
    sns.set(font_scale=5.0, style='ticks')

    # Get legends aka data dictionary keys
    legends = [key for key in data_dictionary_x.keys()]

    # # Max, min x
    # min_x = min([min(data_dictionary[i]) for i in data_dictionary.keys()])
    # max_x = max([max(data_dictionary[i]) for i in data_dictionary.keys()])
    # # X range
    # #x_range = max_x - min_x
    # #binwidth = x_range / 20
    # patches
    patches = []

    if m_dict is None:
        m_dict = {key: 'o' for key in legends}

    if ms_dict is None:
        ms_dict = {key: font_size for key in legends}

    for i in range(len(legends)):
        # x values for distribution i
        data_dist_x = data_dictionary_x[legends[i]]
        # y values for distribution i
        data_dist_y = data_dictionary_y[legends[i]]


        if colormap is None:
            sns.scatterplot(x=data_dist_x, y=data_dist_y, ax=ax, s=m_size_factor*ms_dict[legends[i]], color=color_dictionary[legends[i]], marker=m_dict[legends[i]], alpha=alpha[legends[i]], linewidth=m_linewidth)
        else:
            sns.scatterplot(x=data_dist_x, y=data_dist_y, ax=ax, s=m_size_factor*ms_dict[legends[i]], marker=m_dict[legends[i]], alpha=alpha[legends[i]], hue=color_values[legends[i]], palette=colormap, linewidth=m_linewidth)
            norm = plt.Normalize(np.nanmin(color_values[legends[i]]), np.nanmax(color_values[legends[i]]))
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])


        # else:
        #     patches.append(mpatches.Patch(color=color_dictionary[legends[i]], label=legends[i]))

        if do_linear_reg_fit:
            fitted_line = LinearRegression().fit(data_dist_x.reshape(-1, 1), data_dist_y.reshape(-1, 1))
            predicted_y = fitted_line.predict(data_dist_x.reshape(-1, 1))
            ax.plot(data_dist_x, predicted_y, color=color_dictionary[legends[i]], linewidth=linear_reg_fit_lw, linestyle='solid', alpha=1)
            # Get R2 score
            if plot_r2_score:
                r2 = r2_score(data_dist_y.reshape(-1, 1), predicted_y)
                # x_to_plot = np.sort(data_dist_x)[int(len(data_dist_x)/2)]
                # y_to_plot = fitted_line.predict(np.array(x_to_plot).reshape(1, -1))[0][0]
                #
                # line_slope = fitted_line.coef_[0][0]
                # line_slope_deg = np.arctan(line_slope) * 180 / np.pi
                # ax.annotate(f'R2: {round(r2, 2)}', xy=[x_to_plot, 1.1*y_to_plot], xycoords='data',
                #             fontsize=font_size, fontweight="bold", rotation=line_slope_deg,
                #             color=color_dictionary[legends[i]])
                if legend_dictionary[legends[i]] == '':
                    legend_dictionary[legends[i]] = f'R2: {round(r2, 2)}'
                else:
                    legend_dictionary[legends[i]] = legend_dictionary[legends[i]] + f', R2: {round(r2, 2)}'


        if legend_dictionary:
            if legend_dictionary[legends[i]] != '':
                patches.append(mpatches.Patch(color=color_dictionary[legends[i]], label=legend_dictionary[legends[i]]))

    if colormap is not None:
        cbar = ax.figure.colorbar(sm)
        cbar.set_label(label=colorbar_label, labelpad=20, fontweight='bold', size=font_size)
        cbar_tick_labels = cbar.ax.get_yticklabels()
        [t.set_fontsize(label_size) for t in cbar_tick_labels]
        [t.set_fontweight('bold') for t in cbar_tick_labels]

    if xticks_values is None:
        xticks_values = ax.get_xticks()
    if yticks_values is None:
        yticks_values = ax.get_yticks()


    if plot_linear_line:
        min_min_value = np.min([np.min(xticks_values), np.min(yticks_values)])
        min_max_value = np.min([np.max(xticks_values), np.max(yticks_values)])
        # min_x = min(xticks_values)
        # max_x = max(xticks_values)
        # min_y = min(yticks_values)
        # max_y = max(yticks_values)

        ax.plot([min_min_value, min_max_value], [min_max_value, min_min_value], color='black', linewidth=5, linestyle='solid', alpha=0.5)

    if plot_y_equal_x_line:
        min_x = min(xticks_values)
        max_x = max(xticks_values)
        min_y = min(yticks_values)
        max_y = max(yticks_values)

        if min_x == min_y:
            if max_x <= max_y:
                ax.plot([min_x, max_x], [min_x, max_x], color='black', linewidth=5, linestyle='solid', alpha=0.5)
            else:
                ax.plot([min_x, max_y], [min_x, max_y], color='black', linewidth=5, linestyle='solid', alpha=0.5)
    # Max, min y
    # min_y, max_y = ax.get_ylim()


    ax.tick_params(which='major', axis='x', direction='out', pad=20)
    ax.tick_params(which='major', axis='y', direction='out', pad=20)
    ax.set_yticks(yticks_values)
    ax.set_xticks(xticks_values)

    # ax.set_xlim([min_x, max_x])
    # ax.set_ylim([min_y, max_y])
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=20, fontsize=font_size)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=20, fontsize=font_size)
    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)
    ax.tick_params(axis='x', pad=15)
    if xlims is not None:
        ax.set_xlim([xlims[0], xlims[1]])
    if ylims is not None:
        ax.set_ylim([ylims[0], ylims[1]])

    if not isinstance(None, type(yticks_labels)):
        ax.set_yticklabels(labels=yticks_labels, fontsize=label_size, fontweight='bold')

    if not isinstance(None, type(xticks_labels)):
        ax.set_xticklabels(labels=xticks_labels, fontsize=label_size, fontweight='bold')

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontweight('bold') for label in labels]
    # plt.locator_params(axis='x', nbins=3)
    # plt.locator_params(axis='y', nbins=4)
    # plt.yticks(rotation=90)
    legend = ax.legend(handles=patches, handlelength=1, frameon=False, loc=legend_loc, ncol=ncol, handletextpad=handletextpad, labelspacing=labelspacing, columnspacing=columnspacing)
    for patch in legend.get_patches():
        patch.set_height(font_size/1.5)
        patch.set_width(font_size/1.5)
        patch.set_x(font_size/10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # if remove_first_y:
    #     ax.yaxis.get_major_ticks()[0].set_visible(False)
    [i.set_linewidth(spine_width) for i in ax.spines.values()]
    ax.xaxis.set_tick_params(width=tick_width, length=tick_length, direction='out')
    ax.yaxis.set_tick_params(width=tick_width, length=tick_length, direction='out')
    plt.setp(ax.get_legend().get_texts(), fontsize=font_size, fontweight='bold')
    plt.title(figure_title, fontsize=font_size, fontweight='bold')

    if plot_horizontal_line:
        for i in range(len(horizontal_line_y)):
            plt.axhline(y=horizontal_line_y[i], color=horizontal_line_color[i], linewidth=horizontal_line_lw[i], linestyle=horizontal_line_ls[i])

    if plot_vertical_line:
        for i in range(len(vertical_line_x)):
            plt.axvline(x=vertical_line_x[i], color=vertical_line_color[i], linewidth=vertical_line_lw[i], linestyle=vertical_line_ls[i])

    if use_log_scale:
        plt.yscale('log')
        plt.xscale('log')
        ax.xaxis.set_tick_params(which='minor', width=tick_width, length=tick_length, direction='out')
        ax.yaxis.set_tick_params(which='minor', width=tick_width, length=tick_length, direction='out')

    # If there is text to place
    if place_text is not None:
        if type(place_text) == list:
            for i in range(len(place_text)):
                ax.annotate(place_text[i], xy=place_text_loc[i], xycoords='axes fraction',
                            fontsize=place_text_font_size[i], fontweight=f'{"bold" if make_text_bold else "normal"}', rotation=f'{"vertical" if rotate_place_text[i] else "horizontal"}',
                            color=place_text_color[i])

        elif place_text:
            ax.annotate(place_text, xy=place_text_loc, xycoords='axes fraction',
                        fontsize=place_text_font_size, fontweight=f'{"bold" if make_text_bold else "normal"}', rotation=f'{"vertical" if rotate_place_text else "horizontal"}',
                        color=place_text_color)

    fig.tight_layout(pad=pad)

    if return_fig_instead_of_save:
        return fig
    else:
        # Save the figure
        plt.savefig(data_path_out + filename + '.png')
        # Save to svg as well
        if save_svg:
            svg_path = data_path_out + r'\SVG'
            if not os.path.exists(svg_path):
                os.makedirs(svg_path)
            plt.savefig(svg_path + filename + '.svg', format='svg')

    plt.close()


'''
This is a parameterized function for barplots. You can ignore the arguments with default values if you want and only pass the ones with no default values. 
Make sure you read the description of each argument to see what it does.
1) data_dictionary: A dictionary that contains the height of each bar in the form {'Bar1': height1, 'Bar2': height2, etc}
2) xlabel: The label on the xaxis
3) ylabel: The label on the yaxis
4) xticks: A numpy.array or list of values to show as ticks in the xaxis.
5) xticks_labels: A numpy.array or list of labels to overwrite the ticks in the xaxis.
6) figure_title: The title of the figure
7) color_dict_legends: A dictionary that contains the color of each legend in the form {'Legend1': 'color1', 'Legend2': 'color2', etc} (This way of implementation is because you might have multiple bars that correspond to the same distribution)
8) color_dict_bars: A dictionary that contains the color of each bar in the form {'Bar1': 'color1', 'Bar2': 'color2', etc}
9) bins: A dictionary that contains the position of each bar in the x axis in the form {'Bar1': xvalue1, 'Bar2': xvalue2, etc}
10) bar_width: A dictionary that contains the width of each bar in the form {'Bar1': width1, 'Bar2': width2, etc}
11) data_path_out: The absolute path of the folder at which you want to store the figure (without the name of the file). That path must exist so you can create it before the call to this function if it 
doesn't. You could also add that part inside this function if you want.
12) filename: The name of the file where the plot will be stored in the form '\Somename' (you don't need to add a file extension)
13) yticks: A numpy.array or list of values to show as ticks in the yaxis.
13) yticks_labels: A numpy.array or list of labels to overwrite the ticks in the yaxis.
14) edgecolor: A dictionary that contains the color of the contour of each bar in the form {'Bar1': 'color1', 'Bar2': 'color2', etc}
15) rotate_xticks: True/False flag on whether you want to have the xticks appear vertically or horizontally
16) legends: A list containing the distribution names you want to show in the legend in the form ['Dist1', 'Dist2', etc]
17) legend_dictionary: A dictionary that contains the legend name of each distribution you want to plot in the form {'Dist1': 'Legend1', 'Dist2': 'Legend2', etc}
18) align: 'edge'/'center' string that determines whether the bars are plotted with their bin in their left edge or in their
 center
19) sem_data_dictionary: A dictionary that contains the standard error of the mean of each bar in the form {'Bar1': sem1, 'Bar2': sem2, etc}
20) remove_first_y: True/False flag which removes the first tick on the y axis (this is used in cases where the x and the y axis have the same starting tick and you don't want them to be overlapping)
21) xlims: A list of min and max values to be shown for the xaxis [min_value, max_value]
22) ylims: A list of min and max values to be shown for the yaxis [min_value, max_value]
23) figure_size: A tuple for the size of the image (width, height)
24) legend_loc: The location of the legend
25) ncol: The number of columns used to show the legend
26) font_size: The font size of the xlabel, ylabel, title, and the legend. This param is also used in order to adjust the patches of the legend. Could use a better practice but w/e :P
27) label_size: The font size of the xticks, yticks
28) alpha: The alpha / opacity value (in [0, 1]) for all bars
29) capsize: The width of the cap in the error bar
30) elinewidth: The linewidth of the error bar line
31) capthick: The thickness of the cap in the error bar
32) plot_minor_ticks: True/False flag on whether you want to show minor ticks as well
33) spine_width: The width of the axis lines
34) yticks_font_size: The font size of the yticks (in case you want it to be different than label_size)
35) xticks_font_size: The font size of the xticks (in case you want it to be different than label_size)
36) set_ylims_one_extra_tick: True/False flag that expands the yaxis limits for one extra tick.
37) pad: The padding used for the image with tight layout
38) save_svg: Whether you want to save the image in an '.svg' format. The image is automatically saved as a '.png' either way.
39) place_text: A list containing strings of texts you want to annotate in the image (e.g. ['text1', 'text2', etc]
40) place_text_loc: A list containing a [x, y] list for the location of each annotated text (e.g. [[x1, y1], [x2, y2], etc])
41) text_font_size: A list containing the font size of each annotated text (I usually set this either to font_size or label_size, e.g. [font_size, font_size, etc])
42) make_text_bold: True/False flag on whether you want the annotated text to be bold or not (you could change this to apply to each individual text)
43) place_text_color: A list containing the color of each annotated text (['color1', 'color2', etc]) 
44) use_sns: True/False flag on whether to use the seaborn library for setting up some params
45) disable_xtick_edges: True/False flag on whether you want to remove the tick lines from the xaxis
46) bbox_to_anchor: The location of the bounding box corner to use to frame the legend
47) xalbel_pad: The padding used between the xlabel and the xaxis
48) make_legend_patch_visible: True/False flag on whether you want to show the patch of each legend
49) legend_linecolor: True/False flag on whether you want the legend text to have the same color as its corresponding distribution
50) handletextpad: The padding between the legend text and its patch
51) tick_width: The width of the ticks in the x and y axis.
52) tick_length: The length of the ticks in the x and y axis
53) return_fig_instead_of_save: True/False flag, which if False the plot is stored in an image, whereas if it's True the image is returned as an object.
'''
# Create a parametrized function for barplots
def plot_parameterized_barplot(data_dictionary, xlabel, ylabel, xticks, xticks_labels, figure_title,
                               color_dict_legends, color_dict_bars, bins, bar_width, data_path_out, filename, yticks=None, yticks_labels=None,
                               edgecolor=None, rotate_xticks=False, legends=None, legend_dictionary=None, align='edge',
                               sem_data_dictionary=None, remove_first_y=False, xlims=None, ylims=None,
                               figure_size=(25, 25), legend_loc='upper right', ncol=2, font_size=90, label_size=75, alpha=1,
                               capsize=20, elinewidth=5, capthick=3, plot_minor_ticks=False, spine_width=7,
                               yticks_font_size=None, xticks_font_size=None, set_ylims_one_extra_tick=False, pad=0.25,
                               save_svg=True, place_text=None, place_text_loc=None, text_font_size=None, make_text_bold=False,
                               place_text_color=None, use_sns=True, disable_xtick_edges=False, bbox_to_anchor=None, xlabel_pad=20,
                               make_legend_patch_visible=False, legend_linecolor=True, handletextpad=1, tick_width=4, tick_length=16,
                               return_fig_instead_of_save=False):

    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()

    fig, ax = plt.subplots(1)
    fig.set_size_inches(figure_size)
    if use_sns:
        sns.set(font_scale=5.0, style='ticks')
    # plt.rcParams["font.weight"] = "bold"
    # Get legends aka data dictionary keys
    bars = [key for key in data_dictionary.keys()]

    # Place alpha at colors
    # color_dict_bars = {key: tuple(list(hex2rgb(color_dict_bars[key], normalise=True)) + [alpha]) for key in bars}

    if edgecolor is None:
        edgecolor = {key: None for key in bars}
        edgecolor_lw = 0
    elif edgecolor == 'Same':
        # Place alpha at edge color
        # edgecolor = {key: tuple(list(hex2rgb(color_dict_bars[key], normalise=True)) + [alpha]) for key in bars}
        edgecolor = {key: color_dict_bars[key] for key in bars}
        edgecolor_lw = 0.5
    else:
        edgecolor = {key: edgecolor for key in bars}
        edgecolor_lw = 0.5

    for i in range(len(bars)):
        if sem_data_dictionary is not None:
            plt.bar(bins[bars[i]], data_dictionary[bars[i]], width=bar_width[bars[i]], align=align,
                    facecolor=color_dict_bars[bars[i]], yerr=sem_data_dictionary[bars[i]], ecolor='black', capsize=capsize,
                    error_kw={'elinewidth': elinewidth, 'capthick': capthick}, edgecolor=edgecolor[bars[i]], lw=edgecolor_lw, alpha=alpha)


        else:
            plt.bar(bins[bars[i]], data_dictionary[bars[i]], width=bar_width[bars[i]], align=align,
                    facecolor=color_dict_bars[bars[i]], edgecolor=edgecolor[bars[i]], alpha=alpha, lw=edgecolor_lw)


    ax.tick_params(which='major', axis='x', direction='out', pad=15, labelsize=label_size)
    ax.tick_params(which='major', axis='y', direction='out', pad=20, labelsize=label_size)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xticks(xticks)
    if xticks_font_size is None:
        xticks_font_size = label_size
    ax.set_xticklabels(xticks_labels)
    if rotate_xticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=xlabel_pad, fontsize=font_size)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=20, fontsize=font_size)
    if yticks_font_size is None:
        yticks_font_size = label_size
    if yticks_labels is not None:
        ax.set_yticklabels(labels=yticks_labels, fontsize=yticks_font_size, fontweight='bold')

    if xlims:
        ax.set_xlim([xlims[0], xlims[1]])
    if ylims:
        ax.set_ylim([ylims[0], ylims[1]])
    if set_ylims_one_extra_tick:
        yticks_default = ax.get_yticks()
        y_tick_step = yticks_default[1] - yticks_default[0]
        ax.set_ylim([yticks_default[0], yticks_default[-1] + y_tick_step])

    labels_x = ax.get_xticklabels()
    labels_y = ax.get_yticklabels()
    [label.set_fontweight('bold') for label in labels_x]
    [label.set_fontweight('bold') for label in labels_y]
    [label.set_fontsize(xticks_font_size) for label in labels_x]
    [label.set_fontsize(yticks_font_size) for label in labels_y]

    if legends:
        patches = []
        for i in range(0, len(legends)):
            if legend_dictionary is not None:
                patches.append(mpatches.Patch(color=color_dict_legends[legends[i]], label=legend_dictionary[legends[i]], visible=make_legend_patch_visible))
            else:
                patches.append(mpatches.Patch(color=color_dict_legends[legends[i]], label=legends[i], visible=make_legend_patch_visible))

        if bbox_to_anchor is not None:
            ax.legend(handles=patches, handlelength=1, frameon=False, loc=legend_loc, labelcolor=f'{"linecolor" if legend_linecolor else "None"}', ncol=ncol, bbox_to_anchor=bbox_to_anchor)
        else:
            ax.legend(handles=patches, handlelength=1, frameon=False, loc=legend_loc, labelcolor=f'{"linecolor" if legend_linecolor else "None"}', ncol=ncol)

        if make_legend_patch_visible:
            legend = ax.legend(handles=patches, handlelength=1, frameon=False, loc=legend_loc, ncol=ncol, handletextpad=handletextpad)
            for patch in legend.get_patches():
                patch.set_height(font_size - 30)
                patch.set_width(font_size - 30)
                patch.set_x(-font_size / 4)

        plt.setp(ax.get_legend().get_texts(), fontsize=font_size, fontweight='bold')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if remove_first_y:
        ax.yaxis.get_major_ticks()[0].set_visible(False)
    [i.set_linewidth(spine_width) for i in ax.spines.values()]
    ax.xaxis.set_tick_params(which='major', width=tick_width, length=tick_length, direction='out')
    ax.yaxis.set_tick_params(which='major', width=tick_width, length=tick_length, direction='out')
    if plot_minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_tick_params(which='minor', width=tick_width/2, length=tick_length/2, direction='out')
        ax.yaxis.set_tick_params(which='minor', width=tick_width/2, length=tick_length/2, direction='out')
    plt.title(figure_title, fontsize=font_size, fontweight='bold')

    if disable_xtick_edges:
        ax.xaxis.set_tick_params(which='major', width=0, length=0, direction='out')

    if place_text is not None:
        if type(place_text) == list:
            for i in range(len(place_text)):
                ax.annotate(place_text[i], xy=place_text_loc[i], xycoords='axes fraction',
                            fontsize=text_font_size[i], fontweight=f'{"bold" if make_text_bold[i] else "normal"}',
                            color=place_text_color[i])

        elif place_text:
            ax.annotate(place_text, xy=place_text_loc, xycoords='axes fraction',
                        fontsize=text_font_size, fontweight=f'{"bold" if make_text_bold else "normal"}',
                        color=place_text_color)

    fig.tight_layout(pad=pad)

    if return_fig_instead_of_save:
        return fig
    else:
        # Save figure
        plt.savefig(data_path_out + filename + f'.png', format='png')
        if save_svg:
            data_path_out2 = data_path_out + r'\SVG'
            if not os.path.exists(data_path_out2):
                os.makedirs(data_path_out2)
            plt.savefig(data_path_out2 + filename + f'.svg', format='svg')
    plt.close()



'''
This is a parameterized function for barplots with two y axis. You can ignore the arguments with default values if you want and only pass the ones with no default values. 
Make sure you read the description of each argument to see what it does.
1) data_dictionary: A dictionary that contains the height of each bar in the form {'Bar1': height1, 'Bar2': height2, etc}
2) xlabel: The label on the xaxis
3) ylabel_left: The label on the left yaxis
4) ylabel_right: The label on the right yaxis
5) xticks: A numpy.array or list of values to show as ticks in the xaxis.
6) xticks_labels: A numpy.array or list of labels to overwrite the ticks in the xaxis.
7) figure_title: The title of the figure
8) color_dict_legends: A dictionary that contains the color of each legend in the form {'Legend1': 'color1', 'Legend2': 'color2', etc} (This way of implementation is because you might have multiple bars that correspond to the same distribution)
9) color_dict_bars: A dictionary that contains the color of each bar in the form {'Bar1': 'color1', 'Bar2': 'color2', etc}
10) bins: A dictionary that contains the position of each bar in the x axis in the form {'Bar1': xvalue1, 'Bar2': xvalue2, etc}
11) bar_width: A dictionary that contains the width of each bar in the form {'Bar1': width1, 'Bar2': width2, etc}
12) data_path_out: The absolute path of the folder at which you want to store the figure (without the name of the file). That path must exist so you can create it before the call to this function if it 
doesn't. You could also add that part inside this function if you want.
13) filename: The name of the file where the plot will be stored in the form '\Somename' (you don't need to add a file extension)
14) yticks_left: A numpy.array or list of values to show as ticks in the left yaxis.
15) yticks_right: A numpy.array or list of values to show as ticks in the right yaxis.
16) edgecolor: A dictionary that contains the color of the contour of each bar in the form {'Bar1': 'color1', 'Bar2': 'color2', etc}
17) rotate_xticks: True/False flag on whether you want to have the xticks appear vertically or horizontally
18) legends: A list containing the distribution names you want to show in the legend in the form ['Dist1', 'Dist2', etc]
19) legend_dictionary: A dictionary that contains the legend name of each distribution you want to plot in the form {'Dist1': 'Legend1', 'Dist2': 'Legend2', etc}
20) align: 'edge'/'center' string that determines whether the bars are plotted with their bin in their left edge or in their
 center
21) sem_data_dictionary: A dictionary that contains the standard error of the mean of each bar in the form {'Bar1': sem1, 'Bar2': sem2, etc}
22) remove_first_y_left: True/False flag which removes the first tick on the left y axis (this is used in cases where the x and the y axis have the same starting tick and you don't want them to be overlapping)
23) remove_first_y_left: True/False flag which removes the first tick on the right y axis (this is used in cases where the x and the y axis have the same starting tick and you don't want them to be overlapping)
24) xlims: A list of min and max values to be shown for the xaxis [min_value, max_value]
25) ylims_left: A list of min and max values to be shown for the left yaxis [min_value, max_value]
26) ylims_right: A list of min and max values to be shown for the right yaxis [min_value, max_value]
27) figure_size: A tuple for the size of the image (width, height)
28) legend_loc: The location of the legend
29) ncol: The number of columns used to show the legend
30) font_size: The font size of the xlabel, ylabel, title, and the legend. This param is also used in order to adjust the patches of the legend. Could use a better practice but w/e :P
31) label_size: The font size of the xticks, yticks
32) alpha: The alpha / opacity value (in [0, 1]) for all bars
33) capsize: The width of the cap in the error bar
34) elinewidth: The linewidth of the error bar line
35) capthick: The thickness of the cap in the error bar
36) yticks_labels_left: A numpy.array or list of labels to overwrite the ticks in the left yaxis.
37) yticks_labels_right: A numpy.array or list of labels to overwrite the ticks in the right yaxis.
38) plot_minor_ticks: True/False flag on whether you want to show minor ticks as well
39) spine_width: The width of the axis lines
40) yticks_font_size: The font size of the yticks (in case you want it to be different than label_size)
41) xticks_font_size: The font size of the xticks (in case you want it to be different than label_size)
42) set_ylims_one_extra_tick: True/False flag that expands the yaxis limits for one extra tick.
43) pad: The padding used for the image with tight layout
44) save_svg: Whether you want to save the image in an '.svg' format. The image is automatically saved as a '.png' either way.
45) place_text: A list containing strings of texts you want to annotate in the image (e.g. ['text1', 'text2', etc]
46) place_text_loc: A list containing a [x, y] list for the location of each annotated text (e.g. [[x1, y1], [x2, y2], etc])
47) text_font_size: A list containing the font size of each annotated text (I usually set this either to font_size or label_size, e.g. [font_size, font_size, etc])
48) make_text_bold: True/False flag on whether you want the annotated text to be bold or not (you could change this to apply to each individual text)
49) place_text_color: A list containing the color of each annotated text (['color1', 'color2', etc]) 
50) use_sns: True/False flag on whether to use the seaborn library for setting up some params
51) disable_xtick_edges: True/False flag on whether you want to remove the tick lines from the xaxis
52) bbox_to_anchor: The location of the bounding box corner to use to frame the legend
53) xalbel_pad: The padding used between the xlabel and the xaxis
54) legend_axis_assignment: A dictionary that maps each legend to either the 'left' or the 'right' y axis in the form {'Legend1': 'left', 'Legend2': 'right', etc}
55) return_fig_instead_of_save: True/False flag, which if False the plot is stored in an image, whereas if it's True the image is returned as an object. (I should review how this works)
'''
# Create a parametrized function for barplots
def plot_parameterized_barplot_two_y_axis(data_dictionary, xlabel, ylabel_left, ylabel_right, xticks, xtick_labels, figure_title,
                               color_dict_legends, color_dict_bars, bins, bar_width, data_path_out, filename, yticks_left=None, yticks_right=None,
                               edgecolor=None, rotate_xticks=False, legends=None, legend_dictionary=None, align='edge',
                               sem_data_dictionary=None, remove_first_y_left=False, remove_first_y_right=False, xlims=None, ylims_left=None, ylims_right=None,
                               figure_size=(25, 25), legend_loc='upper right', ncol=2, font_size=90, label_size=75, alpha=1,
                               capsize=20, elinewidth=5, capthick=3, yticks_labels_left=None, yticks_labels_right=None, plot_minor_ticks=False, spine_width=7,
                               yticks_font_size=None, xticks_font_size=None, set_ylims_one_extra_tick=False, pad=0.25,
                               save_svg=True, place_text=None, place_text_loc=None, text_font_size=None, make_text_bold=False,
                               place_text_color=None, use_sns=True, disable_xtick_edges=False, bbox_to_anchor=None, xlabel_pad=20,
                               legend_axis_assignment=None, return_fig_instead_of_save=False):

    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()

    fig, ax1 = plt.subplots(1)
    ax2 = ax1.twinx()


    for key in legends:
        if legend_axis_assignment[key] == 'left':
            legend_axis_assignment[key] = ax1
        elif legend_axis_assignment[key] == 'right':
            legend_axis_assignment[key] = ax2
        else:
            print('Wrong axis given: Acceptable values are "left" and "right".')

    fig.set_size_inches(figure_size)
    if use_sns:
        sns.set(font_scale=5.0, style='ticks')

    # Get all bar keys
    bars = [key for key in data_dictionary.keys()]

    if edgecolor is None:
        edgecolor = {key: None for key in bars}
        edgecolor_lw = 0
    elif edgecolor == 'Same':
        # Place alpha at edge color
        # edgecolor = {key: tuple(list(hex2rgb(color_dict_bars[key], normalise=True)) + [alpha]) for key in bars}
        edgecolor = {key: color_dict_bars[key] for key in bars}
        edgecolor_lw = 0.5
    else:
        edgecolor = {key: edgecolor for key in bars}
        edgecolor_lw = 0.5

    for i in range(len(bars)):
        if sem_data_dictionary is not None:
            for key in legends:
                if key in bars[i]:
                    legend_axis_assignment[key].bar(bins[bars[i]], data_dictionary[bars[i]], width=bar_width[bars[i]], align=align,
                            facecolor=color_dict_bars[bars[i]], yerr=sem_data_dictionary[bars[i]], ecolor='black', capsize=capsize,
                            error_kw={'elinewidth': elinewidth, 'capthick': capthick}, edgecolor=edgecolor[bars[i]], lw=edgecolor_lw, alpha=alpha)


        else:
            for key in legends:
                if key in bars[i]:
                    legend_axis_assignment[key].bar(bins[bars[i]], data_dictionary[bars[i]], width=bar_width[bars[i]], align=align,
                            facecolor=color_dict_bars[bars[i]], edgecolor=edgecolor[bars[i]], alpha=alpha, lw=edgecolor_lw)


    ax1.tick_params(which='major', axis='x', direction='out', pad=15, labelsize=label_size)
    ax1.tick_params(which='major', axis='y', direction='out', pad=20, labelsize=label_size)
    ax2.tick_params(which='major', axis='y', direction='out', pad=20, labelsize=label_size)
    if yticks_left is not None:
        ax1.set_yticks(yticks_left)
    if yticks_right is not None:
        ax2.set_yticks(yticks_right)
    ax1.set_xticks(xticks)
    if xticks_font_size is None:
        xticks_font_size = label_size
    ax1.set_xticklabels(xtick_labels)
    if rotate_xticks:
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax1.set_xlabel(xlabel, fontweight='bold', labelpad=xlabel_pad, fontsize=font_size)
    ax1.set_ylabel(ylabel_left, fontweight='bold', labelpad=20, fontsize=font_size)
    ax2.set_ylabel(ylabel_right, fontweight='bold', labelpad=20, fontsize=font_size)
    if yticks_font_size is None:
        yticks_font_size = label_size
    if yticks_labels_left is not None:
        ax1.set_yticklabels(labels=yticks_labels_left, fontsize=yticks_font_size, fontweight='bold')
    if yticks_labels_right is not None:
        ax2.set_yticklabels(labels=yticks_labels_right, fontsize=yticks_font_size, fontweight='bold')
    if xlims:
        ax1.set_xlim([xlims[0], xlims[1]])
    if ylims_left:
        ax1.set_ylim([ylims_left[0], ylims_left[1]])
    if ylims_right:
        ax2.set_ylim([ylims_right[0], ylims_right[1]])
    if set_ylims_one_extra_tick:
        yticks_default = ax1.get_yticks()
        y_tick_step = yticks_default[1] - yticks_default[0]
        ax1.set_ylim([yticks_default[0], yticks_default[-1] + y_tick_step])

    labels_x = ax1.get_xticklabels()
    labels_y = ax1.get_yticklabels() + ax2.get_yticklabels()
    [label.set_fontweight('bold') for label in labels_x]
    [label.set_fontweight('bold') for label in labels_y]
    [label.set_fontsize(xticks_font_size) for label in labels_x]
    [label.set_fontsize(yticks_font_size) for label in labels_y]

    if legends:
        patches = []
        for i in range(0, len(legends)):
            if legend_dictionary is not None:
                patches.append(mpatches.Patch(color=color_dict_legends[legends[i]], label=legend_dictionary[legends[i]], visible=False))
            else:
                patches.append(mpatches.Patch(color=color_dict_legends[legends[i]], label=legends[i], visible=False))

        if bbox_to_anchor is not None:
            ax1.legend(handles=patches, handlelength=1, frameon=False, loc=legend_loc, labelcolor='linecolor', ncol=ncol, bbox_to_anchor=bbox_to_anchor)
        else:
            ax1.legend(handles=patches, handlelength=1, frameon=False, loc=legend_loc, labelcolor='linecolor', ncol=ncol)

        plt.setp(ax1.get_legend().get_texts(), fontsize=font_size, fontweight='bold')

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    if remove_first_y_left:
        ax1.yaxis.get_major_ticks()[0].set_visible(False)
    if remove_first_y_right:
        ax2.yaxis.get_major_ticks()[0].set_visible(False)
    [i.set_linewidth(spine_width) for i in ax1.spines.values()]
    [i.set_linewidth(spine_width) for i in ax2.spines.values()]

    ax1.xaxis.set_tick_params(which='major', width=5, length=20, direction='out')
    ax1.yaxis.set_tick_params(which='major', width=5, length=20, direction='out')
    ax2.yaxis.set_tick_params(which='major', width=5, length=20, direction='out')

    if plot_minor_ticks:
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.xaxis.set_tick_params(which='minor', width=3, length=12, direction='out')
        ax1.yaxis.set_tick_params(which='minor', width=3, length=12, direction='out')
        ax2.yaxis.set_tick_params(which='minor', width=3, length=12, direction='out')

    plt.title(figure_title, fontsize=font_size, fontweight='bold')

    if disable_xtick_edges:
        ax1.xaxis.set_tick_params(which='major', width=0, length=0, direction='out')

    if place_text is not None:
        if type(place_text) == list:
            for i in range(len(place_text)):
                ax1.annotate(place_text[i], xy=place_text_loc[i], xycoords='axes fraction',
                            fontsize=text_font_size[i], fontweight=f'{"bold" if make_text_bold else "normal"}',
                            color=f'{place_text_color[i] if place_text_color[i] is not None else "black"}')

        elif place_text:
            ax1.annotate(place_text, xy=place_text_loc, xycoords='axes fraction',
                        fontsize=text_font_size, fontweight=f'{"bold" if make_text_bold else "normal"}',
                        color=f'{place_text_color if place_text_color is not None else "black"}')

    fig.tight_layout(pad=pad)

    # Save figure
    if return_fig_instead_of_save:
        return fig
    else:
        plt.savefig(data_path_out + filename + f'.png', format='png')
        if save_svg:
            data_path_out2 = data_path_out + r'\SVG'
            if not os.path.exists(data_path_out2):
                os.makedirs(data_path_out2)
            plt.savefig(data_path_out2 + filename + f'.svg', format='svg')
    plt.close()


'''
This is a parameterized function for a heatmap. You can ignore the arguments with default values if you want and only pass the ones with no default values. 
Make sure you read the description of each argument to see what it does.
1) data: A dataframe containing the data
2) figure_title: The title of the figure
3) xlabel: The label on the xaxis
4) ylabel: The label on the yaxis
5) data_path_out: The absolute path of the folder at which you want to store the figure (without the name of the file). That path must exist so you can create it before the call to this function if it 
doesn't. You could also add that part inside this function if you want.
6) filename: The name of the file where the plot will be stored in the form '\Somename' (you don't need to add a file extension)
7) font_size: The font size of the xlabel, ylabel, title, and the legend. This param is also used in order to adjust the patches of the legend. Could use a better practice but w/e :P
8) label_size: The font size of the xticks, yticks
9) figure_size: A tuple for the size of the image (width, height)
10) plot_colorbar: True/False flag on whether you want to plot a colorbar
11) vmin: The minimum data value to use for the coloring range
12) vmax: The maximum data value to use for the coloring range
13) rotate_ticks: True/False flag on whether you want the ticks on each axis to rotated or not
14) cmap: The color palette you want to use as a colormap 
15) return_fig_instead_of_save: True/False flag, which if False the plot is stored in an image, whereas if it's True the image is returned as an object. (I should review how this works)
'''
def parameterized_heatmap(data, title, xlabel, ylabel, data_path_out, filename,
                          font_size=45, label_size=45/1.125, figure_size=(20, 20), plot_colorbar=False, vmin=0, vmax=1,
                          rotate_ticks=False, cmap='rocket', return_fig_instead_of_save=False):
    # Make the heatmap plot and store it
    plt.ioff()
    fig, ax = plt.subplots(1)
    fig.set_size_inches(figure_size)
    sns.heatmap(data, annot=True, ax=ax, annot_kws={'fontsize': label_size}, cbar_kws={'pad': 0.01}, cbar=plot_colorbar, vmin=vmin, vmax=vmax, cmap=cmap)
    if plot_colorbar:
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=font_size)
    ax.set_xlabel(xlabel, fontweight='bold', labelpad=40, fontsize=font_size)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=40, fontsize=font_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontsize(label_size) for label in labels]
    [label.set_fontweight('bold') for label in labels]
    if rotate_ticks:
        [label.set_rotation('vertical') for label in ax.get_xticklabels()]
        [label.set_rotation('horizontal') for label in ax.get_yticklabels()]

    plt.title(title, fontsize=font_size, fontweight='bold')
    fig.tight_layout(pad=0.5)
    if return_fig_instead_of_save:
        return fig
    else:
        plt.savefig(data_path_out + filename)

    plt.close()


'''
A function that receives as inputs multiple figures and draws them together as subplots
- All figures must have the same height but they can have different width
- Add figures in list
- figures_list = [fig1, fig2, fig3, fig4, fig5]
- Define rows for each figure. The rows must always be presented in a incremental order (e.g. format [0, 1, 2, 1, 0] is invalid)
- figure_rows = [0, 0, 1, 1, 1]
- The column that each figure belongs to is from left to right based on how they were given in the list
IMPORTANT NOTES:
1) When merging individual figures and then placing text, all individual figures must have the same pad, say pad_start, the
merged figure must have a pad of pad = 2*pad_start and the font size of the placed text must be font_size * single_figure_height / (merged_figure_single_row_height + 1 - pad)
2) When merging individual figures and an already merged figure from this function that belong in different rows, all individual figures must have the same pad, say pad_start, the new
merged figure must have a pad of pad = pad_start and the font size of the new individual figures must be * first_merged_figure_individual_plot_height / (merged_figure_single_row_height +1 - pad)
'''
def draw_multiple_figures_as_subplots_at_single_figure(figures_list, figure_rows, data_path_out, filename, grid_hspace=0, return_fig_instead_of_save=False,
                                                       place_text=None, place_text_loc=None, text_font_size=None, make_text_bold=False,
                                                       place_text_color=None, rotate_place_text=None, pad=0.5):

    # Total figures
    total_figs = len(figures_list)
    # Total rows
    total_rows = len(np.unique(figure_rows))

    # Create a dict that maps the rows to the figures
    row_to_fig_dict = {f'Row{i}': [f'F{j}' for j in np.where(figure_rows == i)[0]] for i in np.unique(figure_rows)}

    # Get the number of figures each row contains
    row_to_fig_num_dict = {row: len(row_to_fig_dict[row]) for row in row_to_fig_dict.keys()}

    # Create a dict that maps the figures to the rows
    figure_to_row_dict = {f'F{i}': figure_rows[i] for i in range(total_figs)}

    # Get the sizes of each figure (as integers)
    figure_widths = {f'F{i}': int(figures_list[i].get_size_inches()[0]) for i in range(total_figs)}
    figure_heights = {f'F{i}': int(figures_list[i].get_size_inches()[1]) for i in range(total_figs)}

    # Make sure that all figures have the same height
    assert len(set(figure_heights.values()))

    # Get the width of each row
    row_width_dict = {f'Row{i}': np.sum([figure_widths[fig] for fig in row_to_fig_dict[f'Row{i}']]) for i in range(total_rows)}

    # Get the max width across rows
    max_row_width = np.max([row_width_dict[row] for row in row_width_dict.keys()])
    # Define the number of columns in the grid, which will be the same as the image width
    # grid_cols = max_row_width + np.max([row_to_fig_num_dict[row] for row in row_to_fig_num_dict.keys()]) + 1
    grid_cols = max_row_width
    image_width = grid_cols
    # Define the height of the image
    image_height = figure_heights['F0'] * total_rows + total_rows + 1
    # image_height = figure_heights['F0'] * total_rows

    # Get the canvas for each figure
    figure_canvas = {f'F{i}': figures_list[i].canvas for i in range(total_figs)}
    # Draw the canvas for each figure
    [figure_canvas[f'F{i}'].draw() for i in range(total_figs)]
    # Get the array canvas for each figure
    array_canvas = {f'F{i}': np.array(figure_canvas[f'F{i}'].buffer_rgba()) for i in range(total_figs)}

    # Define the grid
    grid = gridspec.GridSpec(nrows=total_rows, ncols=grid_cols, width_ratios=[1] * grid_cols, height_ratios=[1] * total_rows)
    # grid.update(hspace=grid_hspace, wspace=grid_wspace)

    # Create the figure
    fig, ax = plt.subplots(figsize=(image_width, image_height), zorder=1)
    ax.axis('off')
    # Create a dict for the axis that will be created for each plot
    axis_dict = {}
    # Add a subplot for each figure starting from the first row
    current_row = 0
    for i in range(total_figs):
        fig_width = figure_widths[f'F{i}']
        # fig_height = figure_heights[f'F{i}']
        fig_row = figure_to_row_dict[f'F{i}']
        if fig_row != current_row or i == 0:
            current_row = fig_row
            current_col = 0
            # Get the total width of the figures of the current row
            current_row_figure_width = row_width_dict[f'Row{current_row}']
            if row_to_fig_num_dict[f'Row{current_row}'] > 1:
                # Define the width space
                wspace_cols = int((grid_cols - current_row_figure_width) / (row_to_fig_num_dict[f'Row{current_row}'] - 1))
                wspace = wspace_cols/grid_cols
                grid.update(hspace=grid_hspace, wspace=wspace)
            else:
                wspace_cols = 0

        print(grid[current_row, current_col:(current_col + fig_width)])
        # Add subplot and axis to dict
        axis_dict[f'F{i}'] = fig.add_subplot(grid[current_row, current_col:(current_col + fig_width)], zorder=-1)
        # Render the data
        axis_dict[f'F{i}'].matshow(array_canvas[f'F{i}'], zorder=-1)
        axis_dict[f'F{i}'].axis('off')

        # Update current column
        current_col += fig_width + wspace_cols

    if place_text is not None:
        # Set the zorder of the annotation to be higher than the default value of 0
        for i in range(len(place_text)):
            # ax.annotate(place_text[i], xy=place_text_loc[i], xycoords='axes fraction',
            #             fontsize=text_font_size[i], fontweight=f'{"bold" if make_text_bold else "normal"}',
            #             color=f'{place_text_color[i] if place_text_color[i] is not None else "black"}', rotation=f'{"vertical" if rotate_place_text[i] else "horizontal"}', zorder=10)
            ax.annotate(place_text[i], xy=place_text_loc[i], xycoords='axes fraction',
                        fontsize=text_font_size[i], fontweight=f'{"bold" if make_text_bold else "normal"}',
                        color=place_text_color[i], rotation=f'{"vertical" if rotate_place_text[i] else "horizontal"}', zorder=10)
    fig.tight_layout(pad=pad)

    if return_fig_instead_of_save:
        return fig

    else:
        if not os.path.exists(data_path_out):
            os.makedirs(data_path_out)
        plt.savefig(data_path_out + filename + '.png', format='png')

        data_path_out_svg = data_path_out + r'\SVG'
        if not os.path.exists(data_path_out_svg):
            os.makedirs(data_path_out_svg)

        plt.savefig(data_path_out_svg + filename + '.svg', format='svg')

    plt.close()


'''
Creates an empty figure with figure_size (width, height)
'''
def create_empty_figure(figure_size):
    fig = plt.figure(figsize=figure_size, zorder=-1)

    return fig