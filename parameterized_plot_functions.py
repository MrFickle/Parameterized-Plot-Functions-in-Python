"""
This script contains functions that are used only for plotting data.
"""

# Modules
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
import os
from matplotlib.ticker import (AutoMinorLocator)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import diptest
import matplotlib.gridspec as gridspec

plt.rcParams["font.family"] = "Arial"


def add_color_bar(fig, ax, colormap, label, loc, ticks, tick_labels, orientation, font_size, label_size):
    """Add a colorbar to the plot with customized settings."""
    sm = plt.cm.ScalarMappable(cmap=colormap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation=orientation, location=loc, label=label, ticks=ticks)
    cbar.set_label(label, fontweight='bold', fontsize=font_size)
    cbar.ax.set_yticklabels(tick_labels, fontweight='bold', fontsize=label_size)


def create_line_legend(ax, legend_dictionary, color_dictionary, lw_dict, ls_dict, m_dict, ms_dict,
                       show_lines, legend_loc, ncol, handletextpad, legend_line_length, bbox_to_anchor, font_size,
                       legend_lw_multiplier, use_line_color_for_legends):
    """Create a custom legend based on the plot parameters."""
    handles = [mlines.Line2D([], [], color=color_dictionary[key], marker=m_dict.get(key, None),
                             markersize=ms_dict.get(key, 5), linestyle=ls_dict.get(key, '-'),
                             linewidth=lw_dict.get(key, 2) * legend_lw_multiplier, label=legend_name,
                             visible=show_lines) for key, legend_name in legend_dictionary.items()]
    ax.legend(handles=handles, loc=legend_loc, ncol=ncol, frameon=False, prop={'size': font_size, 'weight': 'bold'},
              handletextpad=handletextpad, bbox_to_anchor=bbox_to_anchor, handlelength=legend_line_length,
              labelcolor=f'{"linecolor" if use_line_color_for_legends else "None"}')


def create_patch_legend(ax, patches, legend_loc, ncol, handletextpad, font_size, bbox_to_anchor=None):
    legend = ax.legend(handles=patches, handlelength=1, frameon=False, loc=legend_loc, ncol=ncol,
                       handletextpad=handletextpad, bbox_to_anchor=bbox_to_anchor)
    plt.setp(ax.get_legend().get_texts(), fontsize=font_size, fontweight='bold')
    for patch in legend.get_patches():
        patch.set_height(0.75 * font_size)
        patch.set_width(0.75 * font_size)
        patch.set_x(-font_size / 4)


def customize_axes(ax, xlabel, ylabel, xticks_values, yticks_values, xticks_labels, yticks_labels, xlims, ylims,
                   fs_dict, tick_width, tick_length, spine_width, pad_labels, pad_ticks, use_log_scale):
    """Customize axes with labels, ticks, limits, and log scale if needed."""
    ax.set_xlabel(xlabel, fontsize=fs_dict['xlabel'], labelpad=20, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=fs_dict['ylabel'], labelpad=20, fontweight='bold')
    ax.set_xlim(xlims) if xlims is not None else None
    ax.set_ylim(ylims) if ylims is not None else None
    if xticks_values is not None:
        ax.set_xticks(xticks_values)
    if yticks_values is not None:
        ax.set_yticks(yticks_values)
    if xticks_labels is not None:
        ax.set_xticklabels(xticks_labels, fontsize=fs_dict['xticks'])
    if yticks_labels is not None:
        ax.set_yticklabels(yticks_labels, fontsize=fs_dict['yticks'])
    if use_log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.tick_params(axis='x', which='major', width=tick_width, length=tick_length, labelsize=fs_dict['xticks'], pad=pad_labels)
    ax.tick_params(axis='y', which='major', width=tick_width, length=tick_length, labelsize=fs_dict['yticks'], pad=pad_ticks)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontweight('bold') for label in labels]


def extend_y_axis(ax):
    """Extend the y-axis by one tick step."""
    yticks = ax.get_yticks()
    y_step = yticks[1] - yticks[0]
    ax.set_ylim(yticks[0], yticks[-1] + y_step)


def annotate_text(ax, texts, locations, font_sizes, bold, colors, rotate_place_text):
    """Annotate text on the plot."""
    if isinstance(texts, list):
        for text, loc, font_size, color, rotation in zip(texts, locations, font_sizes, colors, rotate_place_text):
            ax.annotate(text, xy=loc, xycoords='axes fraction', fontsize=font_size,
                        fontweight='bold' if bold else 'normal', color=color,
                        rotation=f'{"vertical" if rotation else "horizontal"}')
    else:
        ax.annotate(texts, xy=locations, xycoords='axes fraction', fontsize=font_sizes,
                    fontweight='bold' if bold else 'normal', color=colors,
                    rotation=f'{"vertical" if rotate_place_text else "horizontal"}')


def save_figure(fig, data_path_out, filename, save_svg):
    """Save the figure to disk."""
    if data_path_out and filename:
        if not os.path.exists(data_path_out):
            os.makedirs(data_path_out)
        path = os.path.join(data_path_out, filename + '.png')
        fig.savefig(path)
        if save_svg:
            svg_path = os.path.join(data_path_out, 'SVG')
            if not os.path.exists(svg_path):
                os.makedirs(svg_path)
            svg_path = os.path.join(svg_path, filename + '.svg')
            fig.savefig(svg_path, format='svg')


# Create a parametrized function for histograms
def plot_parameterized_hist(data_dictionary, bins, xlabel, ylabel, figure_title, data_path_out, filename, color_dict,
                            alpha, hist_stat='probability', xlims=None, ylims=None, xticks=None, yticks=None,
                            plot_vertical_th_line=None, legend_dictionary=None, plot_legend=True,
                            legend_loc='upper right', ncol=2, plot_kde=False, perform_dip_test=False,
                            plot_mean=False, plot_std=False, mean_round_decimal=2, std_round_decimal=2,
                            figure_size=(25, 25), font_size=115, label_size=100,
                            place_text=None, place_text_loc=None, place_text_color=None, place_text_font_size=None,
                            rotate_place_text=None,
                            make_text_bold=False, pad=0.25, handletextpad=1, save_svg=True,
                            set_ylims_one_extra_tick=False, remove_first_y=False, return_fig_instead_of_save=False):
    """
    Generate a parameterized histogram plot for given distributions.

    Parameters
    ----------
    - data_dictionary (dict): A dictionary containing the distributions to plot, formatted as {'Dist1': numpy.array, 'Dist2': numpy.array}.
    - bins (numpy.array): The bins to use for the histogram, e.g., np.arange(0, 1.1, 0.1).
    - xlabel (str): The label for the x-axis.
    - ylabel (str): The label for the y-axis.
    - figure_title (str): The title of the figure.
    - data_path_out (str): The absolute path of the folder to store the figure.
    - filename (str): The name of the file to save the plot, without file extension.
    - color_dict (dict): A dictionary mapping distribution names to colors, formatted as {'Dist1': 'color1', 'Dist2': 'color2'}.
    - alpha (dict): A dictionary mapping distribution names to opacity values (in [0, 1]), formatted as {'Dist1': value1, 'Dist2': value2}.
    - hist_stat (str, optional): The type of statistic to display on the y-axis ('probability', 'frequency', etc.).
    - xlims (list, optional): Minimum and maximum values for the x-axis [min_value, max_value].
    - ylims (list, optional): Minimum and maximum values for the y-axis [min_value, max_value].
    - xticks (numpy.array or list, optional): Values to display as ticks on the x-axis.
    - yticks (numpy.array or list, optional): Values to display as ticks on the y-axis.
    - plot_vertical_th_line (list, optional): Thresholds for the x-axis to plot vertical lines [th1, th2, th3].
    - legend_dictionary (dict, optional): A dictionary mapping distribution names to their legend names {'Dist1': 'Legend1', 'Dist2': 'Legend2'}.
    - plot_legend (bool, optional): Whether to plot the legend (default is True).
    - legend_loc (str, optional): The location of the legend (default is 'upper right').
    - ncol (int, optional): The number of columns in the legend (default is 2).
    - plot_kde (bool, optional): Whether to plot a kernel density estimation function for each histogram (default is False).
    - perform_dip_test (bool, optional): Whether to perform Hartigans' dip test for multimodality (default is False).
    - plot_mean (bool, optional): Whether to plot the mean of each distribution in the legend (default is False).
    - plot_std (bool, optional): Whether to plot the standard deviation of each distribution in the legend (default is False).
    - mean_round_decimal (int, optional): The decimal point at which to round the mean of each distribution (default is 2).
    - std_round_decimal (int, optional): The decimal point at which to round the standard deviation of each distribution (default is 2).
    - figure_size (tuple, optional): The size of the figure (width, height) (default is (25, 25)).
    - font_size (int, optional): The font size for labels, title, and legend (default is 115).
    - label_size (int, optional): The font size for x and y ticks (default is 100).
    - place_text (list, optional): Strings of text to annotate on the image ['text1', 'text2'].
    - place_text_loc (list, optional): Locations for each annotated text [[x1, y1], [x2, y2]].
    - place_text_color (list, optional): Colors for each annotated text ['color1', 'color2'].
    - place_text_font_size (int, optional): The font size for annotated text (default uses `font_size` or `label_size`).
    - rotate_place_text (bool, optional): Whether to rotate text annotations.
    - make_text_bold (bool, optional): Whether to make the annotated text bold (default is False).
    - pad (float, optional): The padding used for the image with tight layout (default is 0.25).
    - handletextpad (float, optional): The padding between the legend text and its patch (default is 1).
    - save_svg (bool, optional): Whether to save the image in an '.svg' format. The image is saved as '.png' by default.
    - set_ylims_one_extra_tick (bool, optional): Whether to expand the y-axis limits by one extra tick (default is False).
    - remove_first_y (bool, optional): Whether to remove the first tick on the y-axis to prevent overlap with the x-axis (default is False).
    - return_fig_instead_of_save (bool, optional): If True, returns the figure object instead of saving (default is False).

    Returns
    -------
    - fig (matplotlib.figure.Figure): The figure object, if `return_fig_instead_of_save` is True. Otherwise, the plot is saved to a file.
    """
    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()
    fig, ax = plt.subplots(figsize=figure_size)
    sns.set(font_scale=5.0, style='ticks')

    # Patches for legend
    patches = []
    # Plot each distribution
    for dist_name, data_dist in data_dictionary.items():
        color = color_dict.get(dist_name, 'blue')  # Default color to blue if not specified
        alpha_value = alpha.get(dist_name, 1)  # Default alpha to 1 if not specified

        sns.histplot(data=data_dist, ax=ax, color=color, stat=hist_stat, bins=bins, alpha=alpha_value,
                     edgecolor='none', kde=plot_kde, line_kws={'alpha': 1, 'linewidth': 7})

        # Build legend suffix based on options
        legend_suffix = ''
        if perform_dip_test:
            dip, pval = diptest.diptest(data_dist)
            legend_suffix += f', Dip test p-value: {round(pval, 3)}'

        if plot_mean:
            data_mean = round(np.mean(data_dist), mean_round_decimal)
            legend_suffix += f', μ: {data_mean}'
        if plot_std:
            data_std = round(np.std(data_dist, ddof=1), std_round_decimal)
            legend_suffix += f', σ: {data_std}'

        legend_name = legend_dictionary.get(dist_name, dist_name) + legend_suffix
        patches.append(mpatches.Patch(color=color, label=legend_name))

    # Vertical lines if specified
    if plot_vertical_th_line:
        for th in plot_vertical_th_line:
            ax.axvline(x=th, color='black', linewidth=5, linestyle='dashed')

    ax.set_xlabel(xlabel, fontweight='bold', labelpad=20, fontsize=font_size)
    ax.set_ylabel(ylabel, fontweight='bold', labelpad=20, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=label_size, width=5, length=20, direction='out', pad=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    [spine.set_linewidth(7) for spine in ax.spines.values()]

    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if set_ylims_one_extra_tick:
        extend_y_axis(ax)

    if remove_first_y:
        ax.yaxis.get_major_ticks()[0].set_visible(False)

    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontweight('bold') for label in labels]

    if plot_legend:
        create_patch_legend(ax, patches, legend_loc, ncol, handletextpad, font_size)

    ax.set_title(figure_title, fontsize=font_size, fontweight='bold')

    # Text annotations
    if place_text:
        annotate_text(ax, place_text, place_text_loc, place_text_font_size, make_text_bold, place_text_color, rotate_place_text)

    fig.tight_layout(pad=pad)

    # Saving logic
    if not return_fig_instead_of_save:
        save_figure(fig, data_path_out, filename, save_svg)

    plt.close(fig)
    return fig if return_fig_instead_of_save else None



def plot_parameterized_lineplot(data_dictionary_x, data_dictionary_y, xlabel, ylabel, figure_title,
                                data_path_out, filename, color_dictionary=None, legend_dictionary=None,
                                lw_dict=None, ls_dict=None, fs_dict=None, m_dict=None, ms_dict=None,
                                figure_size=None, xticks_values=None, yticks_values=None, xticks_labels=None,
                                yticks_labels=None, legend_loc="best", data_dictionary_yerr=None, xlims=None,
                                ylims=None, use_sem_instead_of_errorbar=True, alpha_dict=None, save_svg=True,
                                show_lines=True, place_text=None, place_text_loc=None, make_text_bold=True,
                                place_text_font_size=None, rotate_place_text=False, place_text_color=None,
                                use_log_scale=False, use_mask=True, tick_width=4, tick_length=16,
                                yerr_capsize=20, yerr_elinewidth=5, capthick=3, spine_width=7, pad=0.25,
                                use_sns=True, plot_vertical_line=False, vertical_line_x=0, vertical_line_color='black',
                                vertical_line_ls='dashed', vertical_line_lw=4, plot_horizontal_line=False,
                                horizontal_line_y=0, horizontal_line_color='black', horizontal_line_ls='dashed',
                                horizontal_line_lw=4, add_colorbar=False, colorbar_colormap=None, colorbar_label=None,
                                colorbar_loc='right', colorbar_ticks=None, colorbar_tick_labels=None,
                                colorbar_orientation='vertical', legend_line_length=1, use_line_color_for_error=False,
                                pad_ticks=True, pad_labels=True, ncol=1, handletextpad=1, bbox_to_anchor=None,
                                font_size=90, legend_lw_multiplier=3, use_line_color_for_legends=False,
                                return_fig_instead_of_save=False):
    """
    Creates and optionally saves a parameterized line plot based on provided data, customization options, and plot settings.

    Parameters:
    ----------
    - data_dictionary_x (dict): X values for each line. Format: {'Line1': xvalues1, 'Line2': xvalues2, ...}.
    - data_dictionary_y (dict): Y values for each line. Format: {'Line1': yvalues1, 'Line2': yvalues2, ...}.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figure_title (str): Title of the figure.
    - data_path_out (str): Path where the plot is saved. Filename is added automatically.
    - filename (str): Filename for saving the plot. Extension is added automatically.
    - color_dictionary (dict, optional): Color for each line. Format: {'Line1': 'color1', ...}.
    - legend_dictionary (dict, optional): Legend names for each line. Format: {'Line1': 'Legend1', ...}.
    - lw_dict (dict, optional): Line width for each line. Format: {'Line1': lw1, ...}.
    - ls_dict (dict, optional): Line style for each line. Format: {'Line1': ls1, ...}.
    - fs_dict (dict, optional): Font size for text elements (title, axes labels, ticks). Format: {'xlabel': font_size1, ...}.
    - m_dict (dict, optional): Marker style for each line. Format: {'Line1': 'style1', ...}.
    - ms_dict (dict, optional): Marker size for each line. Format: {'Line1': size1, ...}.
    - figure_size (tuple, optional): Figure size as (width, height).
    - xticks_values (list/array, optional): X-axis tick values.
    - yticks_values (list/array, optional): Y-axis tick values.
    - xticks_labels (list/array, optional): Custom labels for X-axis ticks.
    - yticks_labels (list/array, optional): Custom labels for Y-axis ticks.
    - legend_loc (str, optional): Location of the legend. Defaults to "best".
    - data_dictionary_yerr (dict, optional): Y-error for each point of each line. Format: {'Line1': yerrvalues1, ...}.
    - xlims (list, optional): X-axis limits as [min, max].
    - ylims (list, optional): Y-axis limits as [min, max].
    - use_sem_instead_of_errorbar (bool, optional): If True, use standard error mean; else use error bars.
    - alpha_dict (dict, optional): Opacity for each line. Format: {'Line1': alpha1, ...}.
    - save_svg (bool, optional): If True, saves the plot as SVG in addition to PNG.
    - show_lines (bool, optional): If True, lines are shown in the legend.
    - place_text (list, optional): Texts to annotate on the plot. Format: ['text1', 'text2', ...].
    - place_text_loc (list, optional): Locations for each annotated text. Format: [[x1, y1], [x2, y2], ...].
    - make_text_bold (bool, optional): If True, annotated text is bold.
    - place_text_font_size (list, optional): Font size for each annotated text. Format: [size1, size2, ...].
    - rotate_place_text (bool, optional): If True, annotated text is vertical.
    - place_text_color (list, optional): Color for each annotated text. Format: ['color1', 'color2', ...].
    - use_log_scale (bool, optional): If True, both axes are in log scale.
    - use_mask (bool, optional): If True, plots only points with finite Y values.
    - tick_width (int, optional): Width of axis ticks.
    - tick_length (int, optional): Length of axis ticks.
    - yerr_capsize (int, optional): Width of the cap in error bars.
    - yerr_elinewidth (int, optional): Line width of the error bar line.
    - capthick (int, optional): Thickness of the cap in error bars.
    - spine_width (int, optional): Width of the axis spines.
    - pad (float, optional): Padding used with tight layout.
    - use_sns (bool, optional): If True, uses seaborn for some plot settings.
    - plot_vertical_line (bool, optional): If True, plots a vertical line.
    - vertical_line_x (float, optional): X-axis value for vertical line.
    - vertical_line_color (str, optional): Color of the vertical line.
    - vertical_line_ls (str, optional): Line style of the vertical line.
    - vertical_line_lw (int, optional): Line width of the vertical line.
    - plot_horizontal_line (bool, optional): If True, plots a horizontal line.
    - horizontal_line_y (float, optional): Y-axis value for horizontal line.
    - horizontal_line_color (str, optional): Color of the horizontal line.
    - horizontal_line_ls (str, optional): Line style of the horizontal line.
    - horizontal_line_lw (int, optional): Line width of the horizontal line.
    - add_colorbar (bool, optional): If True, adds a colorbar to the plot.
    - colorbar_colormap (str, optional): Colormap for the colorbar.
    - colorbar_label (str, optional): Label for the colorbar.
    - colorbar_loc (str, optional): Location of the colorbar.
    - colorbar_ticks (list/array, optional): Tick values for the colorbar.
    - colorbar_tick_labels (list/array, optional): Custom labels for colorbar ticks.
    - colorbar_orientation (str, optional): Orientation of the colorbar ('vertical' or 'horizontal').
    - legend_line_length (int, optional): Length of the lines in the legend.
    - use_line_color_for_error (bool, optional): If True, error bars use line color.
    - pad_ticks (bool, optional): If True, pads the axis ticks.
    - pad_labels (bool, optional): If True, pads the axis labels.
    - ncol (int, optional): Number of columns in the legend.
    - handletextpad (float, optional): Padding between legend text and its line.
    - bbox_to_anchor (tuple, optional): Location for the legend's bounding box anchor.
    - font_size (int, optional): Font size for labels, title, and legend.
    - legend_lw_multiplier (int, optional): Multiplier for the linewidth in the legend.
    - use_line_color_for_legends (bool, optional): If True, legend text matches line color.
    - return_fig_instead_of_save (bool, optional): If True, returns the figure instance; otherwise, saves the plot.

    Returns:
    ----------
    - fig (matplotlib.figure.Figure): The figure object, if `return_fig_instead_of_save` is True. Otherwise, the plot is saved to a file.
    """

    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()
    if use_sns:
        sns.set(font_scale=5.0, style='ticks')

    fig, ax = plt.subplots(figsize=figure_size)

    if fs_dict is None:
        fs_dict = {'xlabel': font_size,
                   'ylabel': font_size,
                   'title': font_size,
                   'legend': font_size,
                   'xticks': font_size/1.125,
                   'yticks': font_size/1.125}

    for key in data_dictionary_x:
        x = data_dictionary_x[key]
        y = data_dictionary_y[key]
        if use_mask:
            mask = np.isfinite(y)
        else:
            mask = np.arange(len(y))
        color = color_dictionary.get(key, 'blue')  # Default color is blue if not given
        lw = lw_dict.get(key, 2)  # Default linewidth is 2 if not given
        ls = ls_dict.get(key, 'solid')  # Default linestyle is solid if not given
        marker = m_dict.get(key, None)  # Default marker is None if not given
        markersize = ms_dict.get(key, 5)  # Default marker size is 5 if not given
        alpha = alpha_dict.get(key, 1)  # Default alpha is 1 if not given

        # Plot lines with or without error bars
        if data_dictionary_yerr and key in data_dictionary_yerr:
            yerr = data_dictionary_yerr[key]
            if use_sem_instead_of_errorbar:
                ax.fill_between(x, y - yerr, y + yerr, color=color if use_line_color_for_error else 'black', alpha=alpha)
            else:
                ax.errorbar(x, y, yerr=yerr, fmt=marker, ecolor=color if use_line_color_for_error else 'black',
                            lw=lw, ls=ls, markersize=markersize, capsize=yerr_capsize,
                            elinewidth=yerr_elinewidth, capthick=capthick, alpha=alpha, color=color)
        else:
            ax.plot(x[mask], y[mask], color=color, marker=marker, lw=lw, ls=ls, markersize=markersize, alpha=alpha)

    # Additional customization for axes, ticks, labels, etc.
    customize_axes(ax, xlabel, ylabel, xticks_values, yticks_values, xticks_labels, yticks_labels, xlims, ylims, fs_dict,
                   tick_width, tick_length, spine_width, pad_labels, pad_ticks, use_log_scale)


    if legend_dictionary:
        create_line_legend(ax, legend_dictionary, color_dictionary, lw_dict, ls_dict, m_dict, ms_dict, show_lines,
                           legend_loc, ncol, handletextpad, legend_line_length, bbox_to_anchor, fs_dict['legend'], legend_lw_multiplier,
                           use_line_color_for_legends)

    if place_text:
        annotate_text(ax, place_text, place_text_loc, place_text_font_size, make_text_bold, place_text_color, rotate_place_text)

    if plot_vertical_line:
        for i in range(len(vertical_line_x)):
            ax.axvline(x=vertical_line_x[i], color=vertical_line_color[i], ls=vertical_line_ls[i], lw=vertical_line_lw[i])

    if plot_horizontal_line:
        for i in range(len(horizontal_line_y)):
            ax.axhline(y=horizontal_line_y[i], color=horizontal_line_color[i], ls=horizontal_line_ls[i], lw=horizontal_line_lw[i])

    if add_colorbar:
        add_color_bar(fig, ax, colorbar_colormap, colorbar_label, colorbar_loc, colorbar_ticks, colorbar_tick_labels, colorbar_orientation,
                      fs_dict['xlabel'], fs_dict['xticks'])

    ax.set_title(figure_title, fontsize=fs_dict['title'], fontweight='bold')
    fig.tight_layout(pad=pad)

    if return_fig_instead_of_save:
        return fig
    else:
        save_figure(fig, data_path_out, filename, save_svg)
        plt.close(fig)


def parameterized_scatterplot(data_dictionary_x, data_dictionary_y, xlabel, ylabel, figure_title,
                              data_path_out, filename, xlims=None, ylims=None, color_dictionary=None,
                              alpha=None, legend_dictionary=None, font_size=90, label_size=80, m_dict=None,
                              ms_dict=None, figure_size=(25, 25), xticks_values=None, yticks_values=None,
                              xticks_labels=None, yticks_labels=None, legend_loc="best",
                              plot_r2_score=False, save_svg=True, m_size_factor=4, use_log_scale=False,
                              add_colorbar=False, colorbar_colormap=None, color_values=None, colorbar_label=None,
                              colorbar_loc='right', colorbar_ticks=None, colorbar_tick_labels=None, colorbar_orientation=None,
                              place_text=None, place_text_loc=None, place_text_font_size=None, make_text_bold=False,
                              rotate_place_text=False, place_text_color=None, pad=0.25, plot_horizontal_line=False,
                              horizontal_line_y=[0], horizontal_line_color=['black'], horizontal_line_ls=['dashed'],
                              horizontal_line_lw=[4], do_linear_reg_fit=False, linear_reg_fit_lw=5, linear_reg_fit_ls='solid',
                              plot_vertical_line=False, vertical_line_x=[0], vertical_line_color=['black'],
                              vertical_line_ls=['dashed'], vertical_line_lw=[4], spine_width=7,
                              ncol=1, handletextpad=0.8, m_linewidth=1, tick_width=4, tick_length=16,
                              return_fig_instead_of_save=False):
    """
    Creates and optionally saves a parameterized scatter plot based on provided data,
    customization options, and plot settings.

    Parameters:
    ----------
    - data_dictionary_x (dict): X values for each distribution. Format: {'Dist1': xvalues1, ...}.
    - data_dictionary_y (dict): Y values for each distribution. Format: {'Dist1': yvalues1, ...}.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figure_title (str): Title of the figure.
    - data_path_out (str): Output directory for saving the plot.
    - filename (str): Filename for saving the plot without file extension.
    - xlims (list, optional): X-axis limits as [min, max].
    - ylims (list, optional): Y-axis limits as [min, max].
    - color_dictionary (dict, optional): Colors for each distribution. Format: {'Dist1': 'color1', ...}.
    - alpha (dict, optional): Opacity for each distribution. Format: {'Dist1': alpha1, ...}.
    - legend_dictionary (dict, optional): Legend names for each distribution. Format: {'Dist1': 'Legend1', ...}.
    - font_size (int, optional): Font size for the title, labels, and legend text.
    - label_size (int, optional): Font size for the axis tick labels.
    - m_dict (dict, optional): Marker style for each distribution. Format: {'Dist1': 'style1', ...}.
    - ms_dict (dict, optional): Marker size for each distribution. Format: {'Dist1': size1, ...}.
    - figure_size (tuple, optional): Size of the figure (width, height).
    - xticks_values, yticks_values (list/array, optional): Values for custom x and y axis ticks.
    - xticks_labels, yticks_labels (list/array, optional): Custom labels for x and y axis ticks.
    - legend_loc (str, optional): Location of the legend.
    - plot_r2_score (bool, optional): Whether to annotate R2 score for each distribution; requires linear regression fit.
    - save_svg (bool, optional): Whether to save the plot as SVG in addition to PNG.
    - m_size_factor (float, optional): Factor to scale the marker sizes defined in ms_dict.
    - use_log_scale (bool, optional): Whether to use log scale for both axes.
    - add_colorbar (bool, optional): If True, adds a colorbar to the plot.
    - colorbar_colormap (str, optional): Colormap for the colorbar.
    - color_values (list/array, optional): Values to map colors using the colormap.
    - colorbar_label (str, optional): Label for the colorbar when using a colormap.
    - colorbar_loc (str, optional): Location of the colorbar.
    - colorbar_ticks (list/array, optional): Tick values for the colorbar.
    - colorbar_tick_labels (list/array, optional): Custom labels for colorbar ticks.
    - colorbar_orientation (str, optional): Orientation of the colorbar ('vertical' or 'horizontal').
    - place_text (list, optional): Texts to annotate on the plot.
    - place_text_loc (list, optional): Locations for each text annotation.
    - place_text_font_size (list, optional): Font sizes for text annotations.
    - make_text_bold (bool, optional): Whether annotated text should be bold.
    - rotate_place_text (bool, optional): Whether to rotate text annotations.
    - place_text_color (list, optional): Colors for text annotations.
    - pad (float, optional): Padding for layout adjustment.
    - plot_horizontal_line (bool, optional): Whether to plot horizontal line(s).
    - horizontal_line_y (list, optional): A list that contains values in the y axis where you want the horizontal lines to be (e.g. [5, -1, etc]).
    - horizontal_line_color (list, optional): A list that contains the color of each horizontal line (e.g. ['color1', 'color2', etc]).
    - horizontal_line_ls (list, optional): A list that contains the linestyle of each horizontal line (e.g. ['solid', 'dashed', etc]).
    - horizontal_line_lw (list, optional): A list that contains the linewidth of each horizontal line (e.g. [2, 5, etc]).
    - do_linear_reg_fit (bool, optional): Whether to perform linear regression fitting.
    - linear_reg_fit_lw (float, optional): Line width for linear regression fit lines.
    - linear_reg_fit_ls (str, optional): Line style for linear regression fit lines.
    - plot_vertical_line (bool, optional): Whether to plot vertical line(s).
    - vertical_line_x (list, optional): A list that contains values in the x axis where you want the vertical lines to be (e.g. [5, -1, etc]).
    - vertical_line_color (list, optional): A list that contains the color of each vertical line (e.g. ['color1', 'color2', etc]).
    - vertical_line_ls (list, optional): A list that contains the linestyle of each vertical line (e.g. ['solid', 'dashed', etc]).
    - vertical_line_lw (list, optional): A list that contains the linewidth of each vertical line (e.g. [2, 5, etc]).
    - spine_width (float, optional): Width of the plot spines.
    - ncol (int, optional): Number of columns in the legend.
    - handletextpad (float, optional): Padding between the legend text and its patch.
    - m_linewidth (float, optional): Line width for connecting markers in each distribution.
    - tick_width, tick_length (float, optional): Width and length of axis ticks.
    - return_fig_instead_of_save (bool, optional): If True, returns figure object instead of saving.

    Returns:
    ----------
    - matplotlib.figure.Figure: The figure object if `return_fig_instead_of_save` is True. Otherwise, saves the plot.
    """
    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()
    fig, ax = plt.subplots(figsize=figure_size)
    sns.set(font_scale=5.0, style='ticks')

    # Prepare the font size dictionary for axis customization
    fs_dict = {'xlabel': font_size, 'ylabel': font_size, 'xticks': label_size, 'yticks': label_size}

    # Plot each distribution
    for key in data_dictionary_x.keys():
        x = data_dictionary_x[key]
        y = data_dictionary_y[key]
        # Apply marker size factor
        current_ms = m_size_factor * ms_dict.get(key, 5)

        if add_colorbar:
            # If colormap is specified, plot scatter with colormap
            ax.scatter(x, y, c=color_values[key], cmap=colorbar_colormap,
                       alpha=alpha.get(key, 1), s=current_ms, label=legend_dictionary.get(key, ''), linewidth=m_linewidth)
            if key == list(data_dictionary_x.keys())[-1]:  # Add colorbar on the last key iteration
                add_color_bar(fig, ax, colorbar_colormap, colorbar_label, colorbar_loc, colorbar_ticks,
                              colorbar_tick_labels, colorbar_orientation, font_size, label_size)
        else:
            ax.scatter(x, y, color=color_dictionary.get(key, 'blue'),
                       alpha=alpha.get(key, 1), s=current_ms, label=legend_dictionary.get(key, ''), marker=m_dict.get(key, 'o'), linewidth=m_linewidth)

        if do_linear_reg_fit and plot_r2_score:
            # Perform linear regression fitting if requested
            x_values = np.array(x).reshape(-1, 1)
            y_values = np.array(y)
            reg = LinearRegression().fit(x_values, y_values)
            y_pred = reg.predict(x_values)
            ax.plot(data_dictionary_x[key], y_pred, color=color_dictionary.get(key, 'blue'), linewidth=linear_reg_fit_lw)
            r2 = r2_score(y_values, y_pred)
            legend_dictionary[key] += f', R2: {round(r2, 2)}'

    # Customize axes with labels, ticks, limits, and optionally apply log scale
    customize_axes(ax, xlabel, ylabel, xticks_values, yticks_values, xticks_labels, yticks_labels, xlims, ylims, fs_dict, tick_width,
                   tick_length, spine_width, pad, pad, use_log_scale)

    # Annotate text on the plot if any
    if place_text:
        annotate_text(ax, place_text, place_text_loc, place_text_font_size, make_text_bold, place_text_color, rotate_place_text)

    # Additional plot customizations
    if plot_horizontal_line:
        for y, color, ls, lw in zip(horizontal_line_y, horizontal_line_color, horizontal_line_ls, horizontal_line_lw):
            ax.axhline(y=y, color=color, linestyle=ls, linewidth=lw)

    if plot_vertical_line:
        for x, color, ls, lw in zip(vertical_line_x, vertical_line_color, vertical_line_ls, vertical_line_lw):
            ax.axvline(x=x, color=color, linestyle=ls, linewidth=lw)

    # Create the legend
    if legend_dictionary:
        create_line_legend(ax, legend_dictionary, color_dictionary, {k: linear_reg_fit_lw for k in data_dictionary_x},
                           {k: linear_reg_fit_ls for k in data_dictionary_x}, m_dict, ms_dict, True, legend_loc, ncol, handletextpad, 2,
                           None, font_size, 1, False)

    ax.set_title(figure_title, fontsize=font_size, fontweight='bold')

    # Adjusting layout
    fig.tight_layout(pad=pad)

    # Save or return the figure
    if return_fig_instead_of_save:
        return fig
    else:
        save_figure(fig, data_path_out, filename, save_svg)
        plt.close(fig)


def plot_parameterized_barplot(data_dictionary, xlabel, ylabel, figure_title, data_path_out, filename,
                               color_dict_bars, color_dict_legends, bins, bar_width,
                               xticks, xticks_labels, legends, yticks=None, yticks_labels=None,
                               align='edge', alpha=1.0, edgecolor=None, rotate_xticks=False,
                               sem_data_dictionary=None,
                               remove_first_y=False, xlims=None, ylims=None, figure_size=(25, 25),
                               legend_loc='upper right', ncol=2, font_size=90, label_size=75,
                               capsize=5, elinewidth=2, capthick=2, plot_minor_ticks=False,
                               spine_width=2, yticks_font_size=None, xticks_font_size=None,
                               set_ylims_one_extra_tick=False, pad=0.25, save_svg=True, place_text=None,
                               place_text_loc=None, place_text_color=None, place_text_font_size=None,
                               rotate_place_text=None, make_text_bold=False, use_sns=True, disable_xtick_edges=False,
                               bbox_to_anchor=None, make_legend_patch_visible=True,
                               handletextpad=0.5, tick_width=1, tick_length=5,
                               return_fig_instead_of_save=False):
    """
    Creates and optionally saves a parameterized bar plot based on provided data,
    customization options, and plot settings.

    Parameters:
    ----------
    - data_dictionary (dict): Heights of bars {'Bar1': height1, 'Bar2': height2, ...}.
    - xlabel (str): Label for the X-axis.
    - ylabel (str): Label for the Y-axis.
    - figure_title (str): Title of the figure.
    - data_path_out (str): Output directory for saving the plot.
    - filename (str): Filename for saving the plot, without file extension.
    - color_dict_bars (dict): Colors for each bar {'Bar1': 'color1', 'Bar2': 'color2', ...}.
    - color_dict_legends (dict): Colors for each legend {'Legend1': 'color1', 'Legend2': 'color2', ...}.
    - bins (dict): X-axis positions of each bar {'Bar1': xvalue1, 'Bar2': xvalue2, ...}.
    - bar_width (dict): Width of each bar {'Bar1': width1, 'Bar2': width2, ...}.
    - xticks (numpy.array/list): Values to show as ticks on the X-axis.
    - xticks_labels (numpy.array/list): Labels to overwrite the ticks on the X-axis.
    - yticks (numpy.array/list): Values to show as ticks on the Y-axis.
    - yticks_labels (numpy.array/list): Labels to overwrite the ticks on the Y-axis.
    - legends (list): Distribution names for the legend ['Dist1', 'Dist2', ...].
    - align (str, optional): Whether bars are centered ('center') or aligned by their edge ('edge').
    - alpha (float, optional): Opacity value for all bars, between 0 (transparent) and 1 (opaque).
    - edgecolor (dict, optional): Colors of the contour of each bar {'Bar1': 'color1', 'Bar2': 'color2', ...}.
    - rotate_xticks (bool, optional): Whether to rotate the X-axis ticks (True for vertical, False for horizontal).
    - sem_data_dictionary (dict, optional): Standard error of the mean for each bar {'Bar1': sem1, 'Bar2': sem2, ...}.
    - remove_first_y (bool, optional): Whether to remove the first Y-axis tick to avoid overlap with the X-axis.
    - xlims (list, optional): Minimum and maximum values for the X-axis [min_value, max_value].
    - ylims (list, optional): Minimum and maximum values for the Y-axis [min_value, max_value].
    - figure_size (tuple, optional): Width and height of the figure (width, height).
    - legend_loc (str, optional): Location of the legend.
    - ncol (int, optional): Number of columns in the legend.
    - font_size (int, optional): Font size for title, labels, and legend.
    - label_size (int, optional): Font size for X and Y ticks.
    - capsize (int, optional): Width of the caps at the ends of error bars.
    - elinewidth (int, optional): Line width of the error bars.
    - capthick (int, optional): Thickness of the error bar caps.
    - plot_minor_ticks (bool, optional): Whether to show minor ticks on axes.
    - spine_width (int, optional): Width of the axes spines.
    - yticks_font_size, xticks_font_size (int, optional): Font size for Y and X ticks, if different from label_size.
    - set_ylims_one_extra_tick (bool, optional): Expand Y-axis limits to include one extra tick beyond the data range.
    - pad (float, optional): Padding used in layout adjustment.
    - save_svg (bool, optional): Whether to save the figure in SVG format in addition to PNG.
    - place_text (list, optional): Strings of text to annotate on the image ['text1', 'text2'].
    - place_text_loc (list, optional): Locations for each annotated text [[x1, y1], [x2, y2]].
    - place_text_color (list, optional): Colors for each annotated text ['color1', 'color2'].
    - place_text_font_size (int, optional): The font size for annotated text (default uses `font_size` or `label_size`).
    - rotate_place_text (bool, optional): Whether to rotate text annotations.
    - make_text_bold (bool, optional): Whether to make the annotated text bold (default is False).
    - use_sns (bool, optional): Whether to use seaborn for setting some plot parameters.
    - disable_xtick_edges (bool, optional): Whether to remove tick lines from the X-axis.
    - bbox_to_anchor (tuple, optional): Location for the legend's bounding box anchor.
    - make_legend_patch_visible (bool, optional): Whether to show patches (color squares) in the legend.
    - handletextpad (float, optional): Padding between legend text and the patch (color square).
    - tick_width, tick_length (int, optional): Width and length of the ticks on both axes.
    - return_fig_instead_of_save (bool, optional): If True, returns figure object instead of saving.

    Returns:
    ----------
    - matplotlib.figure.Figure or None: The figure object if `return_fig_instead_of_save` is True; otherwise, the plot is saved and nothing is returned.
    """
    # Turn off interactive mode so that the plots don't show unless commanded.
    plt.ioff()
    # Create a figure with the specified figure size.
    fig, ax = plt.subplots(figsize=figure_size)

    # Set seaborn style for plotting if requested.
    if use_sns:
        sns.set(font_scale=5.0, style='ticks')

    # Define font sizes
    fs_dict = {
        'xlabel': font_size, 'ylabel': font_size,
        'xticks': xticks_font_size if xticks_font_size else label_size,
        'yticks': yticks_font_size if yticks_font_size else label_size
    }

    # Customize the appearance of the plot axes and labels
    customize_axes(ax, xlabel, ylabel, xticks, yticks, xticks_labels, yticks_labels, xlims, ylims, fs_dict, tick_width, tick_length, spine_width, 20, 20, False)

    # Initialize list to store legend patches.
    patches = []

    # Plot each bar using the data from the data dictionary.
    for bar, height in data_dictionary.items():
        ax.bar(
            bins[bar], height, color=color_dict_bars.get(bar, 'blue'), width=bar_width[bar],
            edgecolor=None if edgecolor is None else edgecolor.get(bar, 'black'), alpha=alpha,
            align=align, yerr=sem_data_dictionary.get(bar, None),
            capsize=capsize, error_kw={'elinewidth': elinewidth, 'capthick': capthick}
        )

    # Rotate X-axis tick labels if requested.
    if rotate_xticks:
        plt.xticks(rotation=90)

    # Extend Y-axis to include one extra tick if requested.
    if set_ylims_one_extra_tick:
        extend_y_axis(ax)

    # Remove the first Y-axis tick to avoid overlap with the X-axis if requested.
    if remove_first_y:
        ax.yaxis.get_major_ticks()[0].set_visible(False)

    # Add minor ticks to the axes if requested.
    if plot_minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_tick_params(which='minor', width=tick_width/2, length=tick_length/2, direction='out')
        ax.yaxis.set_tick_params(which='minor', width=tick_width/2, length=tick_length/2, direction='out')

    # Remove tick lines from the X-axis if requested.
    if disable_xtick_edges:
        ax.xaxis.set_tick_params(which='major', width=0, length=0, direction='out')

    # Add a legend to the plot using the provided legend names and colors.
    if legends is not None:
        for legend_name in legends:
            patches.append(mpatches.Patch(color=color_dict_legends[legend_name],
                                          label=legend_name, visible=make_legend_patch_visible))
            create_patch_legend(ax, patches, legend_loc, ncol, handletextpad, font_size, bbox_to_anchor)

    # Annotate the plot with text notes if provided.
    if place_text:
        annotate_text(ax, place_text, place_text_loc, place_text_font_size, make_text_bold, place_text_color, rotate_place_text)

    # Set the title of the plot with the specified font size and make it bold.
    ax.set_title(figure_title, fontsize=font_size, fontweight='bold')

    # Adjust the layout of the plot to fit within the figure area.
    fig.tight_layout(pad=pad)

    # Save the figure to the specified path or return the figure object.
    if return_fig_instead_of_save:
        return fig
    else:
        save_figure(fig, data_path_out, filename, save_svg)
        plt.close()

