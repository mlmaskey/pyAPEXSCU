import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from matplotlib import ticker
import os

warnings.filterwarnings('ignore')


def set_ticks_for_axis(df, dim, ax, val_range, cols, ticks):
    min_val, max_val, val_range = val_range[cols[dim]]
    step = val_range / float(ticks - 1)
    tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
    norm_min = df[cols[dim]].min()
    norm_range = np.ptp(df[cols[dim]])
    norm_step = norm_range / float(ticks - 1)
    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
    ax.yaxis.set_ticks(ticks)
    ax.set_yticklabels(tick_labels)


def read_data(site, scenario, metric):
    data = pd.read_csv(f'../post_analysis/Results/Uncertainty_{site}_{scenario}_{metric}.csv', index_col=0)
    data = data.reset_index()
    data = data.drop('index', axis=1)
    data.drop(index=data.index[0], inplace=True)
    data.drop(index=data.index[7], inplace=True)
    data = data.reset_index()
    data = data.drop('index', axis=1)
    return data


def plot_pc_data(df, list_var):
    df_ready = df[list_var]
    df_ready.insert(0, 'id_mode', df_ready.index + 1)
    df_ready['id_mode'] = pd.cut(df_ready['id_mode'], np.arange(df_ready.shape[0] + 1))
    return df_ready


def categorize_color(df, list_var, color_list):
    x = [i for i, _ in enumerate(list_var)]
    color_list = {df['id_mode'].cat.categories[i]: color_list[i] for i, _ in enumerate(df['id_mode'].cat.categories)}
    return color_list, x


def normalize_data(data, list_var):
    range_val = {}
    for col in list_var:
        range_val[col] = [data[col].min(), data[col].max(), np.ptp(data[col])]
        data[col] = np.true_divide(data[col] - data[col].min(), np.ptp(data[col]))
    return range_val, data


def position_label_ticks(df, axes, val_range, list_vars, n_ticks):
    # Tick positions based on normalised data
    # Tick labels are based on original data
    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(df, dim, ax, val_range, list_vars, ticks=n_ticks)
        ax.set_xticklabels([list_vars[dim]])


def parallel_coordinates(site, scenario, cols, colors, metric='OF', fig_size=(12, 4)):
    df1 = read_data(site, scenario, metric)
    df = plot_pc_data(df1, list_var=cols)
    # create dict of categories: colors
    colors, x = categorize_color(df, list_var=cols, color_list=colors)
    # Create (X-1) subplots along xaxis
    fig, axes = plt.subplots(1, len(x) - 1, sharey=False, figsize=fig_size)
    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range, df = normalize_data(data=df, list_var=cols)

    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df.index:
            mode_category = df.loc[idx, 'id_mode']
            ax.plot(x, df.loc[idx, cols], colors[mode_category])
            ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim([x[i], x[i + 1]])

    # Set the tick positions and labels on yaxis for each plot
    position_label_ticks(df, axes, val_range=min_max_range, list_vars=cols, n_ticks=5)
    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ax.tick_params(axis='both', which='major', labelsize=14)
    set_ticks_for_axis(df, dim, ax, val_range=min_max_range, cols=cols, ticks=5)
    ax.set_xticklabels([cols[-2], cols[-1]])

    # Remove space between subplots
    plt.subplots_adjust(wspace=0)
    legend_list = [''r'$\mu-3\sigma$', ''r'$\mu-2\sigma$', ''r'$\mu-\sigma$', ''r'$\mu$', ''r'$\mu+\sigma$',
                   ''r'$\mu+2\sigma$', ''r'$\mu+3\sigma$', 'Calibrated']
    # Add legend to plot
    plt.legend(
        [plt.Line2D((0, 1), (0, 0), color=colors[cat]) for cat in df['id_mode'].cat.categories],
        legend_list, bbox_to_anchor=(0.5, -0.10), borderaxespad=0., ncol=8, fontsize="12")

    return plt


def df_parallel_coordinates(df1, cols, colors, labels, fig_size=(12, 4)):
    df = plot_pc_data(df1, list_var=cols)
    # create dict of categories: colors
    colors, x = categorize_color(df, list_var=cols, color_list=colors)
    # Create (X-1) subplots along xaxis
    fig, axes = plt.subplots(1, len(x) - 1, sharey=False, figsize=fig_size)
    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range, df = normalize_data(data=df, list_var=cols)

    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df.index:
            mode_category = df.loc[idx, 'id_mode']
            ax.plot(x, df.loc[idx, cols], colors[mode_category])
            ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim([x[i], x[i + 1]])

    # Set the tick positions and labels on yaxis for each plot
    position_label_ticks(df, axes, val_range=min_max_range, list_vars=cols, n_ticks=5)
    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ax.tick_params(axis='both', which='major', labelsize=14)
    set_ticks_for_axis(df, dim, ax, val_range=min_max_range, cols=cols, ticks=5)
    ax.set_xticklabels([cols[-2], cols[-1]])

    # Remove space between subplots
    plt.subplots_adjust(wspace=0)
    legend_list = labels
    # Add legend to plot
    plt.legend(
        [plt.Line2D((0, 1), (0, 0), color=colors[cat]) for cat in df['id_mode'].cat.categories],
        legend_list, bbox_to_anchor=(0.5, -0.10), borderaxespad=0., ncol=8, fontsize="12")

    return plt
