import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib import ticker

graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)

print('\014')


def get_heatmap_data(field, is_grazing, stats):
    if is_grazing:
        scn = 'g'
    else:
        scn = 'n'
    read_dir = f'../post_analysis/Results/{field}_{scn}_Uncertainty_range.csv'
    read_dir1 = f'../post_analysis/Results/{field}_{scn}_Uncertainty_annual_likelihood_{stats}.csv'
    range_str = []
    df_gl = pd.read_csv(read_dir, index_col=0)
    range_vec = df_gl.Percent
    for rng in range_vec:
        range_str.append(f'{rng:2.3f}')
    df_range = pd.read_csv(read_dir1, index_col=0)
    df_range.columns = range_str
    df_range[df_range == 0] = np.nan

    df_wt_lh = pd.DataFrame(np.nan, index=df_range.index, columns=df_range.columns)
    for i in range(df_range.shape[0]):
        df_lh = df_range.iloc[i, :]
        sum_lh = df_lh.sum()
        df_wt_lh.iloc[i, :] = df_lh / sum_lh

    mat_gl = np.log10(df_range)
    return mat_gl, df_wt_lh, df_range


def plot_heat_map(df, custom_delta, ax, x_label, y_label, title_str, n_y_ticks):
    data = df.T
    data.sort_index(ascending=True)
    sigma_range = data.index.astype(float)
    y_ticks = np.arange(sigma_range.min(), sigma_range.max()+custom_delta, custom_delta)
    sns.heatmap(data=data, ax=ax, vmin=data.min().min(), vmax=data.max().max(), cmap='BuPu',
                yticklabels=y_ticks, cbar_kws={'label': 'Log likelihood'})
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title_str, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(n_y_ticks))
    cbar = ax.collections[0].colorbar
    # here set the label size by 12
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.label.set_size(12)
    return ax


heat_map_data_1_n, _, _ = get_heatmap_data(field='Farm_1', is_grazing=False, stats='mse')
heat_map_data_1_n = heat_map_data_1_n[heat_map_data_1_n.columns[::-1]]
heat_map_data_1_g, _, _ = get_heatmap_data(field='Farm_1', is_grazing=True, stats='mse')
heat_map_data_1_g = heat_map_data_1_g[heat_map_data_1_g.columns[::-1]]
heat_map_data_8_n, _, _ = get_heatmap_data(field='Farm_8', is_grazing=False, stats='mse')
heat_map_data_8_n = heat_map_data_8_n[heat_map_data_8_n.columns[::-1]]
heat_map_data_8_g, _, _ = get_heatmap_data(field='Farm_8', is_grazing=True, stats='mse')
heat_map_data_8_g = heat_map_data_8_g[heat_map_data_8_g.columns[::-1]]

fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=False, sharey=True)
axes[0, 0] = plot_heat_map(heat_map_data_1_n, custom_delta=0.5, ax=axes[0, 0], x_label=None,
                           y_label=''r'$\mu+\Delta{\sigma}$', title_str="WRE1", n_y_ticks=15)
axes[1, 0] = plot_heat_map(heat_map_data_1_g, custom_delta=0.5, ax=axes[1, 0], x_label=None,
                           y_label=''r'$\mu+\Delta{\sigma}$', title_str=None, n_y_ticks=15)
axes[0, 1] = plot_heat_map(heat_map_data_8_n, custom_delta=0.5, ax=axes[0, 1], x_label='Year', y_label=None,
                           title_str='WRE8', n_y_ticks=15)
axes[1, 1] = plot_heat_map(heat_map_data_8_g, custom_delta=0.5, ax=axes[1, 1], x_label='Year', y_label=None,
                           title_str=None, n_y_ticks=15)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Heatmap_Uncertainty_MSE_GLUE.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Heatmap_Uncertainty_MSE_GLUE.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Heatmap_Uncertainty_MSE_GLUE.jpeg'), dpi=600, bbox_inches="tight")
plt.show()

heat_map_data_1_n, _, _ = get_heatmap_data(field='Farm_1', is_grazing=False, stats='nse')
heat_map_data_1_n = heat_map_data_1_n[heat_map_data_1_n.columns[::-1]]
heat_map_data_1_g, _, _ = get_heatmap_data(field='Farm_1', is_grazing=True, stats='nse')
heat_map_data_1_g = heat_map_data_1_g[heat_map_data_1_g.columns[::-1]]
heat_map_data_8_n, _, _ = get_heatmap_data(field='Farm_8', is_grazing=False, stats='nse')
heat_map_data_8_n = heat_map_data_8_n[heat_map_data_8_n.columns[::-1]]
heat_map_data_8_g, _, _ = get_heatmap_data(field='Farm_8', is_grazing=True, stats='nse')
heat_map_data_8_g = heat_map_data_8_g[heat_map_data_8_g.columns[::-1]]

fig, axes = plt.subplots(2, 2, figsize=(15, 15), sharex=False, sharey=True)
axes[0, 0] = plot_heat_map(heat_map_data_1_n, custom_delta=0.5, ax=axes[0, 0], x_label=None,
                           y_label=''r'$\mu+\Delta{\sigma}$', title_str="WRE1", n_y_ticks=15)
axes[1, 0] = plot_heat_map(heat_map_data_1_g, custom_delta=0.5, ax=axes[1, 0], x_label=None,
                           y_label=''r'$\mu+\Delta{\sigma}$', title_str=None, n_y_ticks=15)
axes[0, 1] = plot_heat_map(heat_map_data_8_n, custom_delta=0.5, ax=axes[0, 1], x_label='Year', y_label=None,
                           title_str='WRE8', n_y_ticks=15)
axes[1, 1] = plot_heat_map(heat_map_data_8_g, custom_delta=0.5, ax=axes[1, 1], x_label='Year', y_label=None,
                           title_str=None, n_y_ticks=15)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Heatmap_Uncertainty_NSE_GLUE.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Heatmap_Uncertainty_NSE_GLUE.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Heatmap_Uncertainty_NSE_GLUE.jpeg'), dpi=600, bbox_inches="tight")
plt.show()

#
# # scaled up
#
# heat_map_data_1_n, _, _ = get_heatmap_data(field='Farm_1', is_grazing=False, stats='nse')
# heat_map_data_1_n = heat_map_data_1_n[heat_map_data_1_n.columns[::-1]]
# heat_map_data_1_g, _, _ = get_heatmap_data(field='Farm_1', is_grazing=True, stats='nse')
# heat_map_data_1_g = heat_map_data_1_g[heat_map_data_1_g.columns[::-1]]
# heat_map_data_8_n, _, _ = get_heatmap_data(field='Farm_8', is_grazing=False, stats='nse')
# heat_map_data_8_n = heat_map_data_8_n[heat_map_data_8_n.columns[::-1]]
# heat_map_data_8_g, _, _ = get_heatmap_data(field='Farm_8', is_grazing=True, stats='nse')
# heat_map_data_8_g = heat_map_data_8_g[heat_map_data_8_g.columns[::-1]]
#
# old_columns = heat_map_data_1_n.columns
# new_columns = old_columns[np.arange(0, len(old_columns), 400)]
# heat_map_data_1_n = heat_map_data_1_n[new_columns]
# fig, axes = plt.subplots(2, 2, figsize=(16, 20), sharex=False, sharey=True)
# sns.heatmap(data=heat_map_data_1_n.T, ax=axes[0, 0],
#             vmin=np.min([heat_map_data_1_n.min().min(), heat_map_data_1_g.min().min()]),
#             vmax=np.max([heat_map_data_1_n.max().max(), heat_map_data_1_g.max().max()]),
#             cmap=sns.color_palette("coolwarm"),
#             cbar_kws={'label': 'Log likelihood'})
# axes[0, 0].set_ylabel("WRE1", fontsize=16)
# axes[0, 0].set_title("Without grazing", fontsize=16)
#
# old_columns = heat_map_data_1_g.columns
# new_columns = old_columns[np.arange(0, len(old_columns), 400)]
# heat_map_data_1_g = heat_map_data_1_g[new_columns]
# sns.heatmap(data=heat_map_data_1_g.T, ax=axes[0, 1],
#             vmin=np.min([heat_map_data_1_n.min().min(), heat_map_data_1_g.min().min()]),
#             vmax=np.max([heat_map_data_1_n.max().max(), heat_map_data_1_g.max().max()]),
#             cmap=sns.color_palette("coolwarm"),
#             cbar_kws={'label': 'Log likelihood'})
# axes[0, 1].set_title("With grazing", fontsize=16)
#
# old_columns = heat_map_data_8_n.columns
# new_columns = ['3.000', '2.600', '2.200', '1.800', '1.400', '1.000', '0.600',
#                '0.200', '-0.200', '-0.600', '-1.000', '-1.400', '-1.800', '-2.200', '-2.600', '-3.000']
# heat_map_data_8_n = heat_map_data_8_n[new_columns]
# sns.heatmap(data=heat_map_data_8_n.T, ax=axes[1, 0],
#             vmin=np.min([heat_map_data_8_n.min().min(), heat_map_data_8_g.min().min()]),
#             vmax=np.max([heat_map_data_8_n.max().max(), heat_map_data_8_g.max().max()]),
#             cmap=sns.color_palette("coolwarm"),
#             cbar_kws={'label': 'Log likelihood'})
# axes[1, 0].set_ylabel("WRE8", fontsize=16)
# axes[1, 0].set_xlabel("Year", fontsize=16)
#
# old_columns = heat_map_data_8_g.columns
# new_columns = old_columns[np.arange(0, len(old_columns), 400)]
# heat_map_data_8_g = heat_map_data_8_g[new_columns]
# sns.heatmap(data=heat_map_data_8_g.T, ax=axes[1, 1],
#             vmin=np.min([heat_map_data_8_n.min().min(), heat_map_data_8_g.min().min()]),
#             vmax=np.max([heat_map_data_8_n.max().max(), heat_map_data_8_g.max().max()]),
#             cmap=sns.color_palette("coolwarm"),
#             cbar_kws={'label': 'Log likelihood'})
# axes[1, 1].set_xlabel("Year", fontsize=16)
# plt.tight_layout()
# plt.savefig(os.path.join(graph_dir, 'Heatmap_Uncertainty_Partial_NSE_GLUE.png'),
#             dpi=600, bbox_inches="tight")
# plt.show()
#
# fig, axes = plt.subplots(2, 2, figsize=(16, 20), sharex=False, sharey=True)
# sns.heatmap(data=heat_map_data_1_n.T, ax=axes[0, 0],
#             vmin=-0.5, vmax=0.6,
#             cmap=sns.color_palette("coolwarm"),
#             cbar_kws={'label': 'Log likelihood'})
# axes[0, 0].set_ylabel("WRE1", fontsize=16)
# axes[0, 0].set_title("Without grazing", fontsize=16)
#
# sns.heatmap(data=heat_map_data_1_g.T, ax=axes[0, 1],
#             vmin=-0.5, vmax=0.6,
#             cmap=sns.color_palette("coolwarm", as_cmap=True),
#             cbar_kws={'label': 'Log likelihood'})
# axes[0, 1].set_title("With grazing", fontsize=16)
#
# sns.heatmap(data=heat_map_data_8_n.T, ax=axes[1, 0],
#             vmin=-0.5, vmax=0.6,
#             cmap=sns.color_palette("coolwarm"),
#             cbar_kws={'label': 'Log likelihood'})
# axes[1, 0].set_ylabel("WRE8", fontsize=16)
# axes[1, 0].set_xlabel("Year", fontsize=16)
#
# sns.heatmap(data=heat_map_data_8_g.T, ax=axes[1, 1],
#             vmin=-0.5, vmax=0.6,
#             cmap=sns.color_palette("coolwarm"),
#             cbar_kws={'label': 'Log likelihood'})
# axes[1, 1].set_xlabel("Year", fontsize=16)
# plt.tight_layout()
# plt.savefig(os.path.join(graph_dir, 'Heatmap_Uncertainty_Partial_NSE_GLUE_short.png'),
#             dpi=600, bbox_inches="tight")
# plt.show()
