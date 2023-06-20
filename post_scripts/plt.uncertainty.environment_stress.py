# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 11:52:55 2023

@author: Mahesh.Maskey
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from utility import get_range
from utility import find_id
import os

warnings.filterwarnings('ignore')
print('\014')
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)


def plot_uncertainty_ts(site, scenario, attribute, ax, title, x_lab, y_lab):
    file = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/annual_{attribute}.csv'
    data = pd.read_csv(file, index_col=0)
    uncertainty_range = get_range(site, scenario, task="Uncertainty")
    id_mean = find_id(uncertainty_range, 0)
    id_lower, id_upper = find_id(uncertainty_range, -1), find_id(uncertainty_range, 1)

    data_sigma = data.loc[:, str(id_lower):str(id_upper)]
    data_sigma.insert(0, 'Stage', data.Stage, True)
    # cal_data = data[data.Stage=='Calibration']
    # val_data = data[data.Stage == 'Validation']
    # cal_start_date = cal_data.index[0]
    # cal_end_date = cal_data.index[-1]
    # val_start_date = val_data.index[0]
    # val_end_date = val_data.index[-1]
    data_mean = data.loc[:, str(id_mean)]

    mean_data = data.mean(axis=1)
    # std_data = data_sigma.std(axis=1)
    # ci = 3*std_data/np.sqrt(data.shape[1])
    max_data = data.max(axis=1)
    min_data = data.min(axis=1)

    max_data_sigma = data_sigma.max(axis=1)
    min_data_sigma = data_sigma.min(axis=1)

    file_best = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/annual_best.csv'
    data_best = pd.read_csv(file_best, index_col=0)
    data_best = pd.DataFrame(data_best[attribute])

    sns.lineplot(x=data_mean.index, y=data_mean.values, color='b', ax=ax, label='Mean')
    ax.fill_between(mean_data.index, min_data_sigma, max_data_sigma, color='y', alpha=0.25, label=r'$\mu \pm \sigma$')
    ax.fill_between(mean_data.index, min_data, max_data, color='k',
                    alpha=0.25, label=r'$\mu \pm 3\sigma$')
    # ax.fill_between(mean_data.index, mean_data.values-ci.values, mean_data.values+ci.values, color='k', alpha=0.25,
    # label='Confidence interval') ax.fill_between(mean_data.index, min_data.values, max_data.values, color='g',
    # alpha=0.25, label='Bounds')
    sns.lineplot(x=data_best.index, y=data_best[attribute], color='k', label='Best solution', ax=ax)
    # ax.axvline(x=val_start_date, ymin=0, ymax=max_data.max(), c="black", ls='--')
    ax.grid(True)
    ax.set_ylabel(y_lab)
    ax.set_xlabel(x_lab)
    # ax.set_ylim((0, max_data.max()))
    ax.set_xlim(mean_data.index[0], mean_data.index[-1])
    ax.set_title(title)
    return data, (min_data.min(), max_data.max()), data_best


fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=False)
_, lim1n, _ = plot_uncertainty_ts(site='Farm_1', scenario='non_grazing', attribute='WS', ax=axes[0, 0],
                                             title='WRE1, without grazing', x_lab='', y_lab='Drought stress, days')
axes[0, 0].legend_.remove()
_, lim1g, _ = plot_uncertainty_ts(site='Farm_1', scenario='grazing', attribute='WS', ax=axes[1, 0],
                                             title='WRE1, with grazing', x_lab='Year', y_lab='Drought stress, days')

axes[1, 0].legend_.remove()
min_pk1 = np.max([lim1n[0], lim1g[0]])
max_pk1 = np.max([lim1n[1], lim1g[1]])
axes[0, 0].set_ylim(min_pk1, max_pk1)
axes[1, 0].set_ylim(min_pk1, max_pk1)
del min_pk1, max_pk1
del lim1n, lim1g
_, lim8n, _ = plot_uncertainty_ts(site='Farm_8', scenario='non_grazing', attribute='WS', ax=axes[0, 1],
                                  title='WRE8, without grazing', x_lab='', y_lab='')
axes[0, 1].legend_.remove()
_, lim8g, _ = plot_uncertainty_ts(site='Farm_8', scenario='grazing', attribute='WS', ax=axes[1, 1],
                                  title='WRE8, with grazing', x_lab='Year', y_lab='')
axes[1, 1].legend_.remove()
min_pk8 = np.max([lim8n[0], lim8g[0]])
max_pk8 = np.max([lim8n[1], lim8n[1]])
axes[0, 1].set_ylim(min_pk8, max_pk8)
axes[1, 1].set_ylim(min_pk8, max_pk8)
del min_pk8, max_pk8
del lim8n, lim8g
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1&8_WS.png'), dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=False)
_, lim1n, _ = plot_uncertainty_ts(site='Farm_1', scenario='non_grazing', attribute='TS', ax=axes[0, 0],
                                  title='WRE1, without grazing', x_lab='', y_lab='Temperature stress, days')
axes[0, 0].legend_.remove()
_, lim1g, _ = plot_uncertainty_ts(site='Farm_1', scenario='grazing', attribute='TS', ax=axes[1, 0],
                                  title='WRE1, with grazing', x_lab='Year', y_lab='Temperature stress, days')
axes[1, 0].legend_.remove()
min_pk1 = np.max([lim1n[0], lim1g[0]])
max_pk1 = np.max([lim1n[1], lim1g[1]])
axes[0, 0].set_ylim(min_pk1, max_pk1)
axes[1, 0].set_ylim(min_pk1, max_pk1)
del min_pk1, max_pk1
del lim1n, lim1g
_, lim8n, _ = plot_uncertainty_ts(site='Farm_8', scenario='non_grazing', attribute='TS', ax=axes[0, 1],
                                  title='WRE8, without grazing', x_lab='', y_lab='')
axes[0, 1].legend_.remove()
_, lim8g, _ = plot_uncertainty_ts(site='Farm_8', scenario='grazing', attribute='TS', ax=axes[1, 1],
                                  title='WRE8, with grazing', x_lab='Year', y_lab='')
axes[1, 1].legend_.remove()
min_pk8 = np.max([lim8n[0], lim8g[0]])
max_pk8 = np.max([lim8n[1], lim8n[1]])
axes[0, 1].set_ylim(min_pk8, max_pk8)
axes[1, 1].set_ylim(min_pk8, max_pk8)
del min_pk8, max_pk8
del lim8n, lim8g
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1&8_TS.png'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=False)
_, lim1n, _ = plot_uncertainty_ts(site='Farm_1', scenario='non_grazing', attribute='NS', ax=axes[0, 0],
                                  title='WRE1, without grazing', x_lab='', y_lab='Nitrogen stress, days')
axes[0, 0].legend_.remove()
_, lim1g, _ = plot_uncertainty_ts(site='Farm_1', scenario='grazing', attribute='NS', ax=axes[1, 0],
                                  title='WRE1, with grazing', x_lab='Year', y_lab='Nitrogen stress, days')
axes[1, 0].legend_.remove()
min_pk1 = np.max([lim1n[0], lim1g[0]])
max_pk1 = np.max([lim1n[1], lim1g[1]])
axes[0, 0].set_ylim(min_pk1, max_pk1)
axes[1, 0].set_ylim(min_pk1, max_pk1)
del min_pk1, max_pk1
del lim1n, lim1g
_, lim8n, _ = plot_uncertainty_ts(site='Farm_8', scenario='non_grazing', attribute='NS', ax=axes[0, 1],
                                  title='WRE8, without grazing', x_lab='', y_lab='')
axes[0, 1].legend_.remove()
_, lim8g, _ = plot_uncertainty_ts(site='Farm_8', scenario='grazing', attribute='NS', ax=axes[1, 1],
                                  title='WRE8, without grazing', x_lab='Year', y_lab='')
axes[1, 1].legend_.remove()
min_pk8 = np.max([lim8n[0], lim8g[0]])
max_pk8 = np.max([lim8n[1], lim8n[1]])
axes[0, 1].set_ylim(min_pk8, max_pk8)
axes[1, 1].set_ylim(min_pk8, max_pk8)
del min_pk8, max_pk8
del lim8n, lim8g
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1&8_NS.png'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=False)
_, lim1n, _ = plot_uncertainty_ts(site='Farm_1', scenario='non_grazing', attribute='PS', ax=axes[0, 0],
                                  title='WRE1, without grazing', x_lab='', y_lab='Phosphorous stress, days')
axes[0, 0].legend_.remove()
_, lim1g, _ = plot_uncertainty_ts(site='Farm_1', scenario='grazing', attribute='PS', ax=axes[1, 0],
                                  title='WRE1, with grazing', x_lab='Year', y_lab='Phosphorous stress, days')
axes[1, 0].legend_.remove()
min_pk1 = np.max([lim1n[0], lim1g[0]])
max_pk1 = np.max([lim1n[1], lim1g[1]])
axes[0, 0].set_ylim(min_pk1, max_pk1)
axes[1, 0].set_ylim(min_pk1, max_pk1)
del min_pk1, max_pk1
del lim1n, lim1g
_, lim8n, _ = plot_uncertainty_ts(site='Farm_8', scenario='non_grazing', attribute='PS', ax=axes[0, 1],
                                  title='WRE8, without grazing', x_lab='', y_lab='')
axes[0, 1].legend_.remove()
_, lim8g, _ = plot_uncertainty_ts(site='Farm_8', scenario='grazing', attribute='PS', ax=axes[1, 1],
                                  title='WRE8, with grazing', x_lab='Year', y_lab='')
axes[1, 1].legend_.remove()
min_pk8 = np.max([lim8n[0], lim8g[0]])
max_pk8 = np.max([lim8n[1], lim8n[1]])
axes[0, 1].set_ylim(min_pk8, max_pk8)
axes[1, 1].set_ylim(min_pk8, max_pk8)
del min_pk8, max_pk8
del lim8n, lim8g
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1&8_PS.png'),
            dpi=600, bbox_inches="tight")
plt.show()
