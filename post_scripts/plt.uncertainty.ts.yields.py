# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:13:05 2023

@author: Mahesh.Maskey
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from utility import get_range, find_id
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
    data_mean = data.loc[:, str(id_mean)]

    mean_data = data.mean(axis=1)
    max_data = data.max(axis=1)
    min_data = data.min(axis=1)

    max_data_sigma = data_sigma.max(axis=1)
    min_data_sigma = data_sigma.min(axis=1)

    file_best = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/annual_best.csv'
    data_best = pd.read_csv(file_best, index_col=0)
    data_best = pd.DataFrame(data_best[attribute])
    data_best = data_best[(data_best.index >= data.index[0]) & (data_best.index <= data.index[-1])]

    sns.lineplot(x=data_mean.index, y=data_mean.values, color='b', ax=ax, label='Mean')
    ax.fill_between(mean_data.index, min_data_sigma, max_data_sigma, color='y', alpha=0.25, label=r'$\mu \pm \sigma$')
    ax.fill_between(mean_data.index, min_data, max_data, color='k',
                    alpha=0.25, label=r'$\mu \pm 3\sigma$')
    sns.lineplot(x=data_best.index, y=data_best[attribute], color='k', label='Best solution', ax=ax)
    ax.grid(True)
    ax.set_ylabel(y_lab, fontsize=16)
    ax.set_xlabel(x_lab, fontsize=16)
    # ax.set_ylim((0, max_data.max()))
    ax.set_xlim(mean_data.index[0], mean_data.index[-1])
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    return data, (min_data.min(), max_data.max()), data_best


fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=False)
_, lim1n, _ = plot_uncertainty_ts(site='Farm_1', scenario='non_grazing', attribute='YLDF', x_lab='',
                                  y_lab='Forage yield, t/ha', ax=axes[0, 0], title='Without grazing')
axes[0, 0].legend_.remove()
_, lim1g, _ = plot_uncertainty_ts(site='Farm_1', scenario='grazing', attribute='YLDF', x_lab='',
                                  y_lab='', ax=axes[0, 1], title='With grazing')
axes[0, 1].legend_.remove()
min_pk1 = np.max([lim1n[0], lim1g[0]])
max_pk1 = np.max([lim1n[1], lim1g[1]])
axes[0, 0].set_ylim(min_pk1, max_pk1)
axes[0, 1].set_ylim(min_pk1, max_pk1)
del min_pk1, max_pk1
del lim1n, lim1g
_, lim1n, _ = plot_uncertainty_ts(site='Farm_8', scenario='non_grazing', attribute='YLDG', x_lab='Year',
                                  y_lab='Crop yield, t/ha', ax=axes[1, 0], title='')
axes[1, 0].legend_.remove()
_, lim1g, _ = plot_uncertainty_ts(site='Farm_8', scenario='grazing', attribute='YLDG', x_lab='Year',
                                  y_lab='', ax=axes[1, 1], title='')
axes[1, 1].legend_.remove()
min_pk1 = np.max([lim1n[0], lim1g[0]])
max_pk1 = np.max([lim1n[1], lim1g[1]])
axes[1, 0].set_ylim(min_pk1, max_pk1)
axes[1, 1].set_ylim(min_pk1, max_pk1)
del min_pk1, max_pk1
del lim1n, lim1g
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_yield_ts.png'), dpi=600, bbox_inches="tight")
plt.show()
