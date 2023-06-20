# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 11:13:05 2023

@author: Mahesh.Maskey
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
from utility import read_obs_data
import os

warnings.filterwarnings('ignore')
print('\014')
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)


def plot_uncertainty_ts(site, scenario, ax, title):
    file_WYLD_1_ng = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/Daily_WYLD.csv'
    data = pd.read_csv(file_WYLD_1_ng, index_col=0)
    data.index = pd.to_datetime(data.index)
    cal_data = data[data.Stage == 'Calibration']
    val_data = data[data.Stage == 'Validation']
    cal_start_date = cal_data.index[0]
    # cal_end_date = cal_data.index[-1]
    val_start_date = val_data.index[0]
    val_end_date = val_data.index[-1]
    data_month = data.resample('M').sum()
    mean_data = data_month.mean(axis=1)
    std_data = data_month.std(axis=1)
    min_data = data_month.min(axis=1)
    max_data = data_month.max(axis=1)

    data_obs = read_obs_data(site=site, attribute='runoff')
    data_obs.index = pd.to_datetime(data_obs.index)
    obs_data = data_obs[(data_obs.index >= cal_start_date) & (data_obs.index <= val_end_date)]
    obs_month = obs_data.resample('M').sum()

    file_best = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/daily_outlet_best.csv'
    data_best = pd.read_csv(file_best, index_col=0)
    data_best.index = pd.to_datetime(data_best.index)
    data_best = pd.DataFrame(data_best['WYLD'])
    best_data = data_best[(data_best.index >= cal_start_date) & (data_best.index <= val_end_date)]
    best_month = best_data.resample('M').sum()

    sns.lineplot(x=mean_data.index, y=mean_data.values, color='b', ax=ax, label='Mean')
    ax.fill_between(mean_data.index, mean_data.values - std_data.values, mean_data.values + std_data.values, color='k',
                    alpha=0.25, label=r'$\mu \pm \sigma$')
    ax.fill_between(mean_data.index, min_data.values, max_data.values, color='g', alpha=0.25, label='Bounds')
    sns.lineplot(x=obs_month.index, y=obs_month.runoff, color='r', label='Observed', ax=ax)
    sns.lineplot(x=best_month.index, y=best_month.WYLD, color='k', label='Best solution', ax=ax)
    # ax.axvline(x=val_start_date, ymin=0, ymax=max_data.max(), c="black", ls='--')
    ax.grid(True)
    ax.set_ylabel('Runoff, mm', fontsize=16)
    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylim((0, max_data.max()))
    ax.set_xlim(mean_data.index[0], mean_data.index[-1])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(title)
    return


fig, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_uncertainty_ts(site='Farm_1', scenario='non_grazing', ax=axes[0], title='Without grazing')
axes[0].legend_.remove()
plot_uncertainty_ts(site='Farm_1', scenario='grazing', ax=axes[1], title='With grazing')
axes[1].legend_.remove()
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_runoff_ts.png'),  dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_runoff_ts.pdf'),  dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_1_runoff_ts.jpeg'),  dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_uncertainty_ts(site='Farm_8', scenario='non_grazing', ax=axes[0], title='Without grazing')
axes[0].legend_.remove()
plot_uncertainty_ts(site='Farm_8', scenario='grazing', ax=axes[1], title='With grazing')
axes[1].legend_.remove()
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_runoff_ts.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_runoff_ts.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_Farm_8_runoff_ts.jpeg'), dpi=600, bbox_inches="tight")
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False)
plot_uncertainty_ts(site='Farm_1', scenario='non_grazing', ax=axes[0, 0], title='WRE1: Grassland')
axes[0, 0].set_xlabel('')
axes[0, 0].legend_.remove()
plot_uncertainty_ts(site='Farm_1', scenario='grazing', ax=axes[1, 0], title='')
axes[1, 0].legend_.remove()

plot_uncertainty_ts(site='Farm_8', scenario='non_grazing', ax=axes[0, 1], title='WRE8: Cropland')
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('')
axes[0, 1].legend_.remove()
plot_uncertainty_ts(site='Farm_8', scenario='grazing', ax=axes[1, 1], title='')
axes[1, 1].set_ylabel('')
axes[1, 1].legend_.remove()
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_runoff_ts.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_runoff_ts.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_runoff_ts.jpeg'), dpi=600, bbox_inches="tight")
plt.show()