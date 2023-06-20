# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 11:45:04 2023

@author: Mahesh.Maskey
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import os
from utility import get_data_melt

warnings.filterwarnings('ignore')
print('\014')
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)


def get_data_month(site, scenario, attribute, site_label, scenario_label):
    file = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/daily_{attribute}.csv'
    data = pd.read_csv(file, index_col=0)
    data.index = pd.to_datetime(data.index)
    data_month = data.resample('M').sum()
    data_month.insert(0, 'Year', data_month.index.year)
    data_month.insert(1, 'Month', data_month.index.month)
    data_month.Month[data_month.Month == 1] = 'Jan'
    data_month.Month[data_month.Month == 2] = 'Feb'
    data_month.Month[data_month.Month == 3] = 'Mar'
    data_month.Month[data_month.Month == 4] = 'Apr'
    data_month.Month[data_month.Month == 5] = 'May'
    data_month.Month[data_month.Month == 6] = 'Jun'
    data_month.Month[data_month.Month == 7] = 'Jul'
    data_month.Month[data_month.Month == 8] = 'Aug'
    data_month.Month[data_month.Month == 9] = 'Sep'
    data_month.Month[data_month.Month == 10] = 'Oct'
    data_month.Month[data_month.Month == 11] = 'Nov'
    data_month.Month[data_month.Month == 12] = 'Dec'
    data_melt = pd.melt(data_month, id_vars=['Year', 'Month'])
    data_melt['Watershed'] = site_label
    data_melt['Operation'] = scenario_label
    return data_melt


data_non_grazing_1 = get_data_month(site='Farm_1', scenario='non_grazing', attribute='DPRK', site_label='WRE1',
                                    scenario_label='Without grazing')
data_grazing_1 = get_data_month(site='Farm_1', scenario='grazing', attribute='DPRK', site_label='WRE1',
                                scenario_label='With grazing')
data_non_grazing_8 = get_data_month(site='Farm_8', scenario='non_grazing', attribute='DPRK', site_label='WRE8',
                                    scenario_label='Without grazing')
data_grazing_8 = get_data_month(site='Farm_8', scenario='grazing', attribute='DPRK', site_label='WRE8',
                                scenario_label='With grazing')
data_bar_DPRK1 = pd.concat([data_non_grazing_1, data_grazing_1], axis=0)
data_bar_DPRK8 = pd.concat([data_non_grazing_8, data_grazing_8], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=True, tight_layout=True)
sns.barplot(data=data_bar_DPRK1, x='Month', y='value', ci=None, hue='Operation', ax=axes[0])
axes[0].legend_.remove()
axes[0].set_title('WRE1 grassland')
axes[0].set_ylabel('Deep percolation, mm')
axes[0].set_xlabel('Month')
axes[0].grid(True)
sns.barplot(data=data_bar_DPRK8, x='Month', y='value', ci=None, hue='Operation', ax=axes[1])
axes[1].set_title('WRE8 cropland')
axes[1].legend_.remove()
axes[1].set_ylabel('Deep percolation, mm')
axes[1].set_xlabel('Watershed')
axes[1].grid(True)
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_barplot_deep_percolation.png'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=False, sharey=True, tight_layout=True)
sns.boxplot(data=data_bar_DPRK1, x='Month', y='value', hue='Operation', ax=axes[0], showfliers=False)
axes[0].legend_.remove()
axes[0].set(yscale="log")
axes[0].set_title('WRE1 grassland')
axes[0].set_ylabel('Deep percolation, mm')
axes[0].set_xlabel('Month')
axes[0].grid(True)
sns.boxplot(data=data_bar_DPRK8, x='Month', y='value', hue='Operation', ax=axes[1], showfliers=False)
axes[1].set_title('WRE8 cropland')
axes[1].set(yscale="log")
axes[1].legend_.remove()
axes[1].set_ylabel('Deep percolation, mm')
axes[1].set_xlabel('Watershed')
axes[1].grid(True)
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_boxplot_deep_percolation.png'),
            dpi=600, bbox_inches="tight")
plt.show()
