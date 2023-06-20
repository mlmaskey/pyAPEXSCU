# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 10:23:08 2023

@author: Mahesh.Maskey
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
print('\014')
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)


def get_data_melt(site, scenario, attribute, site_label, scenario_label):
    file = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/annual_{attribute}.csv'
    data = pd.read_csv(file, index_col=0)
    data.insert(0, 'Year', data.index)
    if 'Stage' in data.columns:
        data_melt = pd.melt(data, id_vars=['Year', 'Stage'])
    else:
        data_melt = pd.melt(data, id_vars=['Year'])
    data_melt['Watershed'] = site_label
    data_melt['Operation'] = scenario_label
    return data_melt


def box_plot_attribute(data, ax, y_label=None, title_str=None, show_fliers=False):
    sns.boxplot(data=data, x='Year', y='value', hue='Operation', ax=ax, showfliers=show_fliers)
    ax.legend_.remove()
    ax.set_title(title_str, fontsize=18)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_xlabel('Year', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)
    return ax


data_non_grazing_1 = get_data_melt(site='Farm_1', scenario='non_grazing', attribute='BIOM', site_label='WRE1',
                                   scenario_label='Without grazing')
data_grazing_1 = get_data_melt(site='Farm_1', scenario='grazing', attribute='BIOM', site_label='WRE1',
                               scenario_label='With grazing')
data_non_grazing_8 = get_data_melt(site='Farm_8', scenario='non_grazing', attribute='BIOM', site_label='WRE8',
                                   scenario_label='Without grazing')
data_grazing_8 = get_data_melt(site='Farm_8', scenario='grazing', attribute='BIOM', site_label='WRE8',
                               scenario_label='With grazing')
data_farm_biom_1 = pd.concat([data_non_grazing_1, data_grazing_1], axis=0)
data_farm_biom_8 = pd.concat([data_non_grazing_8, data_grazing_8], axis=0)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False, sharey=True, tight_layout=True)
axes[0] = box_plot_attribute(data_farm_biom_1, ax=axes[0], y_label='Biomass, t/ha', title_str='WRE1 - Grassland',
                             show_fliers=False)
axes[1] = box_plot_attribute(data_farm_biom_8, ax=axes[1], y_label='Biomass, t/ha', title_str='WRE8 - Cropland',
                             show_fliers=False)
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_boxplot_biom.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_boxplot_biom.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_boxplot_biom.jpeg'), dpi=600, bbox_inches="tight")
plt.show()

data_non_grazing_1 = get_data_melt(site='Farm_1', scenario='non_grazing', attribute='YLDF', site_label='WRE1',
                                   scenario_label='Without grazing')
data_grazing_1 = get_data_melt(site='Farm_1', scenario='grazing', attribute='YLDF', site_label='WRE1',
                               scenario_label='With grazing')
data_non_grazing_8 = get_data_melt(site='Farm_8', scenario='non_grazing', attribute='YLDG', site_label='WRE8',
                                   scenario_label='Without grazing')
data_grazing_8 = get_data_melt(site='Farm_8', scenario='grazing', attribute='YLDG', site_label='WRE8',
                               scenario_label='With grazing')

data_farm_yld_1 = pd.concat([data_non_grazing_1, data_grazing_1], axis=0)
data_farm_yld_8 = pd.concat([data_non_grazing_8, data_grazing_8], axis=0)
# noinspection PyRedeclaration
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False, sharey=True, tight_layout=True)
axes[0] = box_plot_attribute(data_farm_yld_1, ax=axes[0], y_label='Forage yield, t/ha', title_str='WRE1 - Grassland',
                             show_fliers=False)
axes[1] = box_plot_attribute(data_farm_yld_8, ax=axes[1], y_label='Crop yield, t/ha', title_str='WRE8 - Cropland',
                             show_fliers=False)
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_boxplot_yield.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_boxplot_yield.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_boxplot_yield.jpeg'), dpi=600, bbox_inches="tight")
plt.show()
