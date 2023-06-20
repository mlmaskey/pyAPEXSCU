# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:15:08 2023

@author: Mahesh.Maskey
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from utility import get_stress
from matplotlib.ticker import (MultipleLocator)

warnings.filterwarnings('ignore')
print('\014')
# daily based
# Farm 1 without grazing
# get outlet runoff


# data for drought stress
_, _, df_BIOM_1_n = get_stress(site='Farm_1', is_graze=False,
                               attribute='runoff', location='annual',
                               match_scale='annual', scale='daily',
                               metric='OF', var='BIOM', management='Without grazing')
_, _, df_BIOM_1_g = get_stress(site='Farm_1', is_graze=True,
                               attribute='runoff', location='annual',
                               match_scale='annual', scale='daily',
                               metric='OF', var='BIOM', management='With grazing')
_, _, df_BIOM_8_n = get_stress(site='Farm_8', is_graze=False,
                               attribute='runoff', location='annual',
                               match_scale='annual', scale='daily',
                               metric='OF', var='BIOM', management='Without grazing')
_, _, df_BIOM_8_g = get_stress(site='Farm_8', is_graze=True,
                               attribute='runoff', location='annual',
                               match_scale='annual', scale='daily',
                               metric='OF', var='BIOM', management='With grazing')
df_WRE1_BIOM = pd.concat([df_BIOM_1_n, df_BIOM_1_g], axis=0)
df_WRE1_BIOM['Watershed'] = 'WRE1'
df_WRE8_BIOM = pd.concat([df_BIOM_8_n, df_BIOM_8_g], axis=0)
df_WRE8_BIOM['Watershed'] = 'WRE8'
df_WRE_BIOM = pd.concat([df_WRE1_BIOM, df_WRE8_BIOM], axis=0)
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=False, sharex=False)
sns.lineplot(data=df_WRE1_BIOM, x=df_WRE1_BIOM.index.astype(int), y='BIOM', hue='Operation', palette=['blue', 'black'],
             style="Operation", ax=axes[0])
axes[0].set_ylabel('Biomass, t/ha', fontsize=16)
axes[0].set_xlabel('Year', fontsize=16)
axes[0].set_title('WRE1: Grassland', fontsize=18)
axes[0].grid(True)
axes[0].legend(ncol=3)
axes[0].tick_params(axis='x', labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)
axes[0].set_xticks(np.arange(np.min(df_WRE1_BIOM.index),  np.max(df_WRE1_BIOM.index)+1, 1))
axes[0].set_xlim(np.min(df_WRE1_BIOM.index), np.max(df_WRE1_BIOM.index))

sns.lineplot(data=df_WRE8_BIOM, x=df_WRE8_BIOM.index.astype(int), y='BIOM', hue='Operation', palette=['blue', 'black'],
             style="Operation", ax=axes[1])
axes[1].set_ylabel('Biomass, t/ha', fontsize=16)
axes[1].set_xlabel('Year', fontsize=16)
axes[1].set_title('WRE8: Cropland', fontsize=18)
axes[1].grid(True)
axes[1].legend_.remove()
axes[1].tick_params(axis='x', labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)
axes[1].set_xticks(np.arange(np.min(df_WRE8_BIOM.index),  np.max(df_WRE8_BIOM.index)+1, 1))
axes[1].set_xlim(np.min(df_WRE8_BIOM.index), np.max(df_WRE8_BIOM.index))
plt.tight_layout()

plt.savefig('../post_analysis/Figures/Calibration_Farm_1&8annual_biomass_daily_OF.png', dpi=600, bbox_inches="tight")

