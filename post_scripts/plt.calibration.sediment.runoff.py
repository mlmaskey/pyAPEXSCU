# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:15:08 2023

@author: Mahesh.Maskey
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from utility import get_stress
from utility import get_obs_data

warnings.filterwarnings('ignore')
print('\014')
# daily based
# Farm 1 without grazing
# get outlet runoff


# data for drought stress
_, _, df_YSD_1_n = get_stress(site='Farm_1', is_graze=False,
                              attribute='runoff', location='daily_outlet',
                              match_scale='daily', scale='daily',
                              metric='OF', var='YSD', management='Without grazing')
df_YSD_1_n4bar = df_YSD_1_n.resample('Y').sum()
df_YSD_1_n4bar['Operation'] = df_YSD_1_n.Operation
_, _, df_YSD_1_g = get_stress(site='Farm_1', is_graze=True,
                              attribute='runoff', location='daily_outlet',
                              match_scale='daily', scale='daily',
                              metric='OF', var='YSD', management='With grazing')
df_YSD_1_g4bar = df_YSD_1_g.resample('Y').sum()
df_YSD_1_g4bar['Operation'] = df_YSD_1_g.Operation
_, _, df_YSD_8_n = get_stress(site='Farm_8', is_graze=False,
                              attribute='runoff', location='daily_outlet',
                              match_scale='daily', scale='daily',
                              metric='OF', var='YSD', management='Without grazing')
df_YSD_8_n4bar = df_YSD_8_n.resample('Y').sum()
df_YSD_8_n4bar['Operation'] = df_YSD_8_n.Operation
_, _, df_YSD_8_g = get_stress(site='Farm_8', is_graze=True,
                              attribute='runoff', location='daily_outlet',
                              match_scale='daily', scale='daily',
                              metric='OF', var='YSD', management='With grazing')
df_YSD_8_g4bar = df_YSD_8_g.resample('Y').sum()
df_YSD_8_g4bar['Operation'] = df_YSD_8_g.Operation

# import observed data
df_obs_sediment_1 = get_obs_data(site='Farm_1',
                                 attribute='sediment',
                                 factor=0.1)
df_obs_sediment_1.columns = ['YSD', 'Operation']
df_obs_sediment_8 = get_obs_data(site='Farm_8',
                                 attribute='sediment',
                                 factor=0.001)
df_obs_sediment_8.columns = ['YSD', 'Operation']

df_WRE1_YSD = pd.concat([df_YSD_1_n4bar, df_YSD_1_g4bar], axis=0)
df_WRE1_sediment = pd.concat([df_WRE1_YSD, df_obs_sediment_1], axis=0)
df_WRE1_sediment['Watershed'] = 'WRE1'
df_WRE8_YSD = pd.concat([df_YSD_8_n4bar, df_YSD_8_g4bar], axis=0)
df_WRE8_sediment = pd.concat([df_WRE8_YSD, df_obs_sediment_8], axis=0)

df_WRE8_sediment['Watershed'] = 'WRE8'
df_WRE_sediment = pd.concat([df_WRE1_sediment, df_WRE8_sediment], axis=0)

_, _, df_RUS2_1_n = get_stress(site='Farm_1', is_graze=False,
                               attribute='runoff', location='daily_outlet',
                               match_scale='daily', scale='daily',
                               metric='OF', var='RUS2', management='Without grazing')
df_RUS2_1_n4bar = df_RUS2_1_n.resample('Y').sum()
df_RUS2_1_n4bar['Operation'] = df_RUS2_1_n.Operation
_, _, df_RUS2_1_g = get_stress(site='Farm_1', is_graze=True,
                               attribute='runoff', location='daily_outlet',
                               match_scale='daily', scale='daily',
                               metric='OF', var='RUS2', management='With grazing')
df_RUS2_1_g4bar = df_RUS2_1_g.resample('Y').sum()
df_RUS2_1_g4bar['Operation'] = df_RUS2_1_g.Operation
_, _, df_RUS2_8_n = get_stress(site='Farm_8', is_graze=False,
                               attribute='runoff', location='daily_outlet',
                               match_scale='daily', scale='daily',
                               metric='OF', var='RUS2', management='Without grazing')
df_RUS2_8_n4bar = df_RUS2_8_n.resample('Y').sum()
df_RUS2_8_n4bar['Operation'] = df_RUS2_8_n.Operation
_, _, df_RUS2_8_g = get_stress(site='Farm_8', is_graze=True,
                               attribute='runoff', location='daily_outlet',
                               match_scale='daily', scale='daily',
                               metric='OF', var='RUS2', management='With grazing')
df_RUS2_8_g4bar = df_RUS2_8_g.resample('Y').sum()
df_RUS2_8_g4bar['Operation'] = df_RUS2_8_g.Operation
# import obeserved data
df_obs_sediment_1.columns = ['RUS2', 'Operation']
df_obs_sediment_8.columns = ['RUS2', 'Operation']
df_WRE1_RUS2 = pd.concat([df_RUS2_1_n4bar, df_RUS2_1_g4bar], axis=0)
df_WRE1_erosion = pd.concat([df_WRE1_RUS2, df_obs_sediment_1], axis=0)
df_WRE1_erosion['Watershed'] = 'WRE1'
df_WRE8_RUS2 = pd.concat([df_RUS2_8_n4bar, df_RUS2_8_g4bar], axis=0)
df_WRE8_erosion = pd.concat([df_WRE8_RUS2, df_obs_sediment_8], axis=0)

df_WRE8_erosion['Watershed'] = 'WRE8'
df_WRE_erosion = pd.concat([df_WRE1_erosion, df_WRE8_erosion], axis=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
sns.barplot(data=df_WRE_sediment, x='Watershed', y='YSD', hue='Operation', palette=sns.color_palette("Set2"), ci=None,
            ax=axes[0])
axes[0].grid(True)
axes[0].set(xlabel='Watershed', ylabel='Sediment yield, t/ha')
axes[0].legend([], [], frameon=False)

sns.barplot(data=df_WRE_erosion, x='Watershed', y='RUS2', hue='Operation', palette=sns.color_palette("Set2"), ci=None,
            ax=axes[1], errwidth=0)
axes[1].grid(True)
axes[1].set(xlabel='Watershed', ylabel='Soil erosion, t/ha')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=4)

plt.tight_layout()
plt.savefig('../post_analysis/Figures/Calibration_Farm_1&8annual_Sediment_erosion_daily_OF.png', dpi=600,
            bbox_inches="tight")
