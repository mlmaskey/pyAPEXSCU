# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:38:07 2023

@author: Mahesh.Maskey
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 13:29:34 2022

@author: Mahesh.Maskey
"""

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
_, _, df_ws_1_n = get_stress(site='Farm_1', is_graze=False,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='WS', management='Without grazing')

_, _, df_ws_1_g = get_stress(site='Farm_1', is_graze=True,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='WS', management='With grazing')

_, _, df_ws_8_n = get_stress(site='Farm_8', is_graze=False,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='WS', management='Without grazing')

_, _, df_ws_8_g = get_stress(site='Farm_8', is_graze=True,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='WS', management='With grazing')

df_WRE1_ws = pd.concat([df_ws_1_n, df_ws_1_g], axis=0)
df_WRE1_ws.insert(0, 'YR', df_WRE1_ws.index)
df_WRE1_ws['Watershed'] = 'WRE1'

df_WRE8_ws = pd.concat([df_ws_8_n, df_ws_8_g], axis=0)
df_WRE8_ws.insert(0, 'YR', df_WRE8_ws.index)
df_WRE8_ws['Watershed'] = 'WRE8'
df_WRE_ws = pd.concat([df_WRE1_ws, df_WRE8_ws], axis=0)

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)

df_ws_1_g['YR'] = df_ws_1_g.index.values
df_ws_1_n['YR'] = df_ws_1_n.index.values
df_ws_8_n['YR'] = df_ws_8_n.index.values
df_ws_8_g['YR'] = df_ws_8_g.index.values
sns.lineplot(data=df_ws_1_n, x='YR', y='WS', color='g', ax=axes[0, 0], label='WRE1 Without grazing')
sns.lineplot(data=df_ws_1_g, x='YR', y='WS', color='k', ax=axes[0, 0], label='WRE1 With grazing')
sns.lineplot(data=df_ws_8_n, x='YR', y='WS', color='b', ax=axes[0, 0], label='WRE8 With grazing')
sns.lineplot(data=df_ws_8_g, x='YR', y='WS', color='r', ax=axes[0, 0], label='WRE8 With grazing')
axes[0, 0].axvline(x=1995, color='b')
axes[0, 0].axvline(x=2000, color='b')
axes[0, 0].axvspan(1995, 2000, alpha=0.5, color='gray')
axes[0, 0].xaxis.set_major_locator(MultipleLocator(5))
axes[0, 0].set_xlim(df_ws_8_n.index[0], df_ws_8_n.index[-1])
axes[0, 0].set_title('Drought Stress')
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Stress in days')
axes[0, 0].get_legend().remove()
axes[0, 0].grid(True)

# data for Temperature stress

_, _, df_ts_1_n = get_stress(site='Farm_1', is_graze=False,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='TS', management='Without grazing')

_, _, df_ts_1_g = get_stress(site='Farm_1', is_graze=True,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='TS', management='With grazing')

_, _, df_ts_8_n = get_stress(site='Farm_8', is_graze=False,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='TS', management='Without grazing')

_, _, df_ts_8_g = get_stress(site='Farm_8', is_graze=True,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='TS', management='With grazing')

df_WRE1_ts = pd.concat([df_ts_1_n, df_ts_1_g], axis=0)
df_WRE1_ts.insert(0, 'YR', df_WRE1_ts.index)
df_WRE1_ts['Watershed'] = 'WRE1'

df_WRE8_ts = pd.concat([df_ts_8_n, df_ts_8_g], axis=0)
df_WRE8_ts.insert(0, 'YR', df_WRE8_ts.index)
df_WRE8_ts['Watershed'] = 'WRE8'
df_WRE_ts = pd.concat([df_WRE1_ts, df_WRE8_ts], axis=0)

df_ts_1_n['YR'] = df_ts_1_n.index.values
df_ts_1_g['YR'] = df_ts_1_g.index.values
df_ts_8_n['YR'] = df_ts_8_n.index.values
df_ts_8_g['YR'] = df_ts_8_g.index.values

sns.lineplot(data=df_ts_1_n, x='YR', y='TS', color='g', ax=axes[0, 1], label='WRE1 Without grazing')
sns.lineplot(data=df_ts_1_g, x='YR', y='TS', color='k', ax=axes[0, 1], label='WRE1 With grazing')
sns.lineplot(data=df_ts_8_n, x='YR', y='TS', color='b', ax=axes[0, 1], label='WRE8 Without grazing')
sns.lineplot(data=df_ts_8_g, x='YR', y='TS', color='r', ax=axes[0, 1], label='WRE8 With grazing')
axes[0, 1].axvline(x=1995, color='b')
axes[0, 1].axvline(x=2000, color='b')
axes[0, 1].axvspan(1995, 2000, alpha=0.5, color='gray')
axes[0, 1].xaxis.set_major_locator(MultipleLocator(5))
axes[0, 1].set_xlim(df_ts_8_n.index[0], df_ts_8_n.index[-1])
axes[0, 1].set_title('Temperature Stress')
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Stress in days')
axes[0, 1].get_legend().remove()
axes[0, 1].grid(True)

# data for Nitrogen stress

_, _, df_ns_1_n = get_stress(site='Farm_1', is_graze=False,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='NS', management='Without grazing')

_, _, df_ns_1_g = get_stress(site='Farm_1', is_graze=True,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='NS', management='With grazing')

_, _, df_ns_8_n = get_stress(site='Farm_8', is_graze=False,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='NS', management='Without grazing')

_, _, df_ns_8_g = get_stress(site='Farm_8', is_graze=True,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='NS', management='With grazing')

df_WRE1_ns = pd.concat([df_ns_1_n, df_ns_1_g], axis=0)
df_WRE1_ns.insert(0, 'YR', df_WRE1_ns.index)
df_WRE1_ns['Watershed'] = 'WRE1'

df_WRE8_ns = pd.concat([df_ns_8_n, df_ns_8_g], axis=0)
df_WRE8_ns.insert(0, 'YR', df_WRE8_ns.index)
df_WRE8_ns['Watershed'] = 'WRE8'
df_WRE_ns = pd.concat([df_WRE1_ns, df_WRE8_ns], axis=0)

df_ns_1_n['YR'] = df_ns_1_n.index.values
df_ns_1_g['YR'] = df_ns_1_g.index.values
df_ns_8_n['YR'] = df_ns_8_n.index.values
df_ns_8_g['YR'] = df_ns_8_g.index.values

sns.lineplot(data=df_ns_1_n, x='YR', y='NS', color='g', ax=axes[1, 0], label='WRE1 Without grazing')
sns.lineplot(data=df_ns_1_g, x='YR', y='NS', color='k', ax=axes[1, 0], label='WRE1 With grazing')
sns.lineplot(data=df_ns_8_n, x='YR', y='NS', color='b', ax=axes[1, 0], label='WRE8 Without grazing')
sns.lineplot(data=df_ns_8_g, x='YR', y='NS', color='r', ax=axes[1, 0], label='WRE8 With grazing')
axes[1, 0].axvline(x=1995, color='b')
axes[1, 0].axvline(x=2000, color='b')
axes[1, 0].axvspan(1995, 2000, alpha=0.5, color='gray')
axes[1, 0].xaxis.set_major_locator(MultipleLocator(5))
axes[1, 0].set_xlim(df_ns_8_n.index[0], df_ns_8_n.index[-1])
axes[1, 0].set_title('Nitrogen Stress')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Stress in days')
axes[1, 0].get_legend().remove()
axes[1, 0].grid(True)

# data for Phosphorus stress

_, _, df_ps_1_n = get_stress(site='Farm_1', is_graze=False,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='PS', management='Without grazing')

_, _, df_ps_1_g = get_stress(site='Farm_1', is_graze=True,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='PS', management='With grazing')

_, _, df_ps_8_n = get_stress(site='Farm_8', is_graze=False,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='PS', management='Without grazing')

_, _, df_ps_8_g = get_stress(site='Farm_8', is_graze=True,
                             attribute='runoff', location='annual',
                             match_scale='annual', scale='daily',
                             metric='OF', var='PS', management='With grazing')

df_WRE1_ps = pd.concat([df_ps_1_n, df_ps_1_g], axis=0)
df_WRE1_ps.insert(0, 'YR', df_WRE1_ps.index)
df_WRE1_ps['Watershed'] = 'WRE1'

df_WRE8_ps = pd.concat([df_ps_8_n, df_ps_8_g], axis=0)
df_WRE8_ps.insert(0, 'YR', df_WRE8_ps.index)
df_WRE8_ps['Watershed'] = 'WRE8'
df_WRE_ps = pd.concat([df_WRE1_ps, df_WRE8_ps], axis=0)

df_ps_1_n['YR'] = df_ps_1_n.index.values
df_ps_1_g['YR'] = df_ps_1_g.index.values
df_ps_8_n['YR'] = df_ps_8_n.index.values
df_ps_8_g['YR'] = df_ps_8_g.index.values

ax1 = sns.lineplot(data=df_ps_1_n, x='YR', y='PS', color='g', ax=axes[1, 1], label='WRE1 Without grazing')
ax2 = sns.lineplot(data=df_ps_1_g, x='YR', y='PS', color='k', ax=axes[1, 1], label='WRE1 With grazing')
ax3 = sns.lineplot(data=df_ps_8_n, x='YR', y='PS', color='b', ax=axes[1, 1], label='WRE8 Without grazing')
ax4 = sns.lineplot(data=df_ps_8_g, x='YR', y='PS', color='r', ax=axes[1, 1], label='WRE8 With grazing')
# handles, labels = [ax1, ax2, ax3, ax4].get_legend_handles_labels()
axes[1, 1].axvline(x=1995, color='b')
axes[1, 1].axvline(x=2000, color='b')
axes[1, 1].axvspan(1995, 2000, alpha=0.5, color='gray')
axes[1, 1].xaxis.set_major_locator(MultipleLocator(5))
axes[1, 1].set_xlim(df_ps_8_n.index[0], df_ps_8_n.index[-1])
axes[1, 1].set_title('Phosphorus Stress')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Stress in days')
# axes[1, 1].get_legend().remove()
axes[1, 1].grid(True)
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=6)
plt.tight_layout()
plt.savefig('../post_analysis/Figures/Calibration_line_plot_Farm_1_and_8_annual_Environmental_stress_daily.png',
            dpi=600, bbox_inches="tight")
plt.show()

# box plots
df_WRE_ws.insert(5, 'Stress', 'WS', True)
df_WRE_ws.columns = ['YR', 'value', 'Stage', 'Operation', 'Watershed', 'Stress']
df_WRE_ts.insert(5, 'Stress', 'TS', True)
df_WRE_ts.columns = ['YR', 'value', 'Stage', 'Operation', 'Watershed', 'Stress']
df_WRE_ns.insert(5, 'Stress', 'NS', True)
df_WRE_ns.columns = ['YR', 'value', 'Stage', 'Operation', 'Watershed', 'Stress']
df_WRE_ps.insert(5, 'Stress', 'PS', True)
df_WRE_ps.columns = ['YR', 'value', 'Stage', 'Operation', 'Watershed', 'Stress']
df_stress = pd.concat([df_WRE_ws, df_WRE_ts, df_WRE_ns, df_WRE_ps], axis=0)
df_stress1 = df_stress[df_stress.Watershed == 'WRE1']
df_stress8 = df_stress[df_stress.Watershed == 'WRE8']

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
sns.boxplot(data=df_stress1, x='Stress', y='value', hue='Operation', palette=sns.color_palette("Set2"), ax=axes[0])
axes[0].set_xlabel('Environmental stresses', fontsize=16)
axes[0].set_ylabel('Stress in days', fontsize=16)
axes[0].set_title('WRE1: Grassland', fontsize=18)
axes[0].grid(True)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)
axes[0].legend_.remove()

sns.boxplot(data=df_stress8, x='Stress', y='value', hue='Operation', palette=sns.color_palette("Set2"), ax=axes[1])
axes[1].set_xlabel('Environmental stresses', fontsize=16)
axes[1].set_ylabel('')
axes[1].set_title('WRE8: Cropland', fontsize=18)
axes[1].grid(True)
axes[1].legend_.remove()
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6, prop={'size': 12})
plt.savefig('../post_analysis/Figures/Box_plot_Farm_1_and_8_annual_Environmental_stress_daily.png', dpi=600,
            bbox_inches="tight")
plt.show()
