# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:15:08 2023

@author: Mahesh.Maskey
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from utility import get_output
from utility import match_data

warnings.filterwarnings('ignore')
print('\014')


# daily based
# Farm 1 without grazing
# get outlet runoff


def plot_clustered_stacked(dfall, ax, labels=None, H="*", **kwargs):
    """Given a list of dataframes, with identical columns and index, create a clustered stacked bar plot. 
labels is a list of the names of the dataframe, used for the legend
title is a string for the title of the plot
H is the hatch used for identification of the different dataframe"""

    n_df = len(dfall)
    n_col = len(dfall[0].columns)
    n_ind = len(dfall[0].index)

    for df in dfall:  # for each data frame
        ax = df.plot(kind="bar",
                     linewidth=1,
                     stacked=True,
                     ax=ax,
                     legend=False,
                     grid=False,
                     edgecolor="black",
                     **kwargs)  # make bar plots

    h, l = ax.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))  # edited part
                rect.set_width(1 / float(n_df + 1))

    ax.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    ax.set_xticklabels(df.index, rotation=0)

    # Add invisible data to add another legend
    n = []
    # if n_df<=2:
    #     n_df=n_df+1
    for i in range(n_df):
        n.append(ax.bar(0, 0, color="gray", hatch=H * i))

    if n_col == 1:
        l1 = ax.legend(h[:n_col], l[:n_col], loc='upper center', ncol=2)
    else:
        l1 = ax.legend(h[:n_col], l[:n_col], loc='upper left', ncol=2)
    if labels is not None:
        l2 = plt.legend(n, labels, loc='upper right', ncol=2)
    ax.add_artist(l1)
    return ax


# data for yield


_, _, df_YLD_1_n = get_output(site='Farm_1', is_graze=False,
                              attribute='runoff', location='annual',
                              match_scale='annual', scale='daily',
                              metric='OF', var='YLDF')
df_YLD_1_n, _ = match_data(df_YLD_1_n, site='Farm_1', is_graze=False, attribute='runoff',
                           location='model_calibration', match_scale='annual',
                           scale='daily', metric='OF')

_, _, df_YLD_1_g = get_output(site='Farm_1', is_graze=True,
                              attribute='runoff', location='annual',
                              match_scale='annual', scale='daily',
                              metric='OF', var='YLDF')
df_YLD_1_g, _ = match_data(df_YLD_1_g, site='Farm_1', is_graze=True, attribute='runoff',
                           location='model_calibration', match_scale='annual',
                           scale='daily', metric='OF')

_, _, df_YLD_8_n = get_output(site='Farm_8', is_graze=False,
                              attribute='runoff', location='annual',
                              match_scale='annual', scale='daily',
                              metric='OF', var='YLDG')
df_YLD_8_n, _ = match_data(df_YLD_8_n, site='Farm_8', is_graze=False, attribute='runoff',
                           location='model_calibration', match_scale='annual',
                           scale='daily', metric='OF')

_, _, df_YLD_8_g = get_output(site='Farm_8', is_graze=True,
                              attribute='runoff', location='annual',
                              match_scale='annual', scale='daily',
                              metric='OF', var='YLDG')
df_YLD_8_g, _ = match_data(df_YLD_8_g, site='Farm_8', is_graze=True, attribute='runoff',
                           location='model_calibration', match_scale='annual',
                           scale='daily', metric='OF')

df1n = df_YLD_1_n[['PAST']]
df1n.columns = ['Native Prairie']
df1g = df_YLD_1_g[['PAST']]
df1g.columns = ['Native Prairie']
df8n = df_YLD_8_n[['WWHT', 'OATS']]
df8n.columns = ['Winter wheat', 'Oats']
df8g = df_YLD_8_g[['WWHT', 'OATS']]
df8g.columns = ['Winter wheat', 'Oats']

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=False)
axes[0] = plot_clustered_stacked([df1n, df1g], axes[0], ['Without grazing', 'With grazing'])
axes[0].set_title('WRE1-Grassland', fontsize=18)
axes[0].set_xlabel('Year', fontsize=16)
axes[0].set_ylabel('Forage yield, t/ha', fontsize=16)
axes[0].grid(True)
axes[0].tick_params(axis='x', labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)

axes[1] = plot_clustered_stacked([df8n, df8g], axes[1], ['Without grazing', 'With grazing'])
axes[1].set_title('WRE8-Cropland', fontsize=18)
axes[1].set_xlabel('Year', fontsize=16)
axes[1].set_ylabel('Crop yield, t/ha', fontsize=16)
axes[1].grid(True)
axes[1].tick_params(axis='x', labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)
plt.tight_layout()

plt.savefig('../post_analysis/Figures/Calibration_Farm_1&8annual_Crop_daily_OF.png', dpi=600, bbox_inches="tight")
