# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 11:20:26 2023

@author: Mahesh.Maskey
"""

# -*- coding: utf-8 -*-
import pandas as pd

"""
Created on Mon Dec 12 13:11:54 2022

@author: Mahesh.Maskey
"""
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utility import summarize_multi_output4sites

warnings.filterwarnings('ignore')
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)

print('\014')


def compile_summary(metric, isAppend=True):
    tuple_WYLD_1, tuple_WYLD_8 = summarize_multi_output4sites(site1='Farm_1',
                                                              site2='Farm_8',
                                                              location='daily_basin',
                                                              scale='daily',
                                                              match_scale='daily',
                                                              var='WYLD',
                                                              metric=metric,
                                                              isAppend=isAppend)
    df_1 = tuple_WYLD_1[0][3]
    df_1['Response variable'] = 'Water yield'
    df_8 = tuple_WYLD_8[0][3]
    df_8['Response variable'] = 'Water yield'

    tuple_Q_1, tuple_Q_8 = summarize_multi_output4sites(site1='Farm_1',
                                                        site2='Farm_8',
                                                        location='daily_basin',
                                                        scale='daily',
                                                        match_scale='daily',
                                                        var='Q',
                                                        metric=metric,
                                                        isAppend=True)
    df_1_ = tuple_Q_1[0][3]
    df_1_['Response variable'] = 'Surface runoff'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_Q_8[0][3]
    df_8_['Response variable'] = 'Surface runoff'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_DPRK_1, tuple_DPRK_8 = summarize_multi_output4sites(site1='Farm_1',
                                                              site2='Farm_8',
                                                              location='daily_basin',
                                                              scale='daily',
                                                              match_scale='daily',
                                                              var='DPRK',
                                                              metric=metric,
                                                              isAppend=True)
    df_1_ = tuple_DPRK_1[0][3]
    df_1_['Response variable'] = 'Deep percolation'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_DPRK_8[0][3]
    df_8_['Response variable'] = 'Deep percolation'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_YSD_1, tuple_YSD_8 = summarize_multi_output4sites(site1='Farm_1',
                                                            site2='Farm_8',
                                                            location='daily_outlet',
                                                            scale='daily',
                                                            match_scale='daily',
                                                            var='YSD',
                                                            metric=metric,
                                                            isAppend=isAppend)
    df_1_ = tuple_YSD_1[0][3]
    df_1_['Response variable'] = 'Sediment yield'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_YSD_8[0][3]
    df_8_['Response variable'] = 'Sediment yield'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_RUS2_1, tuple_RUS2_8 = summarize_multi_output4sites(site1='Farm_1',
                                                              site2='Farm_8',
                                                              location='daily_outlet',
                                                              scale='daily',
                                                              match_scale='daily',
                                                              var='RUS2',
                                                              metric=metric,
                                                              isAppend=True)
    df_1_ = tuple_RUS2_1[0][3]
    df_1_['Response variable'] = 'Soil erosion'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_RUS2_8[0][3]
    df_8_['Response variable'] = 'Soil erosion'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_ET_1, tuple_ET_8 = summarize_multi_output4sites(site1='Farm_1',
                                                          site2='Farm_8',
                                                          location='daily_basin',
                                                          scale='daily',
                                                          match_scale='daily',
                                                          var='ET',
                                                          metric=metric,
                                                          isAppend=True)
    df_1_ = tuple_ET_1[0][3]
    df_1_['Response variable'] = 'Evapotranspiration'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_ET_8[0][3]
    df_8_['Response variable'] = 'Evapotranspiration'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_PET_1, tuple_PET_8 = summarize_multi_output4sites(site1='Farm_1',
                                                            site2='Farm_8',
                                                            location='daily_basin',
                                                            scale='daily',
                                                            match_scale='daily',
                                                            var='PET',
                                                            metric=metric,
                                                            isAppend=True)
    df_1_ = tuple_PET_1[0][3]
    df_1_['Response variable'] = 'Potential evapotranspiration'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_PET_8[0][3]
    df_8_['Response variable'] = 'Potential evapotranspiration'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_LAI_1, tuple_LAI_8 = summarize_multi_output4sites(site1='Farm_1',
                                                            site2='Farm_8',
                                                            location='daily_basin',
                                                            scale='daily',
                                                            match_scale='daily',
                                                            var='LAI',
                                                            metric=metric,
                                                            isAppend=True)
    df_1_ = tuple_LAI_1[0][3]
    df_1_['Response variable'] = 'Leaf area index'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_LAI_8[0][3]
    df_8_['Response variable'] = 'Leaf area index'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_PRCP_1, tuple_PRCP_8 = summarize_multi_output4sites(site1='Farm_1',
                                                             site2='Farm_8',
                                                             location='daily_basin',
                                                             scale='daily',
                                                             match_scale='daily',
                                                             var='PRCP',
                                                             metric=metric,
                                                             isAppend=True)
    df_1_ = tuple_PRCP_1[0][3]
    df_1_['Response variable'] = 'PRCP'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_PRCP_8[0][3]
    df_8_['Response variable'] = 'PRCP'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_STD_1, tuple_STD_8 = summarize_multi_output4sites(site1='Farm_1',
                                                            site2='Farm_8',
                                                            location='daily_basin',
                                                            scale='daily',
                                                            match_scale='daily',
                                                            var='STD',
                                                            metric=metric,
                                                            isAppend=True)
    df_1_ = tuple_STD_1[0][3]
    df_1_['Response variable'] = 'Standing dead crop residue'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_STD_8[0][3]
    df_8_['Response variable'] = 'Standing dead crop residue'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_STL_1, tuple_STL_8 = summarize_multi_output4sites(site1='Farm_1',
                                                            site2='Farm_8',
                                                            location='daily_basin',
                                                            scale='daily',
                                                            match_scale='daily',
                                                            var='STL',
                                                            metric=metric,
                                                            isAppend=True)
    df_1_ = tuple_STL_1[0][3]
    df_1_['Response variable'] = 'Standing live plant biomass'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_STL_8[0][3]
    df_8_['Response variable'] = 'Standing live plant biomass'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_BIOM_1, tuple_BIOM_8 = summarize_multi_output4sites(site1='Farm_1',
                                                              site2='Farm_8',
                                                              location='daily_basin',
                                                              scale='daily',
                                                              match_scale='daily',
                                                              var='BIOM',
                                                              metric=metric,
                                                              isAppend=True)
    df_1_ = tuple_BIOM_1[0][3]
    df_1_['Response variable'] = 'Crop biomass'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_BIOM_8[0][3]
    df_8_['Response variable'] = 'Crop biomass'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_YLDF_1, tuple_YLDF_8 = summarize_multi_output4sites(site1='Farm_1',
                                                              site2='Farm_8',
                                                              location='annual',
                                                              scale='daily',
                                                              match_scale='annual',
                                                              var='YLDF',
                                                              metric=metric,
                                                              isAppend=True)

    df_1_ = tuple_YLDF_1[3]
    df_1_['Response variable'] = 'Forage yield'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_YLDF_8[3]
    df_8_['Response variable'] = 'Forage yield'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_YLDG_1, tuple_YLDG_8 = summarize_multi_output4sites(site1='Farm_1',
                                                              site2='Farm_8',
                                                              location='annual',
                                                              scale='daily',
                                                              match_scale='annual',
                                                              var='YLDG',
                                                              metric=metric,
                                                              isAppend=True)
    df_1_ = tuple_YLDG_1[3]
    df_1_['Response variable'] = 'Crop yield'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_YLDG_8[3]
    df_8_['Response variable'] = 'Crop yield'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_TN_1, tuple_TN_8 = summarize_multi_output4sites(site1='Farm_1',
                                                          site2='Farm_8',
                                                          location='daily_basin',
                                                          scale='daily',
                                                          match_scale='daily',
                                                          var='TN',
                                                          metric=metric,
                                                          isAppend=True)
    df_1_ = tuple_TN_1[0][3]
    df_1_['Response variable'] = 'Total nitrogen'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_TN_8[0][3]
    df_8_['Response variable'] = 'Total nitrogen'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_TP_1, tuple_TP_8 = summarize_multi_output4sites(site1='Farm_1',
                                                          site2='Farm_8',
                                                          location='daily_basin',
                                                          scale='daily',
                                                          match_scale='daily',
                                                          var='TP',
                                                          metric=metric,
                                                          isAppend=True)
    df_1_ = tuple_TP_1[0][3]
    df_1_['Response variable'] = 'Total phosphorus'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_TP_8[0][3]
    df_8_['Response variable'] = 'Total phosphorus'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_WS_1, tuple_WS_8 = summarize_multi_output4sites(site1='Farm_1',
                                                          site2='Farm_8',
                                                          location='annual',
                                                          scale='daily',
                                                          match_scale='annual',
                                                          var='WS',
                                                          metric=metric,
                                                          isAppend=True)
    df_1_ = tuple_WS_1[3]
    df_1_['Response variable'] = 'Drought stress'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_WS_8[3]
    df_8_['Response variable'] = 'Drought stress'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_TS_1, tuple_TS_8 = summarize_multi_output4sites(site1='Farm_1',
                                                          site2='Farm_8',
                                                          location='annual',
                                                          scale='daily',
                                                          match_scale='annual',
                                                          var='TS',
                                                          metric=metric,
                                                          isAppend=True)
    df_1_ = tuple_TS_1[3]
    df_1_['Response variable'] = 'Temperature stress'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_TS_8[3]
    df_8_['Response variable'] = 'Temperature stress'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_NS_1, tuple_NS_8 = summarize_multi_output4sites(site1='Farm_1',
                                                          site2='Farm_8',
                                                          location='annual',
                                                          scale='daily',
                                                          match_scale='annual',
                                                          var='NS',
                                                          metric=metric,
                                                          isAppend=True)
    df_1_ = tuple_NS_1[3]
    df_1_['Response variable'] = 'Nitrogen stress,'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_NS_8[3]
    df_8_['Response variable'] = 'Nitrogen stress,'
    df_8 = pd.concat([df_8, df_8_], axis=0)

    tuple_PS_1, tuple_PS_8 = summarize_multi_output4sites(site1='Farm_1',
                                                          site2='Farm_8',
                                                          location='annual',
                                                          scale='daily',
                                                          match_scale='annual',
                                                          var='PS',
                                                          metric=metric,
                                                          isAppend=True)
    df_1_ = tuple_PS_1[3]
    df_1_['Response variable'] = 'Phosphorus  stress'
    df_1 = pd.concat([df_1, df_1_], axis=0)
    df_8_ = tuple_PS_8[3]
    df_8_['Response variable'] = 'Phosphorus  stress'
    df_8 = pd.concat([df_8, df_8_], axis=0)
    df_1['metric'] = metric
    df_8['metric'] = metric
    df_1['area'] = 'WRE1'
    df_8['area'] = 'WRE8'
    return df_1, df_8


df_of_1, df_of_8 = compile_summary(metric='OF', isAppend=False)
df_nse_1, df_nse_8 = compile_summary(metric='NSE')
df_pbias_1, df_pbias_8 = compile_summary(metric='PBIAS')
df_cod_1, df_cod_8 = compile_summary(metric='COD')
df1 = pd.concat([df_of_1, df_nse_1, df_cod_1, df_pbias_1], axis=0)
df8 = pd.concat([df_of_8, df_nse_8, df_cod_8, df_pbias_8], axis=0)
df = pd.concat([df1, df8], axis=0)
df.to_csv('../post_analysis/Results/Calibration_multi_metric_summary.csv')

df_n = df[df.index == 'Without grazing']
df_g = df[df.index == 'With grazing']
df_n_cal = df_n[['Calibration', 'Response variable', 'metric', 'area']]
df_n_val = df_n[['Validation', 'Response variable', 'metric', 'area']]
df_g_cal = df_g[['Calibration', 'Response variable', 'metric', 'area']]
df_g_val = df_g[['Validation', 'Response variable', 'metric', 'area']]
df_cal = df_n_cal.copy()
df_cal.columns = ['Without grazing', 'Response variable', 'metric', 'area']
df_cal.insert(1, 'With grazing', df_g_cal.Calibration.values)
df_cal.insert(2, 'Change percent',
              (df_cal['Without grazing'].values - df_cal['With grazing'].values) / df_cal['Without '
                                                                                          'grazing'].values * 100)
df_cal = df_cal.reset_index(drop=True)

df_val = df_n_val.copy()
df_val.columns = ['Without grazing', 'Response variable', 'metric', 'area']
df_val.insert(1, 'With grazing', df_g_val.Validation.values)
df_val.insert(2, 'Change percent',
              (df_val['Without grazing'].values - df_val['With grazing'].values) / df_val['Without '
                                                                                          'grazing'].values * 100)
df_val = df_val.reset_index(drop=True)

df_cal_1 = df_cal[df_cal.area == 'WRE1']
df_cal_8 = df_cal[df_cal.area == 'WRE8']
df_val_1 = df_val[df_val.area == 'WRE1']
df_val_8 = df_val[df_val.area == 'WRE8']

fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=False, sharey=True)
axes[0, 0] = sns.barplot(data=df_cal_1, y='Response variable', x='Change percent', hue='metric', ci=None, ax=axes[0, 0])
# axes[0, 0].set(xscale="log")
axes[0, 0].set_xlabel("")
axes[0, 0].set_ylabel("WRE1", fontsize=16)
axes[0, 0].set_title("Calibration", fontsize=16)
axes[0, 0].grid(True)
axes[0, 0].legend_.remove()
axes[0, 1] = sns.barplot(data=df_val_1, y='Response variable', x='Change percent', hue='metric', ci=None, ax=axes[0, 1])
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("")
axes[0, 1].set_title("Validation", fontsize=16)
axes[0, 1].legend_.remove()
axes[0, 1].grid(True)
# axes[0, 1].set(xscale="log")
axes[1, 0] = sns.barplot(data=df_cal_8, y='Response variable', x='Change percent', hue='metric', ci=None, ax=axes[1, 0])
axes[1, 0].set_xlabel("(Without grazing - grazing), %", fontsize=16)
axes[1, 0].set_ylabel("WRE8", fontsize=16)
axes[1, 0].legend_.remove()
axes[1, 0].grid(True)
# axes[1, 0].set(xscale="log")
axes[1, 1] = sns.barplot(data=df_val_8, y='Response variable', x='Change percent', hue='metric', ci=None, ax=axes[1, 1])
axes[1, 1].set_xlabel("(Without grazing - grazing), %", fontsize=16)
axes[1, 1].set_ylabel("")
axes[1, 1].legend_.remove()
axes[1, 1].grid(True)
# axes[1, 1].set(xscale="log")
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Calibration_multi_metric_Bar_plot.png'),
            dpi=600, bbox_inches="tight")
plt.show()

## heat maps

df_data_heat_cal_1 = pd.DataFrame({'OF': df_cal_1[df_cal_1.metric == 'OF']['Change percent'].values,
                                   'NSE': df_cal_1[df_cal_1.metric == 'NSE']['Change percent'].values,
                                   'COD': df_cal_1[df_cal_1.metric == 'COD']['Change percent'].values,
                                   'PBIAS': df_cal_1[df_cal_1.metric == 'PBIAS']['Change percent'].values})
rv = df_cal_1['Response variable'].unique()
df_data_heat_cal_1.index = rv
df_data_heat_cal_1 = df_data_heat_cal_1.dropna()

df_data_heat_val_1 = pd.DataFrame({'OF': df_val_1[df_val_1.metric == 'OF']['Change percent'].values,
                                   'NSE': df_val_1[df_val_1.metric == 'NSE']['Change percent'].values,
                                   'COD': df_val_1[df_val_1.metric == 'COD']['Change percent'].values,
                                   'PBIAS': df_val_1[df_val_1.metric == 'PBIAS']['Change percent'].values})
rv = df_val_1['Response variable'].unique()
df_data_heat_val_1.index = rv
df_data_heat_val_1 = df_data_heat_val_1.dropna()

df_data_heat_cal_8 = pd.DataFrame({'OF': df_cal_8[df_cal_8.metric == 'OF']['Change percent'].values,
                                   'NSE': df_cal_8[df_cal_8.metric == 'NSE']['Change percent'].values,
                                   'COD': df_cal_8[df_cal_8.metric == 'COD']['Change percent'].values,
                                   'PBIAS': df_cal_8[df_cal_8.metric == 'PBIAS']['Change percent'].values})
rv = df_cal_8['Response variable'].unique()
df_data_heat_cal_8.index = rv
df_data_heat_cal_8 = df_data_heat_cal_8.dropna()

df_data_heat_val_8 = pd.DataFrame({'OF': df_val_8[df_val_8.metric == 'OF']['Change percent'].values,
                                   'NSE': df_val_8[df_val_8.metric == 'NSE']['Change percent'].values,
                                   'COD': df_val_8[df_val_8.metric == 'COD']['Change percent'].values,
                                   'PBIAS': df_val_8[df_val_8.metric == 'PBIAS']['Change percent'].values})
rv = df_val_8['Response variable'].unique()
df_data_heat_val_8.index = rv
df_data_heat_val_8 = df_data_heat_val_8.dropna()

minVal = np.min([df_data_heat_cal_8.min().min(), df_data_heat_val_8.min().min()])

maxVal = np.max([df_data_heat_cal_8.max().max(), df_data_heat_val_8.max().max()])

fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=False, sharey=True)
sns.heatmap(data=df_data_heat_cal_1, ax=axes[0, 0], cmap=sns.color_palette("coolwarm"), cbar_kws={'label': 'Change %'},
            linewidth=.5)
axes[0, 0].set_ylabel('Response variable')
axes[0, 0].set_ylabel("WRE1", fontsize=16)
axes[0, 0].set_title("Calibration", fontsize=16)

sns.heatmap(data=df_data_heat_val_1, ax=axes[0, 1], cmap=sns.color_palette("coolwarm"), cbar_kws={'label': 'Change %'},
            linewidth=.5)
axes[0, 1].set_title("Validation", fontsize=16)

sns.heatmap(data=df_data_heat_cal_8, vmin=minVal, vmax=maxVal, ax=axes[1, 0], cmap=sns.color_palette("coolwarm"),
            cbar_kws={'label': 'Change %'}, linewidth=.5)
axes[1, 0].set_xlabel('Performance Metrics')
axes[1, 0].set_ylabel("WRE8", fontsize=16)

sns.heatmap(data=df_data_heat_val_8, vmin=minVal, vmax=maxVal, ax=axes[1, 1], cmap=sns.color_palette("coolwarm"),
            cbar_kws={'label': 'Change %'}, linewidth=.5)
axes[1, 1].set_xlabel('Performance Metrics')
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Calibration_multi_metric_summary_Heatmap.png'),
            dpi=600, bbox_inches="tight")
plt.show()
