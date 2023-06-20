# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:15:08 2023

@author: Mahesh.Maskey
"""

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')


def plot_water_balance(datadir, site, scenario, attribute, metric, scale, ax, is_x_label, is_y_label, is_y2_label):
    if scenario == 'non_grazing':
        program = 'pyAPEX_n'
    else:
        program = 'pyAPEX_g'
    datadir = f'{datadir}/{site}/{program}/pyAPEX/Output/{attribute}/'

    file_stat = 'summary_stats.csv'
    file_stat_path = os.path.join(datadir, file_stat)
    data_stats = pd.read_csv(file_stat_path, index_col=0)
    if metric == 'OF':
        stats_con = data_stats[(data_stats.index == 'Objective Function') & (data_stats.SCALE == scale)]
    elif metric == 'NSE':
        stats_con = data_stats[(data_stats.index == 'NSE') & (data_stats.SCALE == scale)]
    elif metric == 'COD':
        stats_con = data_stats[(data_stats.index == 'COD') & (data_stats.SCALE == scale)]
    else:
        stats_con = data_stats[(data_stats.index == 'PBIAS') & (data_stats.SCALE == scale)]

    id_run = stats_con.RunId.values[0]
    file_basin = f'daily_basin_{id_run:07}.csv'
    file_outlet = f'daily_outlet_{id_run:07}.csv'
    file_calibration = f'model_calibration_{metric}.csv'
    file_basin_path = os.path.join(datadir, file_basin)
    file_outlet_path = os.path.join(datadir, file_outlet)
    file_calibration_path = os.path.join(datadir, file_calibration)
    df_outlet = pd.read_csv(file_outlet_path, index_col=0)
    df_outlet.index = pd.to_datetime(df_outlet.index)
    df_basin = pd.read_csv(file_basin_path, index_col=0)
    crops = df_basin.CPNM.unique()
    if len(crops) > 1:
        col_names = df_basin.columns[4:]
        new_cols = list(df_outlet.columns[:3]) + list(col_names)
        df_basin_new = pd.DataFrame(columns=new_cols, index=df_outlet.index)
        df_basin_new.iloc[:, :3] = df_outlet.iloc[:, :3]
        for col in col_names:
            df_basin_crops = pd.DataFrame()
            for crop in crops:
                df_basin_CP = df_basin[df_basin.CPNM == crop]
                df_col = pd.DataFrame(df_basin_CP[col])
                df_basin_crops = pd.concat([df_basin_crops, df_col], axis=1)
            df_basin_crops.columns = crops
            df_basin_new[col] = df_basin_crops.sum(axis=1).values
        df_basin = df_basin_new.copy()
    else:
        df_basin = df_basin
    df_calibration = pd.read_csv(file_calibration_path, index_col=0)
    df_calibration.index = pd.to_datetime(df_calibration.index)
    df_calibration = df_calibration[df_calibration.SCALE == scale]
    df_water_balance_daily = pd.DataFrame({'Runoff': df_outlet.WYLD.values, 'Precipitation': df_basin.PRCP.values,
                                           'Percolation': df_basin.DPRK.values, 'ET': df_basin.ET.values,
                                           'PET': df_basin.PET.values}, index=df_outlet.index)
    df_water_balance_daily_model = df_water_balance_daily[df_water_balance_daily.index <= df_calibration.index[-1]]
    df_water_balance_daily_model = df_water_balance_daily_model[
        df_water_balance_daily_model.index >= df_calibration.index[0]]
    df_water_balance_daily_model.insert(5, 'ObservedRunoff', df_calibration.Observed.values)
    df_water_balance_monthly_model = df_water_balance_daily_model.resample('M').sum()
    df_water_balance_yearly_model = df_water_balance_daily_model.resample('Y').sum()
    df_water_balance_yearly_model.index = df_water_balance_yearly_model.index.year.astype(int)
    y_peak = int(df_water_balance_monthly_model.max().max() * 1.5)
    ax = sns.lineplot(data=df_water_balance_monthly_model.Runoff, color="g", label='Runoff (Q)', ax=ax)
    ax = sns.lineplot(data=df_water_balance_monthly_model.Percolation, color="black", label='Deep Percolation (I)',
                      ax=ax)
    ax = sns.lineplot(data=df_water_balance_monthly_model.ET, color="r", label='Evapotranspiration (E)', ax=ax)
    ax.set(ylim=(0, y_peak))
    if is_y_label:
        ax.set_ylabel('Output: Q, I, E, mm', fontsize=16)
    if is_x_label:
        ax.set_xlabel('Month', fontsize=16)
    ax.grid(True)
    ax.set(xlim=(df_water_balance_monthly_model.index[0], df_water_balance_monthly_model.index[-1]))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.get_legend().remove()
    ax2 = ax.twinx()
    sns.lineplot(data=df_water_balance_monthly_model, x=df_water_balance_monthly_model.index, y='Precipitation',
                 color="b", ax=ax2)
    ax2.fill_between(df_water_balance_monthly_model.index, 0, df_water_balance_monthly_model["Precipitation"],
                     alpha=0.5)
    ax2.set(ylim=(0, y_peak))
    ax2.set_ylabel(' ')
    if is_y2_label:
        ax2.set_ylabel('Precipitation (P), mm', fontsize=16)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.invert_yaxis()
    return ax


data_dir = 'C:/Users/Mahesh.Maskey/Documents/Project/OklahomaWRE'
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)
fig, axes = plt.subplots(2, 2, figsize=(20, 8), tight_layout=True)
axes[0, 0] = plot_water_balance(datadir=data_dir, site='Farm_1', scenario='non_grazing', attribute='runoff',
                                metric='OF', scale='daily', ax=axes[0, 0],
                                is_x_label=False, is_y_label=True, is_y2_label=False)
axes[0, 0].set_title('WRE1: Grassland', fontsize=20)
axes[1, 0] = plot_water_balance(datadir=data_dir, site='Farm_1', scenario='grazing', attribute='runoff',
                                metric='OF', scale='daily', ax=axes[1, 0],
                                is_x_label=True, is_y_label=True, is_y2_label=False)
axes[0, 1] = plot_water_balance(datadir=data_dir, site='Farm_8', scenario='non_grazing', attribute='runoff',
                                metric='OF', scale='daily', ax=axes[0, 1],
                                is_x_label=False, is_y_label=False, is_y2_label=True)
axes[0, 1].set_title('WRE8: Cropland', fontsize=20)
axes[1, 1] = plot_water_balance(datadir=data_dir, site='Farm_8', scenario='grazing', attribute='runoff',
                                metric='OF', scale='daily', ax=axes[1, 1],
                                is_x_label=True, is_y_label=False, is_y2_label=True)
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, f'Calibration_Monthly_Water_balance_OF.png'), dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 8), tight_layout=True)
axes[0, 0] = plot_water_balance(datadir=data_dir, site='Farm_1', scenario='non_grazing', attribute='runoff',
                                metric='NSE', scale='daily', ax=axes[0, 0],
                                is_x_label=False, is_y_label=True, is_y2_label=False)
axes[0, 0].set_title('WRE1: Grassland', fontsize=20)
axes[1, 0] = plot_water_balance(datadir=data_dir, site='Farm_1', scenario='grazing', attribute='runoff',
                                metric='NSE', scale='daily', ax=axes[1, 0],
                                is_x_label=True, is_y_label=True, is_y2_label=False)
axes[0, 1] = plot_water_balance(datadir=data_dir, site='Farm_8', scenario='non_grazing', attribute='runoff',
                                metric='NSE', scale='daily', ax=axes[0, 1],
                                is_x_label=False, is_y_label=False, is_y2_label=True)
axes[0, 1].set_title('WRE8: Cropland', fontsize=20)
axes[1, 1] = plot_water_balance(datadir=data_dir, site='Farm_8', scenario='grazing', attribute='runoff',
                                metric='NSE', scale='daily', ax=axes[1, 1],
                                is_x_label=True, is_y_label=False, is_y2_label=True)
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, f'Calibration_Monthly_Water_balance_NSE.png'), dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 8), tight_layout=True)
axes[0, 0] = plot_water_balance(datadir=data_dir, site='Farm_1', scenario='non_grazing', attribute='runoff',
                                metric='COD', scale='daily', ax=axes[0, 0],
                                is_x_label=False, is_y_label=True, is_y2_label=False)
axes[0, 0].set_title('WRE1: Grassland', fontsize=20)
axes[1, 0] = plot_water_balance(datadir=data_dir, site='Farm_1', scenario='grazing', attribute='runoff',
                                metric='COD', scale='daily', ax=axes[1, 0],
                                is_x_label=True, is_y_label=True, is_y2_label=False)
axes[0, 1] = plot_water_balance(datadir=data_dir, site='Farm_8', scenario='non_grazing', attribute='runoff',
                                metric='COD', scale='daily', ax=axes[0, 1],
                                is_x_label=False, is_y_label=False, is_y2_label=True)
axes[0, 1].set_title('WRE8: Cropland', fontsize=20)
axes[1, 1] = plot_water_balance(datadir=data_dir, site='Farm_8', scenario='grazing', attribute='runoff',
                                metric='COD', scale='daily', ax=axes[1, 1],
                                is_x_label=True, is_y_label=False, is_y2_label=True)
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, f'Calibration_Monthly_Water_balance_COD.png'), dpi=600, bbox_inches="tight")
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(20, 8), tight_layout=True)
axes[0, 0] = plot_water_balance(datadir=data_dir, site='Farm_1', scenario='non_grazing', attribute='runoff',
                                metric='PBIAS', scale='daily', ax=axes[0, 0],
                                is_x_label=False, is_y_label=True, is_y2_label=False)
axes[0, 0].set_title('WRE1: Grassland', fontsize=20)
axes[1, 0] = plot_water_balance(datadir=data_dir, site='Farm_1', scenario='grazing', attribute='runoff',
                                metric='PBIAS', scale='daily', ax=axes[1, 0],
                                is_x_label=True, is_y_label=True, is_y2_label=False)
axes[0, 1] = plot_water_balance(datadir=data_dir, site='Farm_8', scenario='non_grazing', attribute='runoff',
                                metric='PBIAS', scale='daily', ax=axes[0, 1],
                                is_x_label=False, is_y_label=False, is_y2_label=True)
axes[0, 1].set_title('WRE8: Cropland', fontsize=20)
axes[1, 1] = plot_water_balance(datadir=data_dir, site='Farm_8', scenario='grazing', attribute='runoff',
                                metric='PBIAS', scale='daily', ax=axes[1, 1],
                                is_x_label=True, is_y_label=False, is_y2_label=True)
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=3, fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, f'Calibration_Monthly_Water_balance_PBIAS.png'), dpi=600, bbox_inches="tight")
plt.show()