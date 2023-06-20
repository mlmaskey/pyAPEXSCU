# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 15:59:11 2023

@author: Mahesh.Maskey
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utility import print_progress_bar
from utility import nash, nancorr, pbias
from configobj import ConfigObj

warnings.filterwarnings('ignore')
print('/014')

graph_dir = '../post_analysis/Figures'


def pair_matrix(nrow, ncol):
    x = np.arange(0, nrow)
    y = np.arange(0, ncol)
    pair = []
    for i in x:
        for j in y:
            pair.append((i, j))
    return pair


def get_uncertainty_param(field, is_grazing):
    if is_grazing:
        scn = 'pyAPEX_g'
    else:
        scn = 'pyAPEX_n'
    file_un = f'../{field}/{scn}/pyAPEX/OutputUncertainty/APEXPARM.csv'
    file_cal = f'../{field}/{scn}/pyAPEX/Output/APEXPARM.csv'
    df_params_cal = pd.read_csv(file_cal, index_col=0)
    df_params_un = pd.read_csv(file_un, index_col=0)
    id_params = [2, 4, 7, 8, 14, 15, 17, 18, 19, 20, 23, 34, 42, 45, 50, 65, 66, 69, 70, 72]
    for i in range(len(id_params)):
        id_params[i] = id_params[i] + 69
    df_params_cal = df_params_cal.iloc[:, id_params]
    df_params_un = df_params_un.iloc[:, id_params]
    param_best = df_params_un.iloc[0, :]
    df_params_un = df_params_un.iloc[1:, :]
    return df_params_cal, df_params_un, param_best


params_name = ['Root growth-soil strength',
               'Water storage N leaching',
               'N fixation',
               'Soluble phosphorus runoff coefficient',
               'Nitrate leaching ratio',
               'Runoff CN Residue Adjustment Parameter',
               'Soil evaporation – plant cover factor',
               'Sediment routing exponent',
               'Sediment routing coefficient',
               'Runoff curve number initial abstraction',
               'Hargreaves PET equation coefficient',
               'Hargreaves PET equation exponent',
               'SCS curve number index coefficient',
               'Sediment routing travel time coefficient',
               'Rainfall interception coefficient',
               'RUSLE2 transport capacity parameter',
               'RUSLE2 threshold transport capacity coefficient',
               'Coefficient adjusts microbial activity function in the topsoil layer',
               'Microbial decay rate coefficient',
               'Volatilization/nitrification partitioning coefficient']

df_params_cal_1_n, df_params_un_1_n, param_best = get_uncertainty_param(field='Farm_1', is_grazing=False)
pairs = pair_matrix(4, 5)
fig, axes = plt.subplots(4, 5, figsize=(25, 25), sharex=False, sharey=True)
for i in range(df_params_cal_1_n.shape[1]):
    sns.histplot(df_params_cal_1_n.iloc[:, i], ax=axes[pairs[i]], stat="probability", kde=True)
    axes[pairs[i]].axvline(param_best[i], color='red')
    axes[pairs[i]].set_xlabel('')
    axes[pairs[i]].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Farm_1_non_grazing_Uncertainty_Calibrated_param_distribution.png'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(4, 5, figsize=(25, 25), sharex=False, sharey=True)
for i in range(df_params_un_1_n.shape[1]):
    sns.histplot(df_params_un_1_n.iloc[:, i], ax=axes[pairs[i]], stat="probability", kde=True)
    axes[pairs[i]].axvline(param_best[i], color='red')
    axes[pairs[i]].set_xlabel('')
    axes[pairs[i]].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Farm_1_non_grazing_Uncertainty_param_distribution.png'),
            dpi=600, bbox_inches="tight")
plt.show()

df_params_cal_1_g, df_params_un_1_g, param_best = get_uncertainty_param(field='Farm_1', is_grazing=True)
pairs = pair_matrix(4, 5)
fig, axes = plt.subplots(4, 5, figsize=(25, 25), sharex=False, sharey=True)
for i in range(df_params_cal_1_g.shape[1]):
    sns.histplot(df_params_cal_1_g.iloc[:, i], ax=axes[pairs[i]], stat="probability", kde=True)
    axes[pairs[i]].axvline(param_best[i], color='red')
    axes[pairs[i]].set_xlabel('')
    axes[pairs[i]].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Farm_1_grazing_Uncertainty_Calibrated_param_distribution.png'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(4, 5, figsize=(25, 25), sharex=False, sharey=True)
for i in range(df_params_un_1_g.shape[1]):
    sns.histplot(df_params_un_1_g.iloc[:, i], ax=axes[pairs[i]], stat="probability", kde=True)
    axes[pairs[i]].axvline(param_best[i], color='red')
    axes[pairs[i]].set_xlabel('')
    axes[pairs[i]].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Farm_1_grazing_Uncertainty_param_distribution.png'),
            dpi=600, bbox_inches="tight")
plt.show()

df_params_cal_8_n, df_params_un_8_n, param_best = get_uncertainty_param(field='Farm_1', is_grazing=False)
pairs = pair_matrix(4, 5)
fig, axes = plt.subplots(4, 5, figsize=(25, 25), sharex=False, sharey=True)
for i in range(df_params_cal_8_n.shape[1]):
    sns.histplot(df_params_cal_8_n.iloc[:, i], ax=axes[pairs[i]], stat="probability", kde=True)
    axes[pairs[i]].axvline(param_best[i], color='red')
    axes[pairs[i]].set_xlabel('')
    axes[pairs[i]].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Farm_8_non_grazing_Uncertainty_Calibrated_param_distribution.png'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(4, 5, figsize=(25, 25), sharex=False, sharey=True)
for i in range(df_params_un_8_n.shape[1]):
    sns.histplot(df_params_un_8_n.iloc[:, i], ax=axes[pairs[i]], stat="probability", kde=True)
    axes[pairs[i]].axvline(param_best[i], color='red')
    axes[pairs[i]].set_xlabel('')
    axes[pairs[i]].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Farm_8_non_grazing_Uncertainty_param_distribution.png'),
            dpi=600, bbox_inches="tight")
plt.show()

df_params_cal_8_g, df_params_un_8_g, param_best = get_uncertainty_param(field='Farm_1', is_grazing=True)
pairs = pair_matrix(4, 5)
fig, axes = plt.subplots(4, 5, figsize=(25, 25), sharex=False, sharey=True)
for i in range(df_params_cal_8_g.shape[1]):
    sns.histplot(df_params_cal_8_g.iloc[:, i], ax=axes[pairs[i]], stat="probability", kde=True)
    axes[pairs[i]].axvline(param_best[i], color='red')
    axes[pairs[i]].set_xlabel('')
    axes[pairs[i]].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Farm_8_grazing_Uncertainty_Calibrated_param_distribution.png'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(4, 5, figsize=(25, 25), sharex=False, sharey=True)
for i in range(df_params_un_8_g.shape[1]):
    sns.histplot(df_params_un_8_g.iloc[:, i], ax=axes[pairs[i]], stat="probability", kde=True)
    axes[pairs[i]].axvline(param_best[i], color='red')
    axes[pairs[i]].set_xlabel('')
    axes[pairs[i]].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Farm_8_grazing_Uncertainty_param_distribution.png'),
            dpi=600, bbox_inches="tight")
plt.show()
'Annual_'