# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:47:41 2023

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
from easypy import easypy as ep

warnings.filterwarnings('ignore')
print('\014')

graph_dir = '../post_analysis/Figures'


def evaluate_best_glue(field='Farm_1', is_grazing=False, df_glue=None, attribute='runoff', file='Calibration_data.csv',
                       parameter='WYLD'):
    df_obs = read_measured(field, is_grazing, attribute, file)
    df_model = get_best_data(field, is_grazing, attribute, parameter)
    df_data = pd.concat([df_obs, df_model], axis=1)
    df_data = df_data.dropna()
    df_data.columns = ['Observed', 'Modeled']
    NSE = nash(df_data.Observed.values, df_data.Modeled.values)
    X, Y = df_data.Observed.values, df_data.Modeled.values
    MSE = np.sum((X - Y) ** 2) / len(X)
    max_nash = df_glue.NSE.max()
    glue_NSE = np.exp(-NSE / max_nash)
    min_mse = df_glue.MSE.min()
    glue_MSE = np.exp(-MSE / min_mse)
    return glue_NSE, glue_MSE


if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)


def get_glue_data(field, is_grazing):
    if is_grazing:
        scn = 'g'
    else:
        scn = 'n'
    read_dir = f'../post_analysis/Results/{field}_{scn}_Uncertainty_range.csv'
    df_gl = pd.read_csv(read_dir, index_col=0)

    return df_gl


def get_measure(data_dir, file_name):
    file_name = os.path.join(data_dir, file_name)
    df_data = pd.read_csv(file_name)
    date_vec = []
    ndata = df_data.shape[0]
    for i in range(ndata):
        date_vec.append(pd.to_datetime(f'{df_data.Year[i]}-{df_data.Month[i]}-{df_data.Day[i]}'))
    df_data.Date = date_vec
    df_data.index = df_data.Date
    df_data = df_data[['Date', 'Year', 'Month', 'Day', 'runoff (mm)', 'sediment (kg)']]
    df_data.columns = ['Date', 'Year', 'Month', 'Day', 'runoff', 'sediment']
    return df_data


def read_measured(field, is_grazing, attribute, file):
    if is_grazing:
        scenario = 'pyAPEX_g'
    else:
        scenario = 'pyAPEX_n'
    obs_path = f'../{field}/{scenario}/pyAPEX/Program/'
    df = get_measure(obs_path, file)
    df = df.drop('Date', axis=1)
    df.index = pd.to_datetime(df.index)
    df = pd.DataFrame(df[attribute])
    return df


def get_best_data(field, is_grazing, attribute, parameter):
    if is_grazing:
        scn = 'grazing'
    else:
        scn = 'non_grazing'
    file = f'../post_analysis/Uncertainty_analysis/{field}/{scn}/{attribute}/daily_outlet_best.csv'
    df_model = pd.read_csv(file)
    df_model.index = pd.to_datetime(df_model.Date)
    df_model = df_model.drop('Date', axis=1)
    df_model = pd.DataFrame(df_model[parameter])
    return df_model


def get_uncertainty_stats(field, is_grazing):
    if is_grazing:
        scn = 'pyAPEX_g'
    else:
        scn = 'pyAPEX_n'
    file_un = f'../{field}/{scn}/pyAPEX/OutputUncertainty/Statistics_runoff.csv'
    file_cal = f'../{field}/{scn}/pyAPEX/Output/Statistics_runoff.csv'
    df_stats_cal = pd.read_csv(file_cal, index_col=0)
    df_stats_un = pd.read_csv(file_un, index_col=0)
    NSEAD_cal = np.array(df_stats_cal.NSEAD)
    max_nash_cal = ep.nanmax(NSEAD_cal)
    glue_NSE_cal = np.exp(-NSEAD_cal / max_nash_cal)

    NSEAD_un = np.array(df_stats_un.NSEAD)
    max_nash_un = ep.nanmax(NSEAD_un)
    glue_NSE_un = np.exp(-NSEAD_un / max_nash_un)
    glue_NSE_best = glue_NSE_un[0]
    glue_NSE_un = glue_NSE_un[1:]

    return glue_NSE_un, glue_NSE_best, glue_NSE_cal


df_gl_1_n = get_glue_data(field='Farm_1', is_grazing=False)
df_gl_1_g = get_glue_data(field='Farm_1', is_grazing=True)
df_gl_8_n = get_glue_data(field='Farm_8', is_grazing=False)
df_gl_8_g = get_glue_data(field='Farm_8', is_grazing=True)

glue_NSE_1_n, glue_MSE_1_n = evaluate_best_glue(field='Farm_1', is_grazing=False, df_glue=df_gl_1_n, attribute='runoff',
                                                file='Calibration_data.csv', parameter='WYLD')
glue_NSE_1_g, glue_MSE_1_g = evaluate_best_glue(field='Farm_1', is_grazing=True, df_glue=df_gl_1_n, attribute='runoff',
                                                file='Calibration_data.csv', parameter='WYLD')
glue_NSE_8_n, glue_MSE_8_n = evaluate_best_glue(field='Farm_8', is_grazing=False, df_glue=df_gl_1_n, attribute='runoff',
                                                file='Calibration_data.csv', parameter='WYLD')
glue_NSE_8_g, glue_MSE_8_g = evaluate_best_glue(field='Farm_8', is_grazing=True, df_glue=df_gl_1_n, attribute='runoff',
                                                file='Calibration_data.csv', parameter='WYLD')
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True, tight_layout=True)
sns.histplot(df_gl_1_n.Likelihood_NSE, ax=axes[0, 0], stat="probability", kde=True)
axes[0, 0].axvline(glue_NSE_1_n, color='red')
axes[0, 0].set_xlabel("Likelihood estimate", fontsize=16)
axes[0, 0].set_ylabel("WRE1", fontsize=16)
axes[0, 0].set_title("Without grazing", fontsize=18)
axes[0, 0].tick_params(axis='both', which='major', labelsize=12)

sns.histplot(df_gl_1_g.Likelihood_NSE, ax=axes[0, 1], stat="probability", kde=True)
axes[0, 1].axvline(glue_NSE_1_g, color='red')
axes[0, 1].set_xlabel("Likelihood estimate", fontsize=16)
axes[0, 1].set_title("With grazing", fontsize=16)
axes[0, 1].tick_params(axis='both', which='major', labelsize=12)

sns.histplot(df_gl_8_n.Likelihood_NSE, ax=axes[1, 0], stat="probability", kde=True)
axes[1, 0].axvline(glue_NSE_8_n, color='red')
axes[1, 0].set_xlabel("Likelihood estimate", fontsize=16)
axes[1, 0].set_ylabel("WRE8", fontsize=18)
axes[1, 0].tick_params(axis='both', which='major', labelsize=12)

sns.histplot(df_gl_8_g.Likelihood_NSE, ax=axes[1, 1], stat="probability", kde=True)
axes[1, 1].axvline(glue_NSE_8_g, color='red')
axes[1, 1].set_xlabel("Likelihood estimate", fontsize=16)
axes[1, 1].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_likelihood_histogram.png'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
glue_NSE_un_1_n, glue_NSE_best_1_n, glue_NSE_cal_1_n = get_uncertainty_stats(field='Farm_1', is_grazing=False)
sns.histplot(glue_NSE_un_1_n, ax=axes[0, 0], stat="probability", kde=True)
axes[0, 0].axvline(glue_NSE_best_1_n, color='red')
axes[0, 0].set_xlabel("Likelihood estimate")
axes[0, 0].set_ylabel("WRE1")
axes[0, 0].set_title("Without grazing")

glue_NSE_un_1_g, glue_NSE_best_1_g, glue_NSE_cal_1_g = get_uncertainty_stats(field='Farm_1', is_grazing=True)
sns.histplot(glue_NSE_un_1_g, ax=axes[0, 1], stat="probability", kde=True)
axes[0, 1].axvline(glue_NSE_best_1_g, color='red')
axes[0, 1].set_xlabel("Likelihood estimate")
axes[0, 1].set_title("With grazing")

glue_NSE_un_8_n, glue_NSE_best_8_n, glue_NSE_cal_8_n = get_uncertainty_stats(field='Farm_8', is_grazing=False)
sns.histplot(glue_NSE_un_8_n, ax=axes[1, 0], stat="probability", kde=True)
axes[1, 0].axvline(glue_NSE_best_8_n, color='red')
axes[1, 0].set_xlabel("Likelihood estimate")
axes[1, 0].set_ylabel("WRE8")

glue_NSE_un_8_g, glue_NSE_best_8_g, glue_NSE_cal_8_g = get_uncertainty_stats(field='Farm_8', is_grazing=True)
sns.histplot(glue_NSE_un_8_g, ax=axes[1, 1], stat="probability", kde=True)
axes[1, 1].axvline(glue_NSE_best_8_g, color='red')
axes[1, 1].set_xlabel("Likelihood estimate")
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_raw_likelihood_histogram.png'),
            dpi=600, bbox_inches="tight")
plt.show()
