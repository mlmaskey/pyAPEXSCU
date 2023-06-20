# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:52:33 2023

@author: Mahesh.Maskey
"""

import pandas as pd
import numpy as np
from utility import nash, SeabornFig2Grid
from easypy import easypy as ep
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

print('\014')
graph_dir = '../post_analysis/Figures'


def read_model_data(field, is_grazing, measured_attribute, attribute):
    if is_grazing:
        scn = 'grazing'
    else:
        scn = 'non_grazing'
    file = f'../post_analysis/Uncertainty_analysis/{field}/{scn}/{measured_attribute}/daily_{attribute}.csv'
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.drop('Stage', axis=1)
    return df


def read_measured(field, is_grazing, file, measured_attribute):
    if is_grazing:
        scenario = 'pyAPEX_g'
    else:
        scenario = 'pyAPEX_n'
    obs_path = f'../{field}/{scenario}/pyAPEX/Program/'
    df = get_measure(obs_path, file)
    df = df.drop('Date', axis=1)
    df.index = pd.to_datetime(df.index)
    df = pd.DataFrame(df[measured_attribute])
    return df


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


def get_glue_data(field, is_grazing):
    if is_grazing:
        scn = 'g'
    else:
        scn = 'n'
    read_dir = f'../post_analysis/Results/{field}_{scn}_Uncertainty_range.csv'
    df_gl = pd.read_csv(read_dir, index_col=0)

    return df_gl


def evaluate_best_glue(field='Farm_1', is_grazing=False, df_glue=None, attribute='runoff', file='Calibration_data.csv',
                       parameter='WYLD'):
    df_obs = read_measured(field, is_grazing, file, attribute)
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


def get_plot_glue_error(field, is_grazing, file, measured_attribute, parameter, xlim, ylim):
    df_model = read_model_data(field, is_grazing, measured_attribute, parameter)
    df_obs = read_measured(field, is_grazing, file, measured_attribute)
    df_best = get_best_data(field, is_grazing, measured_attribute, parameter)
    df_obs = df_obs[df_obs.index >= df_model.index[0]]
    df_model_annual = pd.DataFrame(df_model.resample('Y').sum().mean())
    df_obs_annual = df_obs.resample('Y').sum().mean()
    df_best_annual = pd.DataFrame(df_best.resample('Y').sum().mean())
    diff_best = (df_obs_annual.values - df_best_annual.values) / df_obs_annual.values

    diff = (df_obs_annual.values - df_model_annual) / df_obs_annual.values
    df_gl = get_glue_data(field, is_grazing)
    gl_nse = df_gl.Likelihood_NSE.values
    _, glue_NSE_best, _ = get_uncertainty_stats(field, is_grazing)

    df_scatter = pd.DataFrame({'Likelihood': gl_nse, 'Annual Change': diff.T.values[0]})
    g = sns.JointGrid(data=df_scatter, x="Likelihood", y="Annual Change")
    g.plot_marginals(sns.distplot, kde=True)
    g.plot_joint(plt.scatter, s=10, color='red')
    g.ax_joint.axvline(x=glue_NSE_best)
    g.ax_joint.axhline(y=diff_best[0][0])
    g.ax_marg_x.set_xlim(xlim)
    g.ax_marg_y.set_ylim(ylim)
    g.ax_marg_x.set_xlabel("Likelihood", fontsize=16)
    g.ax_marg_y.set_ylabel("Annual Change", fontsize=16)
    g.ax_marg_x.tick_params(axis='x', labelsize=14)
    g.ax_marg_y.tick_params(axis='y', labelsize=14)
    return g


g_1_n = get_plot_glue_error(field='Farm_1',
                            is_grazing=False,
                            file='Calibration_data.csv',
                            measured_attribute='runoff',
                            parameter='WYLD',
                            xlim=(0.25, 1),
                            ylim=(-2.5, 0.5))
g_1_g = get_plot_glue_error(field='Farm_1',
                            is_grazing=True,
                            file='Calibration_data.csv',
                            measured_attribute='runoff',
                            parameter='WYLD',
                            xlim=(0.25, 1),
                            ylim=(-2.5, 0.5))
g_8_n = get_plot_glue_error(field='Farm_8',
                            is_grazing=False,
                            file='Calibration_data.csv',
                            measured_attribute='runoff',
                            parameter='WYLD',
                            xlim=(0.3, 0.7),
                            ylim=(-2.0, 0.5))
g_8_g = get_plot_glue_error(field='Farm_8',
                            is_grazing=True,
                            file='Calibration_data.csv',
                            measured_attribute='runoff',
                            parameter='WYLD',
                            xlim=(0.3, 0.7),
                            ylim=(-2.0, 0.5))

g_1_n.savefig(os.path.join(graph_dir, 'Uncertaintity_Farm_1_GLUE_error_non_grazing.png'),
              dpi=600, bbox_inches="tight")
g_1_g.savefig(os.path.join(graph_dir, 'Uncertaintity_Farm_1_GLUE_error_grazing.png'),
              dpi=600, bbox_inches="tight")
g_8_n.savefig(os.path.join(graph_dir, 'Uncertaintity_Farm_8_GLUE_error_non_grazing.png'),
              dpi=600, bbox_inches="tight")
g_8_g.savefig(os.path.join(graph_dir, 'Uncertaintity_Farm_8_GLUE_error_grazing.png'),
              dpi=600, bbox_inches="tight")

f, axarr = plt.subplots(2, 2, figsize=(25, 16))
import matplotlib.image as mpimg

axarr[0, 0].imshow(mpimg.imread(os.path.join(graph_dir, 'Uncertaintity_Farm_1_GLUE_error_non_grazing.png')))
axarr[0, 1].imshow(mpimg.imread(os.path.join(graph_dir, 'Uncertaintity_Farm_1_GLUE_error_grazing.png')))
axarr[1, 0].imshow(mpimg.imread(os.path.join(graph_dir, 'Uncertaintity_Farm_8_GLUE_error_non_grazing.png')))
axarr[1, 1].imshow(mpimg.imread(os.path.join(graph_dir, 'Uncertaintity_Farm_8_GLUE_error_grazing.png')))

# turn off x and y axis
[ax.set_axis_off() for ax in axarr.ravel()]

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(25, 16))
gs = gridspec.GridSpec(2, 2)

mg0 = SeabornFig2Grid(g_1_n, fig, gs[0])
mg1 = SeabornFig2Grid(g_1_g, fig, gs[1])
mg2 = SeabornFig2Grid(g_8_n, fig, gs[3])
mg3 = SeabornFig2Grid(g_8_g, fig, gs[2])

gs.tight_layout(fig)
g_8_g.savefig(os.path.join(graph_dir, 'Uncertaintity_GLUE_error_grazing.png'),
              dpi=600, bbox_inches="tight")

plt.show()
