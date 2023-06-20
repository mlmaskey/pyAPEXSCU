# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 08:48:07 2023

@author: Mahesh.Maskey
"""
import collections

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import os
from utility import plot_daily_timeseries, legend_outside

warnings.filterwarnings('ignore')
print('\014')
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)
# Farm 1  based on temporal optimization at different metrics
sites = ['Farm_1', 'Farm_8']
scenario = ['pyAPEX_n', 'pyAPEX_g']
attribute = 'runoff'
read_dir_n_1 = f'../{sites[0]}/{scenario[0]}/pyAPEX/Output/{attribute}/'
read_dir_g_1 = f'../{sites[0]}/{scenario[1]}/pyAPEX/Output/{attribute}/'
read_dir_n_8 = f'../{sites[1]}/{scenario[1]}/pyAPEX/Output/{attribute}/'
read_dir_g_8 = f'../{sites[1]}/{scenario[1]}/pyAPEX/Output/{attribute}/'
file_lists4farm1 = [f'{read_dir_n_1}model_calibration_OF.csv', f'{read_dir_n_1}/model_calibration_NSE.csv',
                    f'{read_dir_n_1}/model_calibration_COD.csv', f'{read_dir_n_1}/model_calibration_PBIAS.csv',
                    f'{read_dir_g_1}/model_calibration_OF.csv',  f'{read_dir_g_1}/model_calibration_NSE.csv',
                    f'{read_dir_g_1}/model_calibration_COD.csv', f'{read_dir_g_1}/model_calibration_PBIAS.csv']

title_lists = ['Objective function', 'NSE', 'COD', 'PBAIS']
fig, axes = plt.subplots(4, 2, figsize=(20, 10), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[0], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm1[1], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm1[2], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 0],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm1[3], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 0],
                      showlegend=False, title=title_lists[3])
plot_daily_timeseries(filepath=file_lists4farm1[4], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm1[5], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm1[6], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 1],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm1[7], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 1],
                      showlegend='brief', title=title_lists[3])
axes[3, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_Timeseries_daily_daily.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_Timeseries_daily_daily.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_Timeseries_daily_daily.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(4, 2, figsize=(20, 10), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[0], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm1[1], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm1[2], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 0],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm1[3], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 0],
                      showlegend=False, title=title_lists[3])
plot_daily_timeseries(filepath=file_lists4farm1[4], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm1[5], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm1[6], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 1],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm1[7], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 1],
                      showlegend='brief', title=title_lists[3])
axes[3, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_Timeseries_daily_monthly.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_Timeseries_daily_monthly.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_Timeseries_daily_monthly.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(4, 2, figsize=(20, 10), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[0], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm1[1], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm1[2], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 0],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm1[3], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 0],
                      showlegend=False, title=title_lists[3])
plot_daily_timeseries(filepath=file_lists4farm1[4], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm1[5], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm1[6], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 1],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm1[7], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 1],
                      showlegend='brief', title=title_lists[3])
axes[3, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_Timeseries_daily_yearly.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_Timeseries_daily_yearly.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_Timeseries_daily_yearly.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()
# plt.close('all')
# Farm 8  based on temporal optimization at different metrices
file_lists4farm8 = [f'{read_dir_n_8}/model_calibration_OF.csv',   f'{read_dir_n_8}/model_calibration_NSE.csv',
                    f'{read_dir_n_8}/model_calibration_COD.csv', f'{read_dir_n_8}/model_calibration_PBIAS.csv',
                    f'{read_dir_g_8}/model_calibration_OF.csv',  f'{read_dir_g_8}/model_calibration_NSE.csv',
                    f'{read_dir_g_8}/model_calibration_COD.csv', f'{read_dir_g_8}/model_calibration_PBIAS.csv']

fig, axes = plt.subplots(4, 2, figsize=(20, 10), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm8[0], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm8[1], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm8[2], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 0],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm8[3], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 0],
                      showlegend=False, title=title_lists[3])
plot_daily_timeseries(filepath=file_lists4farm8[4], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm8[5], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm8[6], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 1],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm8[7], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 1],
                      showlegend='brief', title=title_lists[3])
axes[3, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_Timeseries_daily_daily.png'),
            dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(4, 2, figsize=(20, 10), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm8[0], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm8[1], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm8[2], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 0],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm8[3], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 0],
                      showlegend=False, title=title_lists[3])
plot_daily_timeseries(filepath=file_lists4farm8[4], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm8[5], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm8[6], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 1],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm8[7], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 1],
                      showlegend='brief', title=title_lists[3])
axes[3, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_Timeseries_daily_monthly.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_Timeseries_daily_monthly.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_Timeseries_daily_monthly.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(4, 2, figsize=(20, 10), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm8[0], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm8[1], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm8[2], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 0],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm8[3], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 0],
                      showlegend=False, title=title_lists[3])
plot_daily_timeseries(filepath=file_lists4farm8[4], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title=title_lists[0])
plot_daily_timeseries(filepath=file_lists4farm8[5], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend=False, title=title_lists[1])
plot_daily_timeseries(filepath=file_lists4farm8[6], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[2, 1],
                      showlegend=False, title=title_lists[2])
plot_daily_timeseries(filepath=file_lists4farm8[7], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[3, 1],
                      showlegend='brief', title=title_lists[3])
axes[3, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_Timeseries_daily_yearly.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_Timeseries_daily_yearly.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_Timeseries_daily_yearly.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

# combined plots farme 1 and 2
fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[0], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[4], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[0], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[4], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_OF.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_OF.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_OF.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[1], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[5], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[1], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[5], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_NSE.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_NSE.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_NSE.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()


fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[2], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[6], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[2], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[6], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_COD.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_COD.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_COD.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[3], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[7], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[3], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[7], scale='daily', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_PBIAS.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_PBIAS.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_daily_PBIAS.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[0], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[4], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[0], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[4], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_OF.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_OF.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_OF.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[1], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[5], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[1], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[5], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_NSE.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_NSE.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_NSE.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[2], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[6], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[2], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[6], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_COD.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_COD.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_COD.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[3], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[7], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[3], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[7], scale='monthly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_PBIAS.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_PBIAS.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_monthly_PBIAS.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[0], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[4], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[0], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[4], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_OF.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_OF.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_OF.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[1], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[5], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[1], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[5], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_NSE.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_NSE.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_NSE.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[2], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[6], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[2], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[6], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_COD.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_COD.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_COD.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()

fig, axes = plt.subplots(2, 2, figsize=(20, 6), sharex=True, sharey=True)
plot_daily_timeseries(filepath=file_lists4farm1[3], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 0],
                      showlegend=False, title='WRE1 without grazing')
plot_daily_timeseries(filepath=file_lists4farm1[7], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[0, 1],
                      showlegend=False, title='WRE1 with grazing')
plot_daily_timeseries(filepath=file_lists4farm8[3], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 0],
                      showlegend=False, title='WRE8 without grazing')
plot_daily_timeseries(filepath=file_lists4farm8[7], scale='yearly', xlab='Date', y_lab='Runoff, mm', ax=axes[1, 1],
                      showlegend='brief', title='WRE8 with grazing')
axes[1, 1].legend_.remove()
plt.tight_layout()
legend_outside(axes=axes, fig=fig, loc='lower center', bbox_to_anchor=(0.5, 0), n_col=6, font_size="14")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_PBIAS.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_PBIAS.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1&8_Timeseries_daily_yearly_PBIAS.jpeg'), dpi=600, bbox_inches="tight")
# plt.show()
plt.close('all')
