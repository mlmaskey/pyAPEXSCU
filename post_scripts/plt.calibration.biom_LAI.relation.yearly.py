import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from utility import compile_gather, plot_sns_lines, plot_sns_scatter, plot_sns_2line, _get_data__
from utility import legend_outside

warnings.filterwarnings('ignore')

data_daily_1_n, data_annual_1_n, df_summary_1_n = compile_gather(site='Farm_1', scenario='non_grazing',
                                                                 attribute='runoff', scale='daily', metric='OF',
                                                                 task='Calibration')
data_daily_1_g, data_annual_1_g, df_summary_1_g = compile_gather(site='Farm_1', scenario='grazing', attribute='runoff',
                                                                 scale='daily', metric='OF', task='Calibration')
data_daily_8_n, data_annual_8_n, df_summary_8_n = compile_gather(site='Farm_8', scenario='non_grazing',
                                                                 attribute='runoff', scale='daily', metric='OF',
                                                                 task='Calibration')
data_daily_8_g, data_annual_8_g, df_summary_8_g = compile_gather(site='Farm_8', scenario='grazing', attribute='runoff',
                                                                 scale='daily', metric='OF', task='Calibration')

run_data_1_n = data_daily_1_n['Entire']
run_data_1_g = data_daily_1_g['Entire']
run_data_8_n = data_daily_8_n['Entire']
run_data_8_g = data_daily_8_g['Entire']

graph_dir = '../post_analysis/Figures/LAI_BIOM/Farm_1'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)

years = run_data_1_n.Y.unique()
variable_list = ['BIOM', 'LAI']
variable_list_describe = ['Biomass, t/ha', 'Leaf area index']

for i in range(len(years)):
    plot_data_n, max_n, min_n = _get_data__(run_data_1_n, var_list=variable_list, year=years[i])
    plot_data_g, max_g, min_g = _get_data__(run_data_1_g, var_list=variable_list, year=years[i])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
    axes[0] = plot_sns_2line(plot_data_n, plot_data_g, ax=axes[0], x_var='DOY', y_var=variable_list[0],
                             x_lab='Day of year', y_lab=variable_list_describe[0], str_title=None)
    axes[0].get_legend().remove()
    axes[1] = plot_sns_2line(plot_data_n, plot_data_g, ax=axes[1], x_var='DOY', y_var=variable_list[1],
                             x_lab='Day of year', y_lab=variable_list_describe[1], str_title=f'Year: {years[i]}')
    axes[1].get_legend().remove()
    axes[2] = plot_sns_scatter(plot_data_n, plot_data_g, variable_list, ax=axes[2], x_lab=variable_list_describe[0],
                               y_lab=variable_list_describe[1], str_title=None)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    file_save = os.path.join(graph_dir, f'{years[i]}_year_{variable_list[0]}_{variable_list[1]}.png')
    plt.tight_layout()
    plt.savefig(file_save, dpi=600,  bbox_inches="tight")

del plot_data_n, max_n, min_n, plot_data_g, max_g, min_g

graph_dir = '../post_analysis/Figures/LAI_BIOM/Farm_8'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)
years = run_data_8_n.Y.unique()

for i in range(len(years)):
    plot_data_n, max_n, min_n = _get_data__(run_data_8_n, var_list=variable_list, year=years[i])
    plot_data_g, max_g, min_g = _get_data__(run_data_8_g, var_list=variable_list, year=years[i])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
    axes[0] = plot_sns_2line(plot_data_n, plot_data_g, ax=axes[0], x_var='DOY', y_var=variable_list[0],
                             x_lab='Day of year', y_lab=variable_list_describe[0], str_title=None)
    axes[0].get_legend().remove()
    axes[1] = plot_sns_2line(plot_data_n, plot_data_g, ax=axes[1], x_var='DOY', y_var=variable_list[1],
                             x_lab='Day of year', y_lab=variable_list_describe[1], str_title=f'Year: {years[i]}')
    axes[1].get_legend().remove()
    axes[2] = plot_sns_scatter(plot_data_n, plot_data_g, variable_list, ax=axes[2], x_lab=variable_list_describe[0],
                               y_lab=variable_list_describe[1], str_title=None)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    file_save = os.path.join(graph_dir, f'{years[i]}_year_{variable_list[0]}_{variable_list[1]}.png')
    plt.tight_layout()
    plt.savefig(file_save, dpi=600,  bbox_inches="tight")