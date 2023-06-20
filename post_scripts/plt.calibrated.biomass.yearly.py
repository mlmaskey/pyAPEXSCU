import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from utility import compile_gather, plot_sns_lines, plot_sns_scatter, plot_sns_2line
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

max_1_n_BIOM, max_1_g_BIOM = run_data_1_n['BIOM'].max(), run_data_1_g['BIOM'].max()
max_1_BIOM = max([max_1_n_BIOM, max_1_g_BIOM]) * 1.05
max_1_n_LAI, max_1_g_LAI = run_data_1_n['LAI'].max(), run_data_1_g['LAI'].max()
max_1_LAI = max([max_1_n_LAI, max_1_g_LAI]) * 1.05

max_8_n_BIOM, max_8_g_BIOM = run_data_8_n['BIOM'].max(), run_data_8_g['BIOM'].max()
max_8_BIOM = max([max_8_n_BIOM, max_8_g_BIOM]) * 1.05
max_8_n_LAI, max_8_g_LAI = run_data_8_n['LAI'].max(), run_data_8_g['LAI'].max()
max_8_LAI = max([max_8_n_LAI, max_8_g_LAI]) * 1.05

fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=False, tight_layout=True)
axes[0, 0] = plot_sns_lines(df=run_data_1_n, x_var='DOY', y_var='BIOM', hue='Y', ax=axes[0, 0],
                            str_title='WRE1: Grassland without grazing', x_lab='', y_lab='Biomass, t/ha',
                            y_peak=max_1_BIOM)
axes[1, 0] = plot_sns_lines(df=run_data_1_n, x_var='DOY', y_var='LAI', hue='Y', ax=axes[1, 0],
                            str_title='', x_lab='Day of year', y_lab='Leaf area index', y_peak=max_1_LAI)
axes[0, 1] = plot_sns_lines(df=run_data_1_g, x_var='DOY', y_var='BIOM', hue='Y', ax=axes[0, 1],
                            str_title='WRE1: Grassland with grazing', x_lab='', y_lab='', y_peak=max_1_BIOM)
axes[1, 1] = plot_sns_lines(df=run_data_1_g, x_var='DOY', y_var='LAI', hue='Y', ax=axes[1, 1],
                            str_title='', x_lab='Day of year', y_lab='', y_peak=max_1_LAI)
axes[1, 1].legend(loc='right', bbox_to_anchor=(1.5, 0.5), ncol=2)
plt.tight_layout()
plt.savefig('../post_analysis/Figures/Calibration_Farm_1_biomass_annual_daily_timeseries_OF.png', dpi=600,
            bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=False, tight_layout=True)
axes[0, 0] = plot_sns_lines(df=run_data_8_n, x_var='DOY', y_var='BIOM', hue='Y', ax=axes[0, 0],
                            str_title='WRE8: Cropland without grazing', x_lab='', y_lab='Biomass, t/ha',
                            y_peak=max_8_BIOM)
axes[1, 0] = plot_sns_lines(df=run_data_8_n, x_var='DOY', y_var='LAI', hue='Y', ax=axes[1, 0],
                            str_title='', x_lab='Day of year', y_lab='Leaf area index', y_peak=max_8_LAI)
axes[0, 1] = plot_sns_lines(df=run_data_8_g, x_var='DOY', y_var='BIOM', hue='Y', ax=axes[0, 1],
                            str_title='WRE8: Cropland with grazing', x_lab='', y_lab='', y_peak=max_8_BIOM)
axes[1, 1] = plot_sns_lines(df=run_data_8_g, x_var='DOY', y_var='LAI', hue='Y', ax=axes[1, 1],
                            str_title='', x_lab='Day of year', y_lab='', y_peak=max_8_LAI)
axes[1, 1].legend(loc='right', bbox_to_anchor=(1.5, 0.5), ncol=2)
plt.tight_layout()
plt.savefig('../post_analysis/Figures/Calibration_Farm_8_biomass_annual_daily_timeseries_OF.png', dpi=600,
            bbox_inches="tight")
plt.show()

