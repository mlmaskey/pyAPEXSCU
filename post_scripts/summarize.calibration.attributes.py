import warnings
import os

import pandas as pd

from utility import compile_gather

warnings.filterwarnings('ignore')

metric = 'OF'
data_daily_1_n, data_annual_1_n, df_summary_1_n = compile_gather(site='Farm_1', scenario='non_grazing',
                                                                 attribute='runoff', scale='daily', metric=metric,
                                                                 task='Calibration')
data_daily_1_g, data_annual_1_g, df_summary_1_g = compile_gather(site='Farm_1', scenario='grazing', attribute='runoff',
                                                                 scale='daily', metric=metric, task='Calibration')
data_daily_8_n, data_annual_8_n, df_summary_8_n = compile_gather(site='Farm_8', scenario='non_grazing',
                                                                 attribute='runoff', scale='daily', metric=metric,
                                                                 task='Calibration')
data_daily_8_g, data_annual_8_g, df_summary_8_g = compile_gather(site='Farm_8', scenario='grazing', attribute='runoff',
                                                                 scale='daily', metric=metric, task='Calibration')
df_summary = pd.concat([df_summary_1_n, df_summary_1_g, df_summary_8_n, df_summary_8_g], axis=0)
file_save = os.path.join('../post_analysis/Results', f'Response_variable_summary_{metric}.csv')
df_summary.to_csv(file_save)

metric = 'NSE'
data_daily_1_n, data_annual_1_n, df_summary_1_n = compile_gather(site='Farm_1', scenario='non_grazing',
                                                                 attribute='runoff', scale='daily', metric=metric,
                                                                 task='Calibration')
data_daily_1_g, data_annual_1_g, df_summary_1_g = compile_gather(site='Farm_1', scenario='grazing', attribute='runoff',
                                                                 scale='daily', metric=metric, task='Calibration')
data_daily_8_n, data_annual_8_n, df_summary_8_n = compile_gather(site='Farm_8', scenario='non_grazing',
                                                                 attribute='runoff', scale='daily', metric=metric,
                                                                 task='Calibration')
data_daily_8_g, data_annual_8_g, df_summary_8_g = compile_gather(site='Farm_8', scenario='grazing', attribute='runoff',
                                                                 scale='daily', metric=metric, task='Calibration')
df_summary = pd.concat([df_summary_1_n, df_summary_1_g, df_summary_8_n, df_summary_8_g], axis=0)
file_save = os.path.join('../post_analysis/Results', f'Response_variable_summary_{metric}.csv')
df_summary.to_csv(file_save)

metric = 'COD'
data_daily_1_n, data_annual_1_n, df_summary_1_n = compile_gather(site='Farm_1', scenario='non_grazing',
                                                                 attribute='runoff', scale='daily', metric=metric,
                                                                 task='Calibration')
data_daily_1_g, data_annual_1_g, df_summary_1_g = compile_gather(site='Farm_1', scenario='grazing', attribute='runoff',
                                                                 scale='daily', metric=metric, task='Calibration')
data_daily_8_n, data_annual_8_n, df_summary_8_n = compile_gather(site='Farm_8', scenario='non_grazing',
                                                                 attribute='runoff', scale='daily', metric=metric,
                                                                 task='Calibration')
data_daily_8_g, data_annual_8_g, df_summary_8_g = compile_gather(site='Farm_8', scenario='grazing', attribute='runoff',
                                                                 scale='daily', metric=metric, task='Calibration')
df_summary = pd.concat([df_summary_1_n, df_summary_1_g, df_summary_8_n, df_summary_8_g], axis=0)
file_save = os.path.join('../post_analysis/Results', f'Response_variable_summary_{metric}.csv')
df_summary.to_csv(file_save)

metric = 'PBIAS'
data_daily_1_n, data_annual_1_n, df_summary_1_n = compile_gather(site='Farm_1', scenario='non_grazing',
                                                                 attribute='runoff', scale='daily', metric=metric,
                                                                 task='Calibration')
data_daily_1_g, data_annual_1_g, df_summary_1_g = compile_gather(site='Farm_1', scenario='grazing', attribute='runoff',
                                                                 scale='daily', metric=metric, task='Calibration')
data_daily_8_n, data_annual_8_n, df_summary_8_n = compile_gather(site='Farm_8', scenario='non_grazing',
                                                                 attribute='runoff', scale='daily', metric=metric,
                                                                 task='Calibration')
data_daily_8_g, data_annual_8_g, df_summary_8_g = compile_gather(site='Farm_8', scenario='grazing', attribute='runoff',
                                                                 scale='daily', metric=metric, task='Calibration')
df_summary = pd.concat([df_summary_1_n, df_summary_1_g, df_summary_8_n, df_summary_8_g], axis=0)
file_save = os.path.join('../post_analysis/Results', f'Response_variable_summary_{metric}.csv')
df_summary.to_csv(file_save)