# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 15:55:09 2023

@author: Mahesh.Maskey
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import os
from utility import print_progress_bar

warnings.filterwarnings('ignore')
print('\014')
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)


def summarize_uncertainty_attribute(site, scenario, attribute, location):
    if location == 'annual':
        file = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/Annual_{attribute}.csv'
        file_best = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/{location}_best.csv'
    else:
        file = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/Daily_{attribute}.csv'
        file_best = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/daily_{location}_best.csv'

    data = pd.read_csv(file, index_col=0)
    data_best = pd.read_csv(file_best, index_col=0)
    if location != 'annual':
        data.index = pd.to_datetime(data.index)
        cal_data = data[data.Stage == 'Calibration']
        val_data = data[data.Stage == 'Validation']
        cal_start_date = cal_data.index[0]
        cal_end_date = cal_data.index[0]
        val_start_date = val_data.index[0]
        val_end_date = val_data.index[-1]
        df_data = data.resample('Y').sum()
        df_cal = cal_data.resample('Y').sum()
        df_val = val_data.resample('Y').sum()
        data_best.index = pd.to_datetime(data_best.index)
        data_best = pd.DataFrame(data_best[attribute])
        df_best = data_best[(data_best.index >= cal_start_date) & (data_best.index <= val_end_date)]
        cal_best = data_best[(data_best.index >= cal_start_date) & (data_best.index <= cal_end_date)]
        val_best = data_best[(data_best.index >= val_start_date) & (data_best.index <= val_end_date)]
        df_cal_best = cal_best.resample('Y').sum()
        df_val_best = val_best.resample('Y').sum()
        df_best = df_best.resample('Y').sum()
    else:
        cal_data = data[data.Stage == 'Calibration']
        val_data = data[data.Stage == 'Validation']
        cal_start_date = cal_data.index[0]
        cal_end_date = cal_data.index[0]
        val_start_date = val_data.index[0]
        val_end_date = val_data.index[-1]
        df_data = data
        df_cal = cal_data
        df_val = val_data
        data_best = pd.DataFrame(data_best[attribute])
        df_best = data_best[(data_best.index >= cal_start_date) & (data_best.index <= val_end_date)]
        cal_best = data_best[(data_best.index >= cal_start_date) & (data_best.index <= cal_end_date)]
        val_best = data_best[(data_best.index >= val_start_date) & (data_best.index <= val_end_date)]
        df_cal_best = cal_best
        df_val_best = val_best
        df_best = df_best

    mean_data = df_data.mean(axis=1).mean()
    sd_data = df_data.min(axis=1).std()

    min_data = df_data.min(axis=1).min()
    max_data = df_data.max(axis=1).max()

    mean_cal = df_cal.mean(axis=1).mean()
    sd_cal = df_cal.mean(axis=1).std()
    min_cal = df_cal.min(axis=1).min()
    max_cal = df_cal.max(axis=1).max()

    mean_val = df_val.mean(axis=1).mean()
    sd_val = df_val.mean(axis=1).std()
    min_val = df_val.min(axis=1).min()
    max_val = df_val.max(axis=1).max()

    best = np.array(df_best.mean())[0]
    calbest = np.array(df_cal_best.mean())[0]
    valbest = np.array(df_val_best.mean())[0]
    sum_list_all = [min_data, mean_data, max_data, sd_data, best]
    sum_list_cal = [min_cal, mean_cal, max_cal, sd_cal, calbest]
    sum_list_val = [min_val, mean_val, max_val, sd_val, valbest]

    df_all = pd.DataFrame(sum_list_all).T
    df_all.columns = ['Minimum', 'Mean', 'Maximum', 'Standard deviation', 'Calibrated']
    df_all.index = [attribute]
    df_all['Stage'] = 'Entire'
    df_cal = pd.DataFrame(sum_list_cal).T
    df_cal.columns = ['Minimum', 'Mean', 'Maximum', 'Standard deviation', 'Calibrated']
    df_cal.index = [attribute]
    df_cal['Stage'] = 'Calibration'
    df_val = pd.DataFrame(sum_list_val).T
    df_val.columns = ['Minimum', 'Mean', 'Maximum', 'Standard deviation', 'Calibrated']
    df_val.index = [attribute]
    df_val['Stage'] = 'Validation'
    df_out = pd.concat([df_all, df_cal, df_val], axis=0)
    return df_out


def compile_stats(site, scenario, attribute, location):
    df_out = summarize_uncertainty_attribute(site, scenario, attribute, location)
    df_out['Site'] = site
    df_out['Operation'] = scenario
    return df_out


list_attributes = ['WYLD', 'YSD', 'RUS2',
                   'DPRK', 'ET', 'PET', 'BIOM', 'STD', 'STL', 'TN', 'TP',
                   'YLDF', 'YLDG', 'WS', 'TS', 'NS', 'PS']
list_locations = ['outlet', 'outlet', 'outlet',
                  'basin', 'basin', 'basin', 'basin', 'basin', 'basin', 'basin', 'basin',
                  'annual', 'annual', 'annual', 'annual', 'annual', 'annual']

df_out_1_n = pd.DataFrame()
df_out_1_g = pd.DataFrame()
df_out_8_n = pd.DataFrame()
df_out_8_g = pd.DataFrame()
print_progress_bar(0, len(list_attributes), prefix='Progress:', suffix='Complete', length=50)
for i in range(len(list_attributes)):
    df_i_out = compile_stats(site='Farm_1',
                             scenario='non_grazing',
                             attribute=list_attributes[i],
                             location=list_locations[i])
    df_out_1_n = pd.concat([df_out_1_n, df_i_out], axis=0)
    del df_i_out
    df_i_out = compile_stats(site='Farm_1',
                             scenario='grazing',
                             attribute=list_attributes[i],
                             location=list_locations[i])
    df_out_1_g = pd.concat([df_out_1_g, df_i_out], axis=0)
    del df_i_out
    df_i_out = compile_stats(site='Farm_8',
                             scenario='non_grazing',
                             attribute=list_attributes[i],
                             location=list_locations[i])
    df_out_8_n = pd.concat([df_out_8_n, df_i_out], axis=0)
    del df_i_out
    df_i_out = compile_stats(site='Farm_8',
                             scenario='grazing',
                             attribute=list_attributes[i],
                             location=list_locations[i])
    df_out_8_g = pd.concat([df_out_8_g, df_i_out], axis=0)
    del df_i_out

    print_progress_bar(i, len(list_attributes), prefix=f'Progress {i}:', suffix='Complete', length=50)

save_data_dir = '..\post_analysis\Results'
if not os.path.isdir(save_data_dir):
    os.makedirs(save_data_dir)

df_out = pd.concat([df_out_1_n, df_out_1_g, df_out_8_n, df_out_8_g], axis=0)
df_out.to_csv(save_data_dir, 'UncertaintySummary.csv')
