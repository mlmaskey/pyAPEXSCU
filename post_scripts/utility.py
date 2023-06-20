# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:03:17 2022

@author: Mahesh.Maskey
"""
import os

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from configobj import ConfigObj
import collections


def pbias(ox, sx):
    return np.sum(ox - sx) * 100 / np.sum(ox)


def nancorr(x: list, y: list):
    '''
    Computes pearson correlation cofficient reardless of vectors
    contaning NaN values
    Parameters
    ----------
    x : list
        DESCRIPTION. First vector
    y : list
        DESCRIPTION. Secon Vector

    Returns
    -------
    corrcoef : Pearson correlation coefficient R-square
        DESCRIPTION.
    Example
    ------
    x = [5.798, 1.185, 5.217, 8.222, 6.164, 4.118, 2.465, 6.663, 2.153, 0.205]
    y = [6.162, 6.27, 8.529, 0.127, 5.29, 1.443, 5.189, 8.244, 6.266, 9.582]
    easypy.nancorr(x, y)
    x = [57.8, 11.85, 52.17, 82.22, np.nan, 45.18, 24.65, np.nan, 21.53, 2.00]
    y = [61.62, 62.07, np.nan, 20.17, 51.29, 11.43, 52.19, 82.44, 62.66, 95.82]
    easypy.nancorr(x, y)

    '''
    if len(x) != len(y):
        print(f'AssertionError: two vectors must be same length')
        return
    if isinstance(x, pd.core.series.Series):
        xnew = []
        for xv in x:
            xnew.append(xv[0])
    elif isinstance(y, pd.core.series.Series):
        ynew = []
        for yv in y:
            ynew.append(yv[0])
    else:
        xnew, ynew = x, y
    if isinstance(x, np.ndarray):
        xnew = []
        for xv in x:
            xnew.append(xv)
    elif isinstance(y, np.ndarray):
        ynew = []
        for yv in y:
            ynew.append(yv)
    else:
        xnew, ynew = x, y
    data = {'x': xnew, 'y': ynew}
    df = pd.DataFrame(data, index=range(len(x)))
    df = df.dropna()
    xnew1, ynew1 = df.x.to_numpy(), df.y.to_numpy()
    corr_matrix = np.corrcoef(xnew1, ynew1)
    corrcoef = corr_matrix[1, 0] ** 2
    return corrcoef


def rmse(ox, sx):
    RMSE = (np.sum((ox - sx) ** 2) / len(ox)) ** 0.5
    return RMSE


def nrmse(RMSE, ox):
    return RMSE / np.mean(ox)


def nash(ox, sx):
    omu = np.mean(ox)
    NSE = 1 - np.sum((ox - sx) ** 2) / np.sum((ox - omu) ** 2)
    if NSE < -1:
        NSE = -1
    return NSE


def get_plot_data_daily(file_name, scale):
    data_model = pd.read_csv(file_name)
    data_model.index = data_model.Date
    data_model_daily = data_model[data_model.SCALE == scale]
    data_model_daily.drop('SCALE', axis=1, inplace=True)
    data_model_daily.Date = pd.to_datetime(data_model_daily.Date)
    data_melt = pd.melt(data_model_daily, id_vars=['Date', 'STAGE'], value_vars=['Observed', 'Modeled'])
    data_melt.columns = ['Date', 'Stage', 'Mode', 'Runoff, mm']
    return data_melt


def plot_daily_timeseries(filepath, scale, xlab, y_lab, ax, showlegend, title):
    data_melt = get_plot_data_daily(filepath, scale)
    sns.lineplot(data=data_melt, x=xlab, y=y_lab, hue='Mode', style='Stage', palette=['r', 'b'], ax=ax,
                 legend=showlegend)
    ax.set_xlim(data_melt.Date[0], data_melt.Date[len(data_melt.Date) - 1])
    ax.set_xlabel(xlab, fontsize=16)
    ax.set_ylabel(y_lab, fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_title(title, fontsize=18)
    ax.grid(True)


def legend_outside(axes, fig, loc, bbox_to_anchor, n_col, font_size):
    # Adding legend outside the plot area
    entries = collections.OrderedDict()
    for ax in axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries[label] = handle
    legend = fig.legend(
        entries.values(), entries.keys(),
        loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=n_col, fontsize=font_size)
    bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    fig.tight_layout(rect=(0, bbox.y1, 1, 1), h_pad=0.5, w_pad=0.5)
    return legend


def get_plot_data_monthly(file_name, scale):
    data_model = pd.read_csv(file_name)
    data_model.index = data_model.Date
    data_model_monthly = data_model[data_model.SCALE == scale]
    data_model_monthly.drop('SCALE', axis=1, inplace=True)
    data_model_monthly.index = pd.to_datetime(data_model_monthly.index)
    data_calibration = data_model_monthly[data_model_monthly.STAGE == 'Calibration']
    df_calibration = data_calibration.resample('M').sum()
    data_validation = data_model_monthly[data_model_monthly.STAGE == 'Validation']
    df_validation = data_validation.resample('M').sum()
    df_calibration = pd.concat([df_calibration, pd.DataFrame(df_validation.iloc[0, :]).T], axis=0)
    df_calibration.insert(2, 'STAGE', 'Calibration', True)
    df_validation.insert(2, 'STAGE', 'Validation', True)
    data_model_monthly = pd.concat([df_calibration, df_validation], axis=0)
    data_model_monthly.insert(0, 'Month', data_model_monthly.index, True)
    data_melt = pd.melt(data_model_monthly, id_vars=['Month', 'STAGE'], value_vars=['Observed', 'Modeled'])
    data_melt.columns = ['Month', 'Stage', 'Mode', 'Runoff, mm']
    return data_melt


def plot_monthly_timeseries(filepath, scale, xlab, ylab, ax, showlegend, title):
    data_melt = get_plot_data_monthly(filepath, scale)
    sns.lineplot(data=data_melt, x=xlab, y=ylab, hue='Mode', style='Stage', palette=['r', 'b'], ax=ax,
                 legend=showlegend)
    ax.set_xlim(data_melt.Month[0], data_melt.Month[len(data_melt.Month) - 1])
    ax.set_title(title)
    ax.grid(True)


def get_plot_data_yearly(file_name, scale):
    data_model = pd.read_csv(file_name)
    data_model.index = data_model.Date
    data_model_yearly = data_model[data_model.SCALE == scale]
    data_model_yearly.drop('SCALE', axis=1, inplace=True)
    data_model_yearly.index = pd.to_datetime(data_model_yearly.index)
    data_calibration = data_model_yearly[data_model_yearly.STAGE == 'Calibration']
    df_calibration = data_calibration.resample('Y').sum()
    data_validation = data_model_yearly[data_model_yearly.STAGE == 'Validation']
    df_validation = data_validation.resample('Y').sum()
    df_calibration = pd.concat([df_calibration, pd.DataFrame(df_validation.iloc[0, :]).T], axis=0)
    df_calibration.insert(2, 'STAGE', 'Calibration', True)
    df_validation.insert(2, 'STAGE', 'Validation', True)
    data_model_yearly = pd.concat([df_calibration, df_validation], axis=0)
    data_model_yearly.insert(0, 'Year', data_model_yearly.index.year, True)
    data_melt = pd.melt(data_model_yearly, id_vars=['Year', 'STAGE'], value_vars=['Observed', 'Modeled'])
    data_melt.columns = ['Year', 'Stage', 'Mode', 'Runoff, mm']
    return data_melt


def plot_yearly_timeseries(filepath, scale, xlab, ylab, ax, showlegend, title):
    data_melt = get_plot_data_yearly(filepath, scale)
    sns.lineplot(data=data_melt, x=xlab, y=ylab, hue='Mode', style='Stage', palette=['r', 'b'], ax=ax,
                 legend=showlegend)
    ax.set_xlim(data_melt.Year[0], data_melt.Year[len(data_melt.Year) - 1])
    ax.set_title(title)
    ax.grid(True)


def read_param_file(file):
    with open(file) as f:
        # ref: https://www.delftstack.com/howto/python/python-readlines-without-newline/
        lines = f.read().splitlines()
    f.close()
    return lines


def txt2list(file):
    print(file)
    # read text file output from APEX
    with open(file, encoding="ISO-8859-1") as f:
        lines = f.readlines()
    f.close()
    # perse data from the text list above
    line_list = []
    for line in lines:
        l = list(line.split(' '))
        items = []
        for ele in l:
            if ele != '':
                items.append(ele)
        line_list.append(items)
    del line, lines
    return line_list


def import_weather_data(file_name, header=None, widths=[6, 4, 4, 6, 6, 6, 6]):
    df1 = pd.read_fwf(file_name, skiprows=2, encoding="ISO-8859-1", header=header, widths=widths)
    df1.columns = ['Y', 'M', 'D', 'Sr', 'Tx', 'Tn', 'PRCP']
    df1.insert(3, "Date", df1.Y.astype('str') + '/' + df1.M.astype('str') + '/' + df1.D.astype('str'))
    df1.Date = pd.to_datetime(df1.Date)
    df1.index = df1.Date
    df1 = df1.drop(['Date'], axis=1)
    return df1


def get_dir():
    return r'C:\Users\Mahesh.Maskey\Documents\Project\OklahomaWRE'


def get_output(site, is_graze, attribute, location, match_scale, scale, metric, var):
    data_dir = get_dir()
    if is_graze:
        file = f'{data_dir}/{site}/pyAPEX_g/pyAPEX/Output/{attribute}/{location}_{scale}_{metric}.csv'
    else:
        file = f'{data_dir}/{site}/pyAPEX_n/pyAPEX/Output/{attribute}/{location}_{scale}_{metric}.csv'
    df = pd.read_csv(file)
    if location == 'daily_outlet':
        df.index = df.Date
        df.index = pd.to_datetime(df.index)
        df = df.drop(['Date'], axis=1)
        df1 = df[['Y', 'M', 'D', var]]
        df2 = df1[var]
        df2 = pd.DataFrame(df2)
        return df1, df2
    elif location == 'daily_basin':
        df.index = df.Date
        df.index = pd.to_datetime(df.index)
        df = df.drop(['Date'], axis=1)
        df1 = df[['Y', 'M', 'D', 'CPNM', var]]
        croplist = df1.CPNM.unique()
        df_2, df2_filled = catgorize(df1, croplist, match_scale)
        return df1, df_2, df2_filled
    else:
        df.index = df.YR
        df = df.drop(['YR'], axis=1)
        df1 = df[['CPNM', var]]
        croplist = df1.CPNM.unique()
        df_2, df2_filled = catgorize(df1, croplist, match_scale)
        return df1, df_2, df2_filled


def catgorize(df_, croplist, match_scale):
    df = pd.DataFrame()
    for crop in croplist:
        df_c = df_[df_.CPNM == crop]
        if match_scale != 'annual':
            df_c = df_c.drop(['Y', 'M', 'D'], axis=1)
        df = pd.concat([df, df_c], axis=1)
    df = df.drop('CPNM', axis=1)
    df.columns = croplist
    df_final = df.fillna(0)
    df_final['Total'] = df_final.sum(axis=1)
    return df, df_final


def get_model_set(site, is_graze, attribute, location, scale, metric):
    if is_graze:
        file = f'../{site}/pyAPEX_g/pyAPEX/Output/{attribute}/{location}_{metric}.csv'
    else:
        file = f'../{site}/pyAPEX_n/pyAPEX/Output/{attribute}/{location}_{metric}.csv'
    df = pd.read_csv(file)
    df.index = df.Date
    df1 = df[df.SCALE == scale]
    df1.index = pd.to_datetime(df1.index)
    df1 = df1.drop(['Date', 'Modeled', 'SCALE'], axis=1)
    df_cal = df1[df1.STAGE == 'Calibration']
    df_val = df1[df1.STAGE == 'Validation']
    df1.STAGE = 'Observation'
    df1.columns = ['runoff', 'Stage']
    return df_cal, df_val, df1


def match_data(data, site, is_graze, attribute, location, match_scale, scale, metric):
    df_cal, df_val, df_obs = get_model_set(site, is_graze, attribute, location, scale, metric)
    if type(data) is pd.DataFrame:
        df_sim = data.copy()
    else:
        df_sim = pd.DataFrame(data)
    df_sim['Stage'] = 'Simulation'
    if match_scale == 'annual':
        start_cal, end_cal = df_cal.index[0].year, df_cal.index[-1].year
        start_val, end_val = df_val.index[0].year, df_val.index[-1].year
        df_sim = df_sim[df_sim.index >= start_cal]
        df_sim.Stage[(df_sim.index >= start_cal) & (df_sim.index <= end_cal)] = 'Calibration'
        df_sim.Stage[(df_sim.index >= start_val) & (df_sim.index <= end_val)] = 'Validation'
    else:
        start_cal, end_cal = df_cal.index[0], df_cal.index[-1]
        start_val, end_val = df_val.index[0], df_val.index[-1]
        df_sim = df_sim[df_sim.index >= start_cal]
        df_sim.Stage[(df_sim.index >= start_cal) & (df_sim.index <= end_cal)] = 'Calibration'
        df_sim.Stage[(df_sim.index >= start_val) & (df_sim.index <= end_val)] = 'Validation'
    df_mod_cal = df_sim[df_sim.Stage == 'Calibration']
    df_mod_val = df_sim[df_sim.Stage == 'Validation']
    df_mod = pd.concat([df_mod_cal, df_mod_val], axis=0)
    return df_mod, df_sim


def get_stress(site, is_graze, attribute, location, match_scale, scale, metric, var, management='Without grazing'):
    if location == 'daily_outlet':
        df, df_final = get_output(site, is_graze, attribute, location, match_scale, scale, metric, var)
        df1 = None
        df_final, _ = match_data(df_final, site=site, is_graze=is_graze, attribute=attribute,
                                 location='model_calibration', match_scale=match_scale,
                                 scale=scale, metric=metric)
    else:
        df, df1, df_final = get_output(site, is_graze, attribute, location, match_scale, scale, metric, var)
        df_final, _ = match_data(df_final, site=site, is_graze=is_graze, attribute=attribute,
                                 location='model_calibration', match_scale=match_scale,
                                 scale=scale, metric=metric)
        df_final = df_final[['Total', 'Stage']]
    df_final.insert(df_final.shape[1], 'Operation', management)
    df_final.columns = [var, 'Stage', 'Operation']
    return df, df1, df_final


def sumarize_output4sites(site1, site2, location, scale, match_scale, var, isAppend):
    if isAppend:
        f = open('../post_analysis/CalibrationSummary.txt', 'a')
    else:
        f = open('../post_analysis/CalibrationSummary.txt', 'w')
        f.writelines('---------Calibration and validation result summary of output variable\n')
    f.writelines(f'---------Output variable, {var}: \n')
    f.writelines(f'--------------------------- For {site1} ------------------------------------------------------\n')

    _, _, df_1_n = get_stress(site=site1, is_graze=False,
                              attribute='runoff', location=location,
                              match_scale=match_scale, scale=scale,
                              metric='OF', var=var, management='Without grazing')
    _, _, df_1_g = get_stress(site=site1, is_graze=True,
                              attribute='runoff', location=location,
                              match_scale=match_scale, scale=scale,
                              metric='OF', var=var, management='With grazing')

    df_1 = pd.DataFrame({'WRE1_n': df_1_n[var], 'WRE1_g': df_1_g[var]})
    df_1['Stage'] = df_1_n.Stage
    df_cal_1, df_val_1 = df_1[df_1.Stage == 'Calibration'], df_1[df_1.Stage == 'Validation']
    f.writelines(f'                     Calibration                     Validation                   Operation \n')
    if match_scale == 'annual':
        average_annual_cal_1 = pd.DataFrame(df_cal_1.mean())
        average_annual_val_1 = pd.DataFrame(df_val_1.mean())
        df_annual_average_1 = pd.concat([average_annual_cal_1, average_annual_val_1], axis=1)
        f.writelines(
            f'Average Annual:         {average_annual_cal_1.values[0][0]}                     {average_annual_val_1.values[0][0]}            without grazing\n')
        f.writelines(
            f'Average Annual:         {average_annual_cal_1.values[1][0]}                     {average_annual_val_1.values[1][0]}            with grazing\n')
        df_annual_average_1.columns = ['Calibration', 'Validation']
        f1 = (average_annual_cal_1, average_annual_val_1, df_annual_average_1)
    else:
        annual_average_cal_1 = pd.DataFrame(df_cal_1.resample('Y').sum().sum())
        annual_average_val_1 = pd.DataFrame(df_val_1.resample('Y').sum().sum())
        f.writelines(
            f'Annual Average:         {annual_average_cal_1.values[0][0]}                      {annual_average_val_1.values[0][0]}                 without grazing\n')
        f.writelines(
            f'Annual Average:         {annual_average_cal_1.values[1][0]}                      {annual_average_val_1.values[1][0]}                 with grazing\n')
        df_annual_average_1 = pd.concat([annual_average_cal_1, annual_average_val_1], axis=1)
        df_annual_average_1.columns = ['Calibration', 'Validation']
        average_annual_cal_1 = pd.DataFrame(df_cal_1.resample('M').sum().resample('Y').sum().mean())
        average_annual_val_1 = pd.DataFrame(df_val_1.resample('M').sum().resample('Y').sum().mean())
        f.writelines(
            f'Average Annual:         {average_annual_cal_1.values[0][0]}                     {average_annual_val_1.values[0][0]}            without grazing\n')
        f.writelines(
            f'Average Annual:         {average_annual_cal_1.values[1][0]}                     {average_annual_val_1.values[1][0]}            with grazing\n')
        df_average_annual_1 = pd.concat([average_annual_cal_1, average_annual_val_1], axis=1)
        df_average_annual_1.columns = ['Calibration', 'Validation']

        daily_average_cal_1 = pd.DataFrame(df_cal_1.resample('Y').mean().mean())
        daily_average_val_1 = pd.DataFrame(df_val_1.resample('Y').mean().mean())
        f.writelines(
            f'Daily Average Annual:   {daily_average_cal_1.values[0][0]}                     {daily_average_val_1.values[0][0]}            without grazing\n')
        f.writelines(
            f'Daily Average Annual:   {daily_average_cal_1.values[1][0]}                     {daily_average_val_1.values[1][0]}            with grazing\n')
        df_daily_average_1 = pd.concat([daily_average_cal_1, daily_average_val_1], axis=1)
        df_daily_average_1.columns = ['Calibration', 'Validation']

        del df_1_n, df_1_g
        f1 = ((annual_average_cal_1, annual_average_val_1, df_annual_average_1),
              (average_annual_cal_1, average_annual_val_1, df_average_annual_1),
              (daily_average_cal_1, daily_average_val_1, df_daily_average_1))
    # Farm 8
    f.writelines(f'--------------------------- For {site2} ------------------------------------------------------\n')
    _, _, df_8_n = get_stress(site=site2, is_graze=False,
                              attribute='runoff', location=location,
                              match_scale=match_scale, scale=scale,
                              metric='OF', var=var, management='Without grazing')
    _, _, df_8_g = get_stress(site=site2, is_graze=True,
                              attribute='runoff', location=location,
                              match_scale=match_scale, scale=scale,
                              metric='OF', var=var, management='With grazing')

    df_8 = pd.DataFrame({'WRE1_n': df_8_n[var], 'WRE1_g': df_8_g[var]})
    df_8['Stage'] = df_8_n.Stage
    df_cal_8, df_val_8 = df_8[df_8.Stage == 'Calibration'], df_8[df_8.Stage == 'Validation']

    if match_scale == 'annual':
        average_annual_cal_8 = pd.DataFrame(df_cal_8.mean())
        average_annual_val_8 = pd.DataFrame(df_val_8.mean())
        df_annual_average_8 = pd.concat([average_annual_cal_8, average_annual_val_8], axis=1)
        f.writelines(
            f'Average Annual:         {average_annual_cal_8.values[0][0]}                     {average_annual_val_8.values[0][0]}            without grazing\n')
        f.writelines(
            f'Average Annual:         {average_annual_cal_8.values[1][0]}                     {average_annual_val_8.values[1][0]}            with grazing\n')
        df_annual_average_8.columns = ['Calibration', 'Validation']
        f2 = (average_annual_cal_8, average_annual_val_8, df_annual_average_8)
    else:
        annual_average_cal_8 = pd.DataFrame(df_cal_8.resample('Y').sum().mean())
        annual_average_val_8 = pd.DataFrame(df_val_8.resample('Y').sum().mean())
        df_annual_average_8 = pd.concat([annual_average_cal_8, annual_average_val_8], axis=1)
        f.writelines(
            f'Annual Average:         {annual_average_cal_8.values[0][0]}                      {annual_average_val_8.values[0][0]}                 without grazing\n')
        f.writelines(
            f'Annual Average:         {annual_average_cal_8.values[1][0]}                      {annual_average_val_8.values[1][0]}                 with grazing\n')
        df_annual_average_8.columns = ['Calibration', 'Validation']

        average_annual_cal_8 = pd.DataFrame(df_cal_8.resample('M').sum().resample('Y').mean().mean())
        average_annual_val_8 = pd.DataFrame(df_val_8.resample('M').sum().resample('Y').mean().mean())
        df_average_annual_8 = pd.concat([average_annual_cal_8, average_annual_val_8], axis=1)
        f.writelines(
            f'Average Annual:         {average_annual_cal_8.values[0][0]}                     {average_annual_val_8.values[0][0]}            without grazing\n')
        f.writelines(
            f'Average Annual:         {average_annual_cal_8.values[1][0]}                     {average_annual_val_8.values[1][0]}            with grazing\n')
        df_average_annual_8.columns = ['Calibration', 'Validation']
        daily_average_cal_8 = pd.DataFrame(df_cal_8.resample('Y').mean().mean())
        daily_average_val_8 = pd.DataFrame(df_val_8.resample('Y').mean().mean())
        df_daily_average_8 = pd.concat([daily_average_cal_8, daily_average_val_8], axis=1)
        f.writelines(
            f'Daily Average Annual:   {daily_average_cal_8.values[0][0]}                     {daily_average_val_8.values[0][0]}            without grazing\n')
        f.writelines(
            f'Daily Average Annual:   {daily_average_cal_8.values[1][0]}                     {daily_average_val_8.values[1][0]}            with grazing\n')
        df_daily_average_8.columns = ['Calibration', 'Validation']
        del df_8_n, df_8_g
        f2 = ((annual_average_cal_8, annual_average_val_8, df_annual_average_8),
              (average_annual_cal_8, average_annual_val_8, df_average_annual_8),
              (daily_average_cal_8, daily_average_val_8, df_daily_average_8))
        f.close()

    return f1, f2


def summarize_multi_output4sites(site1, site2, location, scale, match_scale, var, metric, isAppend):
    if isAppend:
        f = open(f'../post_analysis/multiCalibrationSummary_{metric}.txt', 'a')
    else:
        f = open(f'../post_analysis/multiCalibrationSummary_{metric}.txt', 'w')
        f.writelines('---------Calibration and validation result summary of output variable\n')
    f.writelines(f'---------Output variable, {var}: \n')
    f.writelines(f'---------Output based on, {metric}: \n')
    f.writelines(f'--------------------------- For {site1} ------------------------------------------------------\n')

    _, _, df_1_n = get_stress(site=site1, is_graze=False,
                              attribute='runoff', location=location,
                              match_scale=match_scale, scale=scale,
                              metric=metric, var=var, management='Without grazing')
    _, _, df_1_g = get_stress(site=site1, is_graze=True,
                              attribute='runoff', location=location,
                              match_scale=match_scale, scale=scale,
                              metric=metric, var=var, management='With grazing')

    df_1 = pd.DataFrame({'Without grazing': df_1_n[var], 'With grazing': df_1_g[var]})
    df_1['Stage'] = df_1_n.Stage
    df_cal_1, df_val_1 = df_1[df_1.Stage == 'Calibration'], df_1[df_1.Stage == 'Validation']
    f.writelines(f'                     Calibration                     Validation                     Simulation     '
                 f'              Operation \n')
    if match_scale == 'annual':
        average_annual_cal_1 = pd.DataFrame(df_cal_1.mean())
        average_annual_val_1 = pd.DataFrame(df_val_1.mean())
        average_annual_sim_1 = pd.DataFrame(df_1.mean())
        df_annual_average_1 = pd.concat([average_annual_cal_1, average_annual_val_1, average_annual_sim_1], axis=1)
        f.writelines(
            f'Average Annual:         {average_annual_cal_1.values[0][0]}                     '
            f'{average_annual_val_1.values[0][0]}                     {average_annual_sim_1.values[0][0]}            '
            f'without grazing\n')
        f.writelines(
            f'Average Annual:         {average_annual_cal_1.values[1][0]}                     {average_annual_val_1.values[1][0]}                     '
            f'{average_annual_sim_1.values[1][0]}            with grazing\n')
        df_annual_average_1.columns = ['Calibration', 'Validation', 'Simulation']
        f1 = (average_annual_cal_1, average_annual_val_1, average_annual_sim_1, df_annual_average_1)
    else:
        annual_daily_average_cal_1 = pd.DataFrame(df_cal_1.resample('Y').sum().mean())
        annual_daily_average_val_1 = pd.DataFrame(df_val_1.resample('Y').sum().mean())
        annual_daily_average_sim_1 = pd.DataFrame(df_1.resample('Y').sum().mean())
        df_annual_daily_average_1 = pd.concat([annual_daily_average_cal_1, annual_daily_average_val_1,
                                               annual_daily_average_sim_1], axis=1)
        df_annual_daily_average_1.columns = ['Calibration', 'Validation', 'Simulation']
        f.writelines('Annual average of accumulated daily values over a year\n')
        f.writelines(
            f'Annual Average:         {annual_daily_average_cal_1.values[0][0]}                      '
            f'{annual_daily_average_val_1.values[0][0]}                      {annual_daily_average_sim_1.values[0][0]}'
            f'                 without grazing\n')
        f.writelines(
            f'Annual Average:         {annual_daily_average_cal_1.values[1][0]}                      '
            f'{annual_daily_average_val_1.values[1][0]}                      {annual_daily_average_sim_1.values[1][0]} '
            f'                with grazing\n')

        average_annual_monthly_total_cal_1 = pd.DataFrame(df_cal_1.resample('M').sum().resample('Y').sum().mean())
        average_annual_monthly_total_val_1 = pd.DataFrame(df_val_1.resample('M').sum().resample('Y').sum().mean())
        average_annual_monthly_total_sim_1 = pd.DataFrame(df_1.resample('M').sum().resample('Y').sum().mean())
        f.writelines('Annual average of accumulated monthly values accumulated over a year\n')
        f.writelines(
            f'Average Annual Monthly Total:         {average_annual_monthly_total_cal_1.values[0][0]}'
            f'                     {average_annual_monthly_total_val_1.values[0][0]}                     '
            f'{average_annual_monthly_total_sim_1.values[0][0]}'
            f'            without grazing\n')
        f.writelines(
            f'Average  Annual Monthly Total:         {average_annual_monthly_total_cal_1.values[1][0]}'
            f'                     {average_annual_monthly_total_val_1.values[1][0]}'
            f'                     {average_annual_monthly_total_sim_1.values[1][0]}            with grazing\n')
        df_average_annual_monthly_total_1 = pd.concat([average_annual_monthly_total_cal_1,
                                                       average_annual_monthly_total_val_1,
                                                       average_annual_monthly_total_sim_1], axis=1)
        df_average_annual_monthly_total_1.columns = ['Calibration', 'Validation', 'Simulation']

        monthly_average_cal_1 = pd.DataFrame(df_cal_1.resample('M').sum().mean())
        monthly_average_val_1 = pd.DataFrame(df_val_1.resample('M').sum().mean())
        monthly_average_sim_1 = pd.DataFrame(df_1.resample('M').sum().mean())
        f.writelines('Annual average of accumulated daily values over a month\n')
        f.writelines(
            f'Monthly Average:   {monthly_average_cal_1.values[0][0]}                     '
            f'{monthly_average_val_1.values[0][0]}                     {monthly_average_sim_1.values[0][0]}'
            f'            without grazing\n')
        f.writelines(
            f'Monthly Average:   {monthly_average_cal_1.values[1][0]}'
            f'                     {monthly_average_val_1.values[1][0]}                    '
            f'{monthly_average_sim_1.values[1][0]}            with grazing\n')
        df_monthly_average_1 = pd.concat([monthly_average_cal_1, monthly_average_val_1, monthly_average_sim_1], axis=1)
        df_monthly_average_1.columns = ['Calibration', 'Validation', 'Simulation']

        daily_average_cal_1 = pd.DataFrame(df_cal_1.resample('Y').mean().mean())
        daily_average_val_1 = pd.DataFrame(df_val_1.resample('Y').mean().mean())
        daily_average_sim_1 = pd.DataFrame(df_1.resample('Y').mean().mean())
        f.writelines('Annual average of averaged daily values over a year\n')
        f.writelines(
            f'Daily Average:   {daily_average_cal_1.values[0][0]}'
            f'                     {daily_average_val_1.values[0][0]}'
            f'                     {daily_average_sim_1.values[0][0]}            without grazing\n')
        f.writelines(
            f'Daily Average:   {daily_average_cal_1.values[1][0]}'
            f'                     {daily_average_val_1.values[1][0]}'
            f'                     {daily_average_sim_1.values[1][0]}            with grazing\n')
        df_daily_average_1 = pd.concat([daily_average_cal_1, daily_average_val_1, daily_average_sim_1], axis=1)
        df_daily_average_1.columns = ['Calibration', 'Validation', 'Simulation']

        del df_1_n, df_1_g
        f1 = ((annual_daily_average_cal_1, annual_daily_average_val_1, average_annual_monthly_total_val_1,
               df_annual_daily_average_1), (average_annual_monthly_total_cal_1, average_annual_monthly_total_val_1,
                                            average_annual_monthly_total_sim_1, df_average_annual_monthly_total_1),
              (monthly_average_cal_1, monthly_average_val_1, monthly_average_sim_1, df_monthly_average_1),
              (daily_average_cal_1, daily_average_val_1, daily_average_sim_1, df_daily_average_1))
    # Farm 8
    f.writelines(f'--------------------------- For {site2} ------------------------------------------------------\n')
    _, _, df_8_n = get_stress(site=site2, is_graze=False,
                              attribute='runoff', location=location,
                              match_scale=match_scale, scale=scale,
                              metric=metric, var=var, management='Without grazing')
    _, _, df_8_g = get_stress(site=site2, is_graze=True,
                              attribute='runoff', location=location,
                              match_scale=match_scale, scale=scale,
                              metric=metric, var=var, management='With grazing')

    df_8 = pd.DataFrame({'Without grazing': df_8_n[var], 'With grazing': df_8_g[var]})
    df_8['Stage'] = df_8_n.Stage
    df_cal_8, df_val_8 = df_8[df_8.Stage == 'Calibration'], df_8[df_8.Stage == 'Validation']

    if match_scale == 'annual':
        average_annual_cal_8 = pd.DataFrame(df_cal_8.mean())
        average_annual_val_8 = pd.DataFrame(df_val_8.mean())
        average_annual_sim_8 = pd.DataFrame(df_8.mean())
        df_annual_average_8 = pd.concat([average_annual_cal_8, average_annual_val_8, average_annual_sim_8], axis=1)
        f.writelines('Annual average over the period\n')
        f.writelines(
            f'Average Annual:         {average_annual_cal_8.values[0][0]} '
            f'                    {average_annual_val_8.values[0][0]}'
            f'                     {average_annual_sim_8.values[0][0]}            without grazing\n')
        f.writelines(
            f'Average Annual:         {average_annual_cal_8.values[1][0]}'
            f'                     {average_annual_val_8.values[1][0]}'
            f'                     {average_annual_sim_8.values[1][0]}            with grazing\n')
        df_annual_average_8.columns = ['Calibration', 'Validation', 'Simulation']
        f2 = (average_annual_cal_8, average_annual_val_8, average_annual_sim_8, df_annual_average_8)
    else:
        annual_daily_average_cal_8 = pd.DataFrame(df_cal_8.resample('Y').sum().mean())
        annual_daily_average_val_8 = pd.DataFrame(df_val_8.resample('Y').sum().mean())
        annual_daily_average_sim_8 = pd.DataFrame(df_8.resample('Y').sum().mean())
        df_annual_daily_average_8 = pd.concat([annual_daily_average_cal_8, annual_daily_average_val_8,
                                               annual_daily_average_sim_8], axis=1)
        f.writelines('Annual average of accumulated over a year for the period\n')
        f.writelines(
            f'Annual Average:         {annual_daily_average_cal_8.values[0][0]}'
            f'                      {annual_daily_average_val_8.values[0][0]}'
            f'                     {annual_daily_average_sim_8.values[0][0]}                 without grazing\n')
        f.writelines(
            f'Annual Average:         {annual_daily_average_cal_8.values[1][0]}'
            f'                      {annual_daily_average_val_8.values[1][0]}'
            f'                      {annual_daily_average_sim_8.values[1][0]}                 with grazing\n')
        df_annual_daily_average_8.columns = ['Calibration', 'Validation', 'Simulation']

        average_annual_monthly_total_cal_8 = pd.DataFrame(df_cal_8.resample('M').sum().resample('Y').mean().mean())
        average_annual_monthly_total_val_8 = pd.DataFrame(df_val_8.resample('M').sum().resample('Y').mean().mean())
        average_annual_monthly_total_sim_8 = pd.DataFrame(df_8.resample('M').sum().resample('Y').mean().mean())
        df_average_annual_monthly_total_8 = pd.concat([average_annual_monthly_total_cal_8,
                                                       average_annual_monthly_total_val_8,
                                                       average_annual_monthly_total_sim_8], axis=1)
        f.writelines('Annual average of monthly average daily accumulated over a month  for the period\n')
        f.writelines(
            f'Average Annual:         {average_annual_monthly_total_cal_8.values[0][0]}'
            f'                     {average_annual_monthly_total_val_8.values[0][0]}'
            f'                     {average_annual_monthly_total_sim_8.values[0][0]}            without grazing\n')
        f.writelines(
            f'Average Annual:         {average_annual_monthly_total_cal_8.values[1][0]}'
            f'                     {average_annual_monthly_total_val_8.values[1][0]}'
            f'                     {average_annual_monthly_total_sim_8.values[1][0]}            with grazing\n')
        df_average_annual_monthly_total_8.columns = ['Calibration', 'Validation', 'Simulation']

        monthly_average_cal_8 = pd.DataFrame(df_cal_8.resample('M').sum().mean())
        monthly_average_val_8 = pd.DataFrame(df_val_8.resample('M').sum().mean())
        monthly_average_sim_8 = pd.DataFrame(df_8.resample('M').sum().mean())
        df_monthly_average_8 = pd.concat([monthly_average_cal_8, monthly_average_val_8, monthly_average_sim_8], axis=1)
        f.writelines('Monthly average of daily accumulated over a month  for the period\n')
        f.writelines(
            f'Average Annual:         {monthly_average_cal_8.values[0][0]}'
            f'                     {monthly_average_val_8.values[0][0]}'
            f'                     {monthly_average_sim_8.values[0][0]}            without grazing\n')
        f.writelines(
            f'Average Annual:         {monthly_average_cal_8.values[1][0]}'
            f'                     {monthly_average_val_8.values[1][0]}'
            f'                     {monthly_average_sim_8.values[1][0]}            with grazing\n')
        df_monthly_average_8.columns = ['Calibration', 'Validation', 'Simulation']

        daily_average_cal_8 = pd.DataFrame(df_cal_8.resample('Y').mean().mean())
        daily_average_val_8 = pd.DataFrame(df_val_8.resample('Y').mean().mean())
        daily_average_sim_8 = pd.DataFrame(df_8.resample('Y').mean().mean())
        df_daily_average_8 = pd.concat([daily_average_cal_8, daily_average_val_8, daily_average_sim_8], axis=1)
        f.writelines('Annual average of daily accumulated over a year  for the period\n')
        f.writelines(
            f'Daily Average Annual:   {daily_average_cal_8.values[0][0]}'
            f'                     {daily_average_val_8.values[0][0]}'
            f'                     {daily_average_sim_8.values[0][0]}            without grazing\n')
        f.writelines(
            f'Daily Average Annual:   {daily_average_cal_8.values[1][0]}'
            f'                     {daily_average_val_8.values[1][0]}'
            f'                     {daily_average_sim_8.values[1][0]}            with grazing\n')
        df_daily_average_8.columns = ['Calibration', 'Validation', 'Simulation']
        del df_8_n, df_8_g
        f2 = ((annual_daily_average_cal_8, annual_daily_average_val_8, annual_daily_average_sim_8,
               df_annual_daily_average_8), (average_annual_monthly_total_cal_8, average_annual_monthly_total_val_8,
                                            average_annual_monthly_total_sim_8, df_average_annual_monthly_total_8),
              (monthly_average_cal_8, monthly_average_val_8, monthly_average_sim_8, df_monthly_average_8),
              (daily_average_cal_8, daily_average_val_8, daily_average_sim_8, df_daily_average_8))
        f.close()

    return f1, f2


def get_obs_data_stats(site, attribute, factor=1):
    df_obs = pd.read_csv(f'../{site}/pyAPEX_n/pyAPEX/Program/Calibration_data.csv')
    for i in range(df_obs.shape[0]):
        df_obs.Date[i] = str(df_obs.Year[i]) + '/' + str(df_obs.Month[i]) + '/' + str(df_obs.Day[i])
    df_obs.index = df_obs.Date
    df_obs.index = pd.to_datetime(df_obs.index)
    df_obs = df_obs[['sediment (kg)', 'runoff (mm)']]
    df_obs.columns = ['sediment', 'runoff']
    df_cal, df_val, _ = get_model_set(site=site,
                                      is_graze=False,
                                      attribute='runoff',
                                      location='model_calibration',
                                      scale='daily',
                                      metric='OF')
    data = df_obs[attribute] * factor
    if type(data) is pd.DataFrame:
        df_sim = data.copy()
    else:
        df_sim = pd.DataFrame(data)
    df_sim.index = df_obs.index
    start_cal, end_cal = df_cal.index[0], df_cal.index[-1]
    start_val, end_val = df_val.index[0], df_val.index[-1]
    df_sim.Stage[(df_sim.index >= start_cal) & (df_sim.index <= end_cal)] = 'Calibration'
    df_sim.Stage[(df_sim.index >= start_val) & (df_sim.index <= end_val)] = 'Validation'
    df_cal = df_sim[df_sim.Stage == 'Calibration']
    df_val = df_sim[df_sim.Stage == 'Validation']
    annual_average_cal = pd.DataFrame(df_cal.resample('Y').sum().mean())
    annual_average_val = pd.DataFrame(df_val.resample('Y').sum().mean())
    df_annual_average = pd.concat([annual_average_cal, annual_average_val], axis=1)
    df_annual_average.columns = ['Calibration', 'Validation']
    average_annual_cal = pd.DataFrame(df_cal.resample('M').sum().resample('Y').sum().mean())
    average_annual_val = pd.DataFrame(df_val.resample('M').sum().resample('Y').sum().mean())
    df_average_annual = pd.concat([average_annual_cal, average_annual_val], axis=1)
    df_average_annual.columns = ['Calibration', 'Validation']

    daily_average_cal = pd.DataFrame(df_cal.resample('Y').mean().mean())
    daily_average_val = pd.DataFrame(df_val.resample('Y').mean().mean())
    df_daily_average = pd.concat([daily_average_cal, daily_average_val], axis=1)
    df_daily_average.columns = ['Calibration', 'Validation']

    f1 = ((annual_average_cal, annual_average_val, df_annual_average),
          (average_annual_cal, average_annual_val, df_average_annual),
          (daily_average_cal, daily_average_val, df_daily_average))
    return f1


def get_obs_data(site, attribute, factor=1):
    df_obs = pd.read_csv(f'../{site}/pyAPEX_n/pyAPEX/Program/Calibration_data.csv')
    for i in range(df_obs.shape[0]):
        df_obs.Date[i] = str(df_obs.Year[i]) + '/' + str(df_obs.Month[i]) + '/' + str(df_obs.Day[i])
    df_obs.index = df_obs.Date
    df_obs.index = pd.to_datetime(df_obs.index)
    df_obs = df_obs[['sediment (kg)', 'runoff (mm)']]
    df_obs.columns = ['sediment', 'runoff']

    data = df_obs[attribute] * factor
    data_Y = data.resample('Y').sum()
    data_Y = pd.DataFrame(data_Y)
    data_Y.insert(1, 'Operation', 'Observed')
    return data_Y


def read_obs_data(site, attribute):
    df_obs = pd.read_csv(f'../{site}/pyAPEX_n/pyAPEX/Program/Calibration_data.csv')
    for i in range(df_obs.shape[0]):
        df_obs.Date[i] = str(df_obs.Year[i]) + '/' + str(df_obs.Month[i]) + '/' + str(df_obs.Day[i])
    df_obs.index = df_obs.Date
    df_obs.index = pd.to_datetime(df_obs.index)
    df_obs = df_obs[['sediment (kg)', 'runoff (mm)']]
    df_obs.columns = ['sediment', 'runoff']
    df_obs = pd.DataFrame(df_obs[attribute])
    return df_obs


def get_plotdata(site, file, attribute='sediment', factor=1, scale='daily'):
    ob_data = read_obs_data(site, attribute='sediment', factor=1)
    ob_data.columns = ['Observed']
    data_model = pd.read_csv(file)
    data_model = data_model[['Date', 'YSD']]
    data_model.index = data_model.Date
    data_model = data_model.drop('Date', axis=1)
    data_model.index = pd.to_datetime(data_model.index)
    data_model.columns = ['Modeled']
    data_plot = pd.concat([ob_data, data_model], axis=1)
    data_plot.Observed = data_plot.Observed * factor
    data_plot = data_plot.dropna()
    start_year = data_plot.index.year[0]
    n_warm, cal_year = 4, 11
    cal_start_year = start_year + n_warm
    end_cal_year = cal_start_year + cal_year
    cal_data = data_plot[(data_plot.index.year >= cal_start_year) & (data_plot.index.year <= end_cal_year)]
    val_data = data_plot[data_plot.index.year > end_cal_year]
    if scale == 'monthly':
        cal_data = cal_data.resample('M').sum()
        val_data = val_data.resample('M').sum()
    elif scale == 'yearly':
        cal_data = cal_data.resample('Y').sum()
        val_data = val_data.resample('Y').sum()
    return cal_data, val_data


def get_data_melt(site, scenario, attribute, site_label, scenario_label):
    file = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/daily_{attribute}.csv'
    data = pd.read_csv(file, index_col=0)
    data.index = pd.to_datetime(data.index)
    data_annual = data.resample('Y').sum()
    data_annual.insert(0, 'Year', data_annual.index.year)
    data_melt = pd.melt(data_annual, id_vars=['Year'])
    data_melt['Watershed'] = site_label
    data_melt['Operation'] = scenario_label
    mean_best = get_data_best(site, scenario, attribute, data_annual)
    return data_melt, mean_best


def get_data_best(site, scenario, attribute, df_ref):
    year_start = df_ref.Year[0]
    year_end = df_ref.Year[df_ref.shape[0] - 1]
    file = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/runoff/daily_basin_best.csv'
    data = pd.read_csv(file, index_col=0)
    data.index = pd.to_datetime(data.index)
    data = pd.DataFrame(data[attribute])
    data_annual = data.resample('Y').sum()
    data_annual.index = data_annual.index.year
    data_annual = data_annual[(data_annual.index >= year_start) & (data_annual.index <= year_end)]
    mean_best = data_annual.mean()[0]
    return mean_best


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
		ref: https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
	"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class SeabornFig2Grid:

    def __init__(self, seaborngrid, fig, subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
                isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n, m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i, j], self.subgrid[i, j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h = self.sg.ax_joint.get_position().height
        h2 = self.sg.ax_marg_x.get_position().height
        r = int(np.round(h / h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r + 1, r + 1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        # https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure = self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def get_plot_heatmap(filename, ax, custom_delta=-0.5, col_name='Percent_change', cmap='YlGnBu', x_label=None,
                     y_label=None, title_str='WRE1 without grazing',
                     c_bar_title='Percent change in objective function value', n_y_ticks=20):
    data = pd.read_csv(filename)
    # flip the dara from max to min
    data.sort_index(ascending=True)
    data.rename(columns={'Unnamed: 0': col_name}, inplace=True)
    data.index = data.Percent_change
    data = data.drop(col_name, axis=1)
    y_ticks = np.arange(data.index.max(), data.index.min() + custom_delta, custom_delta)
    sns.heatmap(data, cmap=cmap, vmin=-data.min().min(), vmax=data.max().max(), cbar_kws={'label': c_bar_title},
                yticklabels=y_ticks, ax=ax)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title_str, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(n_y_ticks))
    cbar = ax.collections[0].colorbar
    # here set the label size by 12
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.label.set_size(12)
    return ax


def get_params(file_param, id_sensitive, is_all):
    df_param = pd.DataFrame()
    df_params = pd.read_csv(file_param)
    df_params.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
    param_list = df_params.columns
    params_all = df_params
    if is_all:
        params = df_params.copy()
    else:
        param_list = param_list[id_sensitive]
        df_param = df_params[param_list]
        df_param.insert(0, 'RunId', df_params.RunId)
        params = df_param.iloc[1:, ]
    p_best = df_param.iloc[0, :]
    return params, p_best, params_all


def read_stats(file_name):
    df_stats = pd.read_csv(file_name)
    df_stats.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
    df_stats_best = df_stats.iloc[0, :]
    df_stats = df_stats.iloc[1:, :]
    list_stats = ['RunId', 'CODDC', 'RMSEDC', 'NRMSEDC', 'NSEDC', 'PBIASDC', 'OF2DC', 'CODDV', 'RMSEDV',
                  'NRMSEDV', 'NSEDV', 'PBIASDV', 'OF2DV']
    # list_stats = ['RunId', 'CODAD', 'RMSEAD', 'NRMSEAD', 'NSEAD', 'PBIASAD', 'OF2AD', 'CODAM', 'RMSEAM', 'NRMSEAM',
    # 'NSEAM', 'PBIASAM',  'OF2AM',  'CODAY', 'RMSEAY', 'NRMSEAY', 'NSEAY', 'PBIASAY', 'OF2AY']
    df_metrics = df_stats[list_stats]
    df_best_metric = df_stats_best[list_stats]
    return df_metrics, df_best_metric


def discretize_ids(config, n_param):
    max_range = int(config['max_range'])
    increment = float(config['increment'])
    disc_range = np.arange(-max_range, max_range + 1, increment)
    id_sets = np.arange(0, len(disc_range) * (n_param + 1) + len(disc_range), len(disc_range))
    start_sets = id_sets[:-1] + 1
    end_sets = id_sets[1:]
    return disc_range, start_sets, end_sets


def discretize_std(config):
    max_range = int(config['max_range_uncertaintity'])
    increment = float(config['increment_uncertainty'])
    disc_range = np.arange(-max_range, max_range + increment, increment)
    return disc_range


def plot_line_percent(file_name, ax, col_name, x_label=None, y_label=None, title_str=None):
    data = pd.read_csv(file_name)
    data.rename(columns={'Unnamed: 0': col_name}, inplace=True)
    data_melt = data.melt(id_vars=col_name)
    data_melt.columns = ['Percent_change', 'Parameters', 'value']
    sns.lineplot(data=data_melt, x=col_name, y='value', hue='Parameters', ax=ax)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(20))
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title_str, fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(title=None)
    ax.grid(True)
    ax.set_xlim(-5, 5)


def read_sensitive_params(file):
    with open(file) as f:
        line = f.read()
    f.close()
    l = line.split(',')
    id_sensitive = [int(item) for item in l]
    for i in range(len(id_sensitive)):
        id_sensitive[i] = id_sensitive[i] + 70  # change from 69 to 70 because of nature of the importing dataframe
    return id_sensitive


def normalize_dataframe(df, params):
    df_n = df.copy()
    for col in params:
        df_n[col] = (df[col].values - np.mean(df[col].values)) / np.std(df[col].values)
    return df_n


def calculate_euclidean_distance(df):
    n_row, n_col = df.shape
    norms = []
    for i in range(n_row):
        vec = df.iloc[i, :].values
        distance = []
        for j in range(n_col):
            for k in range(n_col):
                distance.append((vec[j] - vec[k]) ** 2)
        ec_distance = np.sqrt(np.sum(distance))
        del distance
        norms.append(ec_distance)
        del ec_distance
    return norms


def get_measure(file_name):
    df_data = pd.read_csv(file_name)
    date_vec = []
    n_data = df_data.shape[0]
    for i in range(n_data):
        date_vec.append(pd.to_datetime(f'{df_data.Year[i]}-{df_data.Month[i]}-{df_data.Day[i]}'))
    df_data.Date = date_vec
    df_data.index = df_data.Date
    df_data = df_data[['Date', 'Year', 'Month', 'Day', 'runoff (mm)', 'sediment (kg)']]
    df_data.columns = ['Date', 'Year', 'Month', 'Day', 'runoff', 'sediment']
    return df_data


def summarize_uncertainty_data(site, scenario, obs_attribute):
    main_dir = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/{obs_attribute}/'
    list_variable = ['PRCP', 'WYLD', 'YSD', 'RUS2', 'ET', 'PET', 'DPRK', 'BIOM', 'STD', 'STDL', 'STL', 'YLDF',
                     'YLDG', 'TN', 'TP', 'LAI', 'Q', 'WS', 'TS', 'NS', 'PS']
    list_scale = ['Daily', 'Daily', 'Daily', 'Daily', 'Daily', 'Daily', 'Daily', 'Daily', 'Daily', 'Daily', 'Daily',
                  'Annual', 'Annual', 'Daily', 'Daily', 'Daily', 'Daily', 'Annual', 'Annual', 'Annual', 'Annual']
    df_annual = pd.DataFrame()
    for i in range(len(list_scale)):
        variable = list_variable[i]
        scale_str = list_scale[i]
        print(f'Summarizing {variable} from {scale_str} scale')
        if scale_str == 'Daily':
            data_annual = convert_daily2annual(main_dir, variable, scale_str='Daily')
        else:
            data_annual = read_annual_result(main_dir, variable, scale_str='Annual')
        annual_average = data_annual.mean(axis=0)
        _annual = pd.DataFrame(annual_average).T
        _annual.index = [variable]
        df_annual = pd.concat([df_annual, _annual], axis=0)

    sigma_vec = np.arange(-3, 3 + 0.001, 0.001)
    id_start = np.where(sigma_vec <= -1)[0][-1]
    id_end = np.where(sigma_vec >= 1)[0][0]
    sigma_vec1 = sigma_vec[id_start:id_end]
    # Select main features
    list_ids = [0, 1000, 2000, 3000, 4000, 5000, 6000]
    list_names = ['Minimum', 'Mean-3(Standard deviation)', 'Mean-2(Standard deviation)', 'Mean-Standard deviation',
                  'Mean', 'Mean+(Standard deviation)', 'Mean+2(Standard deviation)', 'Mean+3(Standard deviation)',
                  'Maximum']
    df_select = pd.DataFrame()
    for i in range(len(list_ids)):
        dfi = pd.DataFrame(df_annual.iloc[:, list_ids[i]]).T
        df_select = pd.concat([df_select, dfi], axis=0)
    df_select = pd.concat([pd.DataFrame(df_annual.min(axis=1)).T, df_select], axis=0)
    df_select = pd.concat([df_select, pd.DataFrame(df_annual.max(axis=1)).T], axis=0)
    df_select.insert(df_select.shape[1], 'Mode', list_names)
    df_select = df_select.reset_index()
    df_select = df_select.drop('index', axis=1)
    # select within Mean - (Standard deviation) and Mean + (Standard deviation)
    df_data = df_annual.iloc[:, id_start:id_end]
    plot_data = df_data.T
    plot_data.insert(plot_data.shape[1], 'Sigma', sigma_vec1)
    plot_data1 = plot_data.copy()
    plot_data1 = plot_data1.dropna()
    return plot_data1, plot_data, df_select


def summarize_uncertainty_attributes(site, scenario, obs_attribute):
    main_dir = f'../post_analysis/Uncertainty_analysis/{site}/{scenario}/{obs_attribute}/'
    list_variable = ['PRCP', 'WYLD', 'YSD', 'RUS2', 'ET', 'PET', 'DPRK', 'BIOM', 'STD', 'STDL', 'STL', 'YLDF',
                     'YLDG', 'TN', 'TP', 'LAI', 'WS', 'TS', 'NS', 'PS']
    df_annual = pd.DataFrame()
    for i in range(len(list_variable)):
        variable = list_variable[i]
        print(f'Summarizing {variable}')
        data_annual = read_annual_result(main_dir, variable, scale_str='Annual')
        annual_average = data_annual.mean(axis=0)
        _annual = pd.DataFrame(annual_average).T
        _annual.index = [variable]
        df_annual = pd.concat([df_annual, _annual], axis=0)
    df_select = df_annual.iloc[:, 1999:3000]
    df_mean = df_select.mean(axis=1)
    df_std = df_select.std(axis=1)
    df_summary = pd.DataFrame({'Mean': df_mean.values, 'Standard deviation': df_std.values}, index=list_variable)
    df_summary.to_csv(f'../post_analysis/Results/Uncertainty_response_variable_range_{site}_{scenario}.csv')

    return df_summary


def get_calibrated_result(site, scenario, obs_attribute, metric, n_warm=4):
    if scenario == 'grazing':
        sub_location = 'pyAPEX_g'
    else:
        sub_location = 'pyAPEX_n'
    file_daily_basin = f'../{site}/{sub_location}/pyAPEX/Output/{obs_attribute}/daily_basin_daily_{metric}.csv'
    file_daily_outlet = f'../{site}/{sub_location}/pyAPEX/Output/{obs_attribute}/daily_outlet_daily_{metric}.csv'
    file_annual_outlet = f'../{site}/{sub_location}/pyAPEX/Output/{obs_attribute}/annual_daily_{metric}.csv'
    file_measure = f'../{site}/{sub_location}/pyAPEX/Program/calibration_data.csv'
    # importing measured data to match the end date of the observation
    data_obs = get_measure(file_measure)
    data_obs.index = data_obs.Date
    data_obs = data_obs.drop('Date', axis=1)
    end_date = data_obs.index[-1]
    end_year = data_obs.index.year[-1]
    # reading basin results
    df_daily_basin = pd.read_csv(file_daily_basin, index_col=0)
    df_daily_basin.index = pd.to_datetime(df_daily_basin.index)
    start_year = df_daily_basin.index.year[0] + n_warm
    start_date = pd.to_datetime(f'{start_year}/01/01')
    df_daily_basin = df_daily_basin.loc[(df_daily_basin.index >= start_date) & (df_daily_basin.index <= end_date), :]
    df_daily_basin1 = df_daily_basin[['PRCP', 'ET', 'PET', 'DPRK', 'BIOM', 'STD', 'STDL', 'STL',
                                      'TN', 'TP', 'LAI']]
    crops = df_daily_basin.CPNM.unique()
    if len(crops) > 1:
        df_basin = pd.DataFrame(columns=['PRCP', 'ET', 'PET', 'DPRK', 'BIOM', 'STD', 'STDL', 'STL', 'TN', 'TP', 'LAI',
                                         ])
        for col in df_daily_basin1.columns:
            df_crops = pd.DataFrame()
            for crop in crops:
                df_crop = df_daily_basin[df_daily_basin.CPNM == crop]
                df_crop = pd.DataFrame(df_crop[col])
                df_crops = pd.concat([df_crops, df_crop], axis=1)
            df_basin[col] = df_crops.sum(axis=1)
        del df_crop, df_crops, df_daily_basin
        data_daily_basin = df_basin
        del df_basin, df_daily_basin1
    else:
        data_daily_basin = df_daily_basin1
        del df_daily_basin1
    # reading basin outlet results
    df_daily_outlet = pd.read_csv(file_daily_outlet, index_col=0)
    df_daily_outlet.index = pd.to_datetime(df_daily_outlet.index)
    df_daily_outlet = df_daily_outlet.loc[(df_daily_outlet.index >= start_date) & (df_daily_outlet.index <= end_date),
                      :]
    data_daily_outlet = df_daily_outlet[['WYLD', 'YSD', 'RUS2', 'Q']]
    # reading annual results
    data_annual = pd.read_csv(file_annual_outlet, index_col=0)
    data_annual = data_annual.loc[(data_annual.index >= start_year) & (data_annual.index <= end_year), :]
    data_annual1 = data_annual[['YLDG', 'YLDF', 'WS', 'NS', 'PS', 'TS']]
    if len(crops) > 1:
        df_annual = pd.DataFrame(columns=['YLDG', 'YLDF', 'WS', 'NS', 'PS', 'TS'])
        for col in data_annual1.columns:
            df_crops = pd.DataFrame()
            for crop in crops:
                df_crop = data_annual[data_annual.CPNM == crop]
                df_crop = pd.DataFrame(df_crop[col])
                df_crops = pd.concat([df_crops, df_crop], axis=1)
            df_annual[col] = df_crops.sum(axis=1)
        del df_crop, df_crops, data_annual
        data_annual = df_annual
        del df_annual, data_annual1
    else:
        data_annual = data_annual1
        del data_annual1
    del start_date, start_year, end_date, end_year

    # aggregating annual data set
    data_annual_basin = data_daily_basin.resample('Y').sum()
    data_annual_basin.index = data_annual.index
    data_annual_outlet = data_daily_outlet.resample('Y').sum()
    data_annual_outlet.index = data_annual.index
    data_annual_calibrated = pd.concat([data_annual_outlet, data_annual_basin, data_annual], axis=1)
    data_annual_calibrated_sum = pd.DataFrame(data_annual_calibrated.mean(axis=0)).T
    # data_annual_calibrated_sum = data_annual_calibrated_sum.drop('Q', axis=1)
    return data_annual_calibrated_sum, data_annual_calibrated, (data_annual_outlet, data_annual_basin, data_annual)


def convert_daily2annual(read_dir, variable, scale_str='Daily'):
    file_name = f'{scale_str}_{variable}.csv'
    data_file = read_dir + file_name
    save_file = read_dir + f'Annual_{variable}.csv'
    df_daily = pd.read_csv(data_file, index_col=0)
    cols = df_daily.columns
    if 'Stage' in cols:
        df_daily = df_daily.drop('Stage', axis=1)
    df_daily.index = pd.to_datetime(df_daily.index)
    df_annual = df_daily.resample('Y').sum()
    df_annual.index = df_annual.index.year
    df_annual.to_csv(save_file)
    return df_annual


def read_annual_result(read_dir, variable, scale_str='Annual'):
    file_name = f'{scale_str}_{variable}.csv'
    data_file = read_dir + file_name
    df_annual = pd.read_csv(data_file, index_col=0)
    cols = df_annual.columns
    if 'Stage' in cols:
        df_annual = df_annual.drop('Stage', axis=1)
    return df_annual


def find_id(vec, val):
    id_vec = np.arange(len(vec))
    for j in range(len(id_vec)):
        if float(vec[j]) == float(val):
            id = j
    return id


def get_range(site, scenario, task):
    if scenario == 'grazing':
        confile = f'../{site}/pyAPEX_g/pyAPEX/runtime.ini'
    else:
        confile = f'../{site}/pyAPEX_n/pyAPEX/runtime.ini'
    config = ConfigObj(confile)
    if task == 'Uncertainty':
        max_range = float(config['max_range_uncertaintity'])
        inc = float(config['increment_uncertainty'])
    elif task == 'Sensitivity':
        max_range = float(config['max_range'])
        inc = float(config['increment'])
    vec = np.arange(-max_range, max_range + inc, inc)
    vecs = []
    for num in vec:
        format_num = '{:0.3f}'.format(num)
        # vecs.append(float(format_num))
        vecs.append(format_num)
    return vecs


def get_best(site, scenario, attribute, scale, metric):
    if scenario == 'non_grazing':
        sub_folder = 'pyAPEX_n'
    else:
        sub_folder = 'pyAPEX_g'
    file = f'../{site}/{sub_folder}/pyAPEX/Output/{attribute}/summary_stats.csv'
    data_stat = pd.read_csv(file, index_col=0)
    data_stat = data_stat[data_stat.SCALE == scale]
    data_stat = data_stat.loc[metric, :]
    best_id = data_stat.RunId
    return best_id


def dayOfYear(date):
    days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d = list(map(int, date.split("-")))
    if d[0] % 400 == 0:
        days[2] += 1
    elif d[0] % 4 == 0 and d[0] % 100 != 0:
        days[2] += 1
    for i in range(1, len(days)):
        days[i] += days[i - 1]
    return int(days[d[1] - 1] + d[2])


# Source https://www.tutorialspoint.com/day-of-the-year-in-python

def read_outlet_result(site, scenario, attribute, scale, metric, id_best, task):
    if scenario == 'non_grazing':
        sub_folder = 'pyAPEX_n'
    else:
        sub_folder = 'pyAPEX_g'
    if task == 'Calibration':
        data_file = f'../{site}/{sub_folder}/pyAPEX/Output/{attribute}/daily_outlet_{id_best:07}.csv'
    elif task == 'Sensitivity':
        data_file = f'../{site}/{sub_folder}/pyAPEX/OutputSensitivity/daily_outlet_{id_best:07}.csv'
    elif task == 'Uncertainty':
        data_file = f'../{site}/{sub_folder}/pyAPEX/OutputUncertainty/daily_outlet_{id_best:07}.csv'

    data_daily = pd.read_csv(data_file)
    data_daily.Date = pd.to_datetime(data_daily.Date)
    model_result = read_model_output(site, scenario, attribute, metric)
    start_cal_date, end_cal_date = get_end_dates(model_result[0], date_attr='Date', scale=scale, stage='Calibration')
    start_val_date, end_val_date = get_end_dates(model_result[0], date_attr='Date', scale=scale, stage='Validation')
    data_run = data_daily[(data_daily.Date >= start_cal_date) & (data_daily.Date <= end_val_date)]
    data_run.index = data_run.Date
    data_run = data_run.drop('Date', axis=1)
    data_cal = data_daily[(data_daily.Date >= start_cal_date) & (data_daily.Date <= end_cal_date)]
    data_cal.index = data_cal.Date
    data_cal = data_cal.drop('Date', axis=1)
    data_val = data_daily[(data_daily.Date >= start_val_date) & (data_daily.Date <= end_val_date)]
    data_val.index = data_val.Date
    data_val = data_val.drop('Date', axis=1)
    data_daily.index = data_daily.Date
    data_daily = data_daily.drop('Date', axis=1)
    return data_run, data_cal, data_val, data_daily


def read_basin_result(site, scenario, attribute, scale, metric, id_best, task):
    if scenario == 'non_grazing':
        sub_folder = 'pyAPEX_n'
    else:
        sub_folder = 'pyAPEX_g'
    if task == 'Calibration':
        data_file = f'../{site}/{sub_folder}/pyAPEX/Output/{attribute}/daily_basin_{id_best:07}.csv'
    elif task == 'Sensitivity':
        data_file = f'../{site}/{sub_folder}/pyAPEX/OutputSensitivity/daily_basin_{id_best:07}.csv'
    elif task == 'Uncertainty':
        data_file = f'../{site}/{sub_folder}/pyAPEX/OutputUncertainty/daily_basin_{id_best:07}.csv'
    data_daily = pd.read_csv(data_file)
    data_daily.Date = pd.to_datetime(data_daily.Date)
    data_daily.index = data_daily.Date
    crops = data_daily.CPNM.unique()
    column_list = data_daily.columns
    df_basin1 = data_daily[column_list[:4]]
    if len(crops) > 1:
        df_basin = pd.DataFrame(columns=column_list[4:])
        for col in column_list[4:]:
            df_crops = pd.DataFrame()
            for crop in crops:
                df_crop = data_daily[data_daily.CPNM == crop]
                df_crop = pd.DataFrame(df_crop[col])
                df_crops = pd.concat([df_crops, df_crop], axis=1)
            df_crops.columns = crops
            if (col == 'TMX') | (col == 'TMN'):
                df_basin[col] = df_crops.mean(axis=1)
            else:
                df_basin[col] = df_crops.sum(axis=1)
        del df_crop, df_crops
        data_daily = pd.concat([df_basin1, df_basin], axis=1)
        del df_basin, df_basin1
    else:
        df_basin = data_daily[column_list[4:]]
        data_daily = pd.concat([df_basin1, df_basin], axis=1)
    model_result = read_model_output(site, scenario, attribute, metric)
    start_cal_date, end_cal_date = get_end_dates(model_result[0], date_attr='Date', scale=scale, stage='Calibration')
    start_val_date, end_val_date = get_end_dates(model_result[0], date_attr='Date', scale=scale, stage='Validation')
    data_run = data_daily[(data_daily.Date >= start_cal_date) & (data_daily.Date <= end_val_date)]
    data_run.index = data_run.Date
    data_run = data_run.drop('Date', axis=1)
    data_cal = data_daily[(data_daily.Date >= start_cal_date) & (data_daily.Date <= end_cal_date)]
    data_cal.index = data_cal.Date
    data_cal = data_cal.drop('Date', axis=1)
    data_val = data_daily[(data_daily.Date >= start_val_date) & (data_daily.Date <= end_val_date)]
    data_val.index = data_val.Date
    data_val = data_val.drop('Date', axis=1)
    data_daily.index = data_daily.Date
    data_daily = data_daily.drop('Date', axis=1)
    return data_run, data_cal, data_val, data_daily


def read_result_annual(site, scenario, attribute, scale, metric, id_best, task):
    if scenario == 'non_grazing':
        sub_folder = 'pyAPEX_n'
    else:
        sub_folder = 'pyAPEX_g'
    if task == 'Calibration':
        data_file = f'../{site}/{sub_folder}/pyAPEX/Output/{attribute}/annual_{id_best:07}.csv'
    elif task == 'Sensitivity':
        data_file = f'../{site}/{sub_folder}/pyAPEX/OutputSensitivity/annual_{id_best:07}.csv'
    elif task == 'Uncertainty':
        data_file = f'../{site}/{sub_folder}/pyAPEX/OutputUncertainty/annual_{id_best:07}.csv'
    data_annual = pd.read_csv(data_file)
    data_annual.index = data_annual.YR
    crops = data_annual.CPNM.unique()
    column_list = data_annual.columns
    if len(crops) > 1:
        df_annual = pd.DataFrame(columns=column_list[1:])
        for col in column_list[1:]:
            df_crops = pd.DataFrame()
            for crop in crops:
                df_crop = data_annual[data_annual.CPNM == crop]
                df_crop = pd.DataFrame(df_crop[col])
                df_crops = pd.concat([df_crops, df_crop], axis=1)
            df_crops.columns = crops
            if (col == 'WS') | (col == 'TS') | (col == 'NS') | (col == 'PS') | (col == 'AS') | (col == 'SS'):
                df_annual[col] = df_crops.mean(axis=1)
            else:
                df_annual[col] = df_crops.sum(axis=1)
        del df_crop, df_crops
        data_annual = df_annual
        del df_annual
    else:
        data_annual = data_annual[column_list[1:]]
    data_annual = data_annual.drop('BIOM', axis=1)
    model_result = read_model_output(site, scenario, attribute, metric)
    start_cal_year, end_cal_year = get_end_dates(model_result[1], date_attr='YR', scale=scale, stage='Calibration')
    start_val_year, end_val_year = get_end_dates(model_result[1], date_attr='YR', scale=scale, stage='Validation')
    data_run = data_annual[(data_annual.YR >= start_cal_year) & (data_annual.YR <= end_val_year)]
    data_run.index = data_run.YR
    data_run = data_run.drop('YR', axis=1)
    data_cal = data_annual[(data_annual.YR >= start_cal_year) & (data_annual.YR <= end_cal_year)]
    data_cal.index = data_cal.YR
    data_cal = data_cal.drop('YR', axis=1)
    data_val = data_annual[(data_annual.YR >= start_val_year) & (data_annual.YR <= end_val_year)]
    data_val.index = data_val.YR
    data_val = data_val.drop('YR', axis=1)
    return data_run, data_cal, data_val, data_annual


def read_model_output(site, scenario, attribute, metric):
    if scenario == 'non_grazing':
        sub_folder = 'pyAPEX_n'
    else:
        sub_folder = 'pyAPEX_g'
    basin_file = f'../{site}/{sub_folder}/pyAPEX/Output/{attribute}/model_result_basin_{metric}.csv'
    annual_file = f'../{site}/{sub_folder}/pyAPEX/Output/{attribute}/model_result_annual_{metric}.csv'
    data_basin = pd.read_csv(basin_file)
    data_annual = pd.read_csv(annual_file)
    data_basin.Date = pd.to_datetime(data_basin.Date)
    return data_basin, data_annual


def get_end_dates(df, date_attr, scale, stage):
    df_simulation = df[df.STAGE == stage]
    df_simulation = df_simulation[df_simulation.SCALE == scale]
    sim_date_vec = df_simulation[date_attr].values
    return sim_date_vec[0], sim_date_vec[-1]


def get_model_result(site, scenario, attribute, scale, metric, task):
    if metric == 'OF':
        id_best = get_best(site, scenario, attribute, scale, metric='Objective Function')
    else:
        id_best = get_best(site, scenario, attribute, scale, metric)
    data_run, data_cal, data_val, data_annual = read_result_annual(site, scenario, attribute, scale, metric, id_best,
                                                                   task)
    data_run_outlet, data_cal_outlet, data_val_outlet, data_daily_outlet = read_outlet_result(site, scenario, attribute,
                                                                                              scale, metric, id_best,
                                                                                              task)
    data_run_basin, data_cal_basin, data_val_basin, data_daily_basin = read_basin_result(site, scenario, attribute,
                                                                                         scale, metric, id_best,
                                                                                         task)
    basin_rvs = ['Y', 'M', 'D', 'LAI', 'BIOM', 'STL', 'STD', 'STDL', 'PRCP', 'TMX', 'TMN', 'PET', 'ET', 'Q', 'CN',
                 'SSF', 'PRK', 'QDR', 'IRGA', 'USLE', 'MUSL', 'REMX', 'MUSS', 'MUST', 'RUSL', 'YN', 'YP', 'QN', 'QP',
                 'QDRN', 'QPRP', 'SSFN', 'RSFN', 'QRFN', 'QRFP', 'QDRP', 'DPRK', 'TN', 'TP']
    outlet_rvs = ['RFV', 'WYLD', 'RUS2', 'YSD']
    run_daily = pd.concat([data_run_basin[basin_rvs], data_run_outlet[outlet_rvs]], axis=1)
    cal_daily = pd.concat([data_cal_basin[basin_rvs], data_cal_outlet[outlet_rvs]], axis=1)
    val_daily = pd.concat([data_val_basin[basin_rvs], data_val_outlet[outlet_rvs]], axis=1)
    data_daily = pd.concat([data_daily_basin[basin_rvs], data_daily_outlet[outlet_rvs]], axis=1)
    run_daily, data_daily = _insert_doy(run_daily), _insert_doy(data_daily)
    cal_daily, val_daily = _insert_doy(cal_daily), _insert_doy(val_daily)
    annual_set = {'Entire': data_run, 'Calibration': data_cal, 'Validation': data_val, 'Simulation': data_annual}
    daily_set = {'Entire': run_daily, 'Calibration': cal_daily, 'Validation': val_daily, 'Simulation': data_daily}
    return daily_set, annual_set, id_best


def _insert_doy(df):
    df_new = df.copy()
    df_new.index = df.index.strftime('%Y-%m-%d')
    df_new.insert(3, 'DOY', np.nan)
    for i in range(df_new.shape[0]):
        df_new.DOY[i] = dayOfYear(df_new.index[i])
    df_new.DOY = df_new.DOY.astype(int)
    df_new.index = pd.to_datetime(df_new.index)
    return df_new


def convert_daily2annual(df_daily, df_annual, avg_rv, max_rv, sum_rv):
    rv_list = df_daily.columns[4:]
    df = df_daily[rv_list]
    df_min, df_max = df.resample('Y').min(), df.resample('Y').max()
    df_avg, df_sum = df.resample('Y').mean(), df.resample('Y').sum()
    annual_max = df_max[max_rv]
    annual_mean = df_avg[avg_rv]
    annual_sum = df_sum[sum_rv]
    data_annual = pd.concat([annual_sum, annual_mean, annual_max], axis=1)
    data_annual.index = data_annual.index.year.astype(int)
    data_annual.insert(data_annual.shape[1], 'TMN', df_min.TMN.values)
    data_annual = pd.concat([data_annual, df_annual], axis=1)
    df_summary = data_annual.mean(axis=0)
    return df_summary


def compile_gather(site, scenario, attribute, scale, metric, task):
    data_daily, data_annual, id_best = get_model_result(site, scenario, attribute, scale, metric, task)

    run_daily, cal_daily, val_daily = data_daily['Entire'], data_daily['Calibration'], data_daily['Validation']
    run_annual, cal_annual, val_annual = data_annual['Entire'], data_annual['Calibration'], data_annual['Validation']

    max_rv = ['BIOM', 'STL', 'STD', 'STDL', 'TMX']
    avg_rv = ['LAI', 'CN']
    sum_rv = ['PRCP', 'PET', 'ET', 'Q', 'SSF', 'PRK', 'QDR', 'IRGA', 'USLE', 'MUSL', 'REMX', 'MUSS', 'MUST', 'RUSL',
              'YN',
              'YP', 'QN', 'QP', 'QDRN', 'QPRP', 'SSFN', 'RSFN', 'QRFN', 'QRFP', 'QDRP', 'DPRK', 'TN', 'TP', 'RFV',
              'WYLD',
              'RUS2', 'YSD']

    df_summary_run = convert_daily2annual(run_daily, run_annual, avg_rv, max_rv, sum_rv)
    df_summary_cal = convert_daily2annual(cal_daily, cal_annual, avg_rv, max_rv, sum_rv)
    df_summary_val = convert_daily2annual(val_daily, val_annual, avg_rv, max_rv, sum_rv)
    df_summary = pd.DataFrame(
        {'Calibration': df_summary_cal, 'Validation': df_summary_val, 'Simulation': df_summary_run})

    cal_daily.insert(cal_daily.shape[1], 'STAGE', 'Calibration')
    val_daily.insert(val_daily.shape[1], 'STAGE', 'Validation')
    daily_run = pd.concat([cal_daily, val_daily], axis=0)
    cal_annual.insert(cal_annual.shape[1], 'STAGE', 'Calibration')
    val_annual.insert(val_annual.shape[1], 'STAGE', 'Validation')
    annual_run = pd.concat([cal_annual, val_annual], axis=0)
    if task == 'Calibration':
        daily_run.to_csv(f'../post_analysis/Results/{task}_daily_{site}_{scenario}_{metric}_{id_best:07}.csv')
        annual_run.to_csv(f'../post_analysis/Results/{task}_annual_{site}_{scenario}_{metric}_{id_best:07}.csv')
    else:
        dir2save = f'../post_analysis/Results/{task}/{site}/{scenario}'
        if not os.path.isdir(dir2save):
            os.makedirs(dir2save)
        daily_run.to_csv(f'{dir2save}/daily_{site}_{scenario}_{metric}_{id_best:07}.csv')
        annual_run.to_csv(f'{dir2save}/annual_{site}_{scenario}_{metric}_{id_best:07}.csv')

    df_summary.insert(df_summary.shape[1], 'Watershed', site)
    df_summary.insert(df_summary.shape[1], 'Operation', scenario)
    df_summary.insert(df_summary.shape[1], 'Scale', scale)
    df_summary.insert(df_summary.shape[1], 'Metric', metric)
    return data_daily, data_annual, df_summary


def plot_sns_lines(df, x_var, y_var, hue, ax, str_title, x_lab, y_lab, y_peak):
    ax = sns.lineplot(data=df, x=x_var, y=y_var, hue=hue, palette='Set1', ax=ax)
    ax.set_title(str_title, fontsize=16)
    ax.set_xlabel(x_lab, fontsize=14)
    ax.set_ylabel(y_lab, fontsize=14)
    ax.set_xlim(df[x_var].min(), df[x_var].max())
    ax.set_ylim(0, y_peak)
    ax.grid(True)
    ax.legend_.remove()
    return ax


def plot_sns_2line(df1, df2, x_var, ax, y_var, x_lab, y_lab, str_title):
    sns.lineplot(x=df1[x_var], y=df1[y_var], color='g', ax=ax, label='Without grazing', legend='auto')
    sns.lineplot(x=df2[x_var], y=df2[y_var], color='k', ax=ax, label='With grazing', legend='auto')
    ax.set_title(str_title, fontsize=14)
    ax.set_xlabel(x_lab, fontsize=12)
    ax.set_ylabel(y_lab, fontsize=12)
    ax.set_xlim(df1[x_var].min(), df1[x_var].max())
    ax.grid(True)
    return ax


def plot_sns_scatter(df1, df2, var_list, ax, x_lab, y_lab, str_title):
    sns.scatterplot(x=df1[var_list[0]], y=df1[var_list[1]], color='g', ax=ax)
    sns.scatterplot(x=df2[var_list[0]], y=df2[var_list[1]], color='k', ax=ax)
    ax.set_title(str_title, fontsize=14)
    ax.set_xlabel(x_lab, fontsize=12)
    ax.set_ylabel(y_lab, fontsize=12)
    ax.grid(True)
    return ax


def _get_data__(df, var_list, year):
    var_min = df[var_list].min()
    var_max = df[var_list].max()
    col_list = ['DOY']
    for var in var_list:
        col_list.append(var)
    df_year = df[df.Y == year]
    df_year = df_year[col_list]
    return df_year, var_max, var_min
