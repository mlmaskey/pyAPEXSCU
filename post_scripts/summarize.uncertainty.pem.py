import pandas as pd
import numpy as np


def read_save_pem_summary(site, scenario):
    if scenario == 'grazing':
        sub_folder = 'pyAPEX_g'
    else:
        sub_folder = 'pyAPEX_n'
    file_stat = f'../{site}/{sub_folder}/pyAPEX/OutputUncertainty/Statistics_runoff.csv'
    sigma_inc = np.arange(-3, 3, 0.001)
    data_stats = pd.read_csv(file_stat)
    data_stats = data_stats.rename({'Unnamed: 0': 'RunId'}, axis=1)
    data_stats.RunId = data_stats.RunId.astype(int)
    stat_list_daily = ['RunId', 'CODDC', 'NSEDC', 'PBIASDC', 'OF2DC', 'CODDV', 'NSEDV', 'PBIASDV', 'OF2DV']
    stat_list_monthly = ['RunId', 'CODMC', 'NSEMC', 'PBIASMC', 'OF2MC', 'CODMV', 'NSEMV', 'PBIASMV', 'OF2MV']
    stat_list_yearly = ['RunId', 'CODYC', 'NSEYC', 'PBIASYC', 'OF2YC', 'CODYV', 'NSEYV', 'PBIASYV', 'OF2YV']
    df_daily = data_stats[stat_list_daily]
    df_monthly = data_stats[stat_list_monthly]
    df_yearly = data_stats[stat_list_yearly]
    df_daily_best = df_daily.iloc[0, :]
    df_monthly_best = df_monthly.iloc[0, :]
    df_yearly_best = df_yearly.iloc[0, :]
    df_best = pd.DataFrame({'Daily': df_daily_best.values, 'Monthly': df_monthly_best.values, 'Yearly': df_yearly_best})
    df_best.index = ['RunId', 'CODC', 'NSEC', 'PBIASC', 'OFC', 'CODV', 'NSEV', 'PBIASV', 'OFV']
    data_stats = data_stats.iloc[1:, :]
    data_stats_1s = data_stats.iloc[1999:3000, :]
    df_daily_1s = data_stats_1s[stat_list_daily]
    df_monthly_1s = data_stats_1s[stat_list_monthly]
    df_yearly_1s = data_stats_1s[stat_list_yearly]
    df_daily_1s.columns = [['RunId', 'CODC', 'NSEC', 'PBIASC', 'OFC', 'CODV', 'NSEV', 'PBIASV', 'OFV']]
    df_monthly_1s.columns = [['RunId', 'CODC', 'NSEC', 'PBIASC', 'OFC', 'CODV', 'NSEV', 'PBIASV', 'OFV']]
    df_yearly_1s.columns = [['RunId', 'CODC', 'NSEC', 'PBIASC', 'OFC', 'CODV', 'NSEV', 'PBIASV', 'OFV']]
    df_mean_daily = df_daily_1s.mean(axis=0)
    df_std_daily = df_daily_1s.std(axis=0)
    df_mean_monthly = df_monthly_1s.mean(axis=0)
    df_std_monthly = df_monthly_1s.std(axis=0)
    df_mean_yearly = df_yearly_1s.mean(axis=0)
    df_std_yearly = df_yearly_1s.std(axis=0)

    df_stats = pd.DataFrame({'Daily mean': df_mean_daily.values, 'Daily standard deviation': df_std_daily.values,
                             'Monthly mean': df_mean_monthly.values,
                             'Monthly standard deviation': df_std_monthly.values,
                             'Yearly mean': df_mean_yearly.values, 'Yearly standard deviation': df_std_yearly.values})
    df_stats.index = ['RunId', 'CODC', 'NSEC', 'PBIASC', 'OFC', 'CODV', 'NSEV', 'PBIASV', 'OFV']
    df_stats.insert(df_stats.shape[1], 'Site', site)
    df_stats.insert(df_stats.shape[1], 'Operation', scenario)
    return df_stats, df_best


df_stats1n, df_best1n = read_save_pem_summary(site='Farm_1', scenario='non_grazing')
df_stats1g, df_best1g = read_save_pem_summary(site='Farm_1', scenario='grazing')
df_stats8n, df_best8n = read_save_pem_summary(site='Farm_8', scenario='non_grazing')
df_stats8g, df_best8g = read_save_pem_summary(site='Farm_8', scenario='grazing')

df_stats_1s = pd.concat([df_stats1n.iloc[1:, :], df_stats1g.iloc[1:, :], df_stats8n.iloc[1:, :],
                         df_stats8g.iloc[1:, :]], axis=0)
df_best = pd.concat([df_best1n, df_best1g, df_best8n, df_best8g], axis=1)

df_stats_1s.to_csv('../post_analysis/Results/Uncertainty_performance_summary.csv')
df_best.to_csv('../post_analysis/Results/Calibration_performance_best.csv')

