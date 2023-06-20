import pandas as pd
import numpy as np

file_name = '../Farm_1/pyAPEX_n/pyAPEX/Output/Statistics_runoff.csv'

df_stats = pd.read_csv(file_name)
df_stats.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
df_stats_daily_all = df_stats[['RunId', 'CODAD', 'RMSEAD', 'NRMSEAD', 'NSEAD', 'PBIASAD', 'IOAAD', 'OF1AD', 'OF2AD']]
df_stats_daily_cal = df_stats[['RunId', 'CODDC', 'RMSEDC', 'NRMSEDC', 'NSEDC', 'PBIASDC', 'IOADC', 'OF1DC', 'OF2DC']]
df_stats_daily_val = df_stats[['RunId', 'CODDV', 'RMSEDV', 'NRMSEDV', 'NSEDV', 'PBIASDV', 'IOADV', 'OF1DV', 'OF2DV']]

df_stats_monthly_all = df_stats[['RunId', 'CODAM', 'RMSEAM', 'NRMSEAM', 'NSEAM', 'PBIASAM', 'IOAAM', 'OF1AM', 'OF2AM']]
df_stats_monthly_cal = df_stats[['RunId', 'CODMC', 'RMSEMC', 'NRMSEMC', 'NSEMC', 'PBIASMC', 'IOAMC', 'OF1MC', 'OF2MC']]
df_stats_monthly_val = df_stats[['RunId', 'CODMV', 'RMSEMV', 'NRMSEMV', 'NSEMV', 'PBIASMV', 'IOAMV', 'OF1MV', 'OF2MV']]

df_stats_yearly_all = df_stats[['RunId', 'CODAY', 'RMSEAY', 'NRMSEAY', 'NSEAY', 'PBIASAY', 'IOAAY', 'OF1AY', 'OF2AY']]
df_stats_yearly_cal = df_stats[['RunId', 'CODYC', 'RMSEYC', 'NRMSEYC', 'NSEYC', 'PBIASYC', 'IOAYC', 'OF1YC', 'OF2YC']]
df_stats_yearly_val = df_stats[['RunId', 'CODYV', 'RMSEYV', 'NRMSEYV', 'NSEYV', 'PBIASYV', 'IOAYV', 'OF1YV', 'OF2YV']]

nCOD_daily = np.sum(df_stats_daily_cal.CODDC.values > 0.6)
nNSE_daily = np.sum(df_stats_daily_cal.NSEDC.values > 0.5)
nPBIAS_daily = np.sum(np.abs(df_stats_daily_cal.PBIASDC.values) <= 15)
nCriteria_daily = np.sum((df_stats_daily_cal.CODDC.values > 0.6) & (df_stats_daily_cal.NSEDC.values > 0.5) & (
            df_stats_daily_cal.PBIASDC.values <= 15))
