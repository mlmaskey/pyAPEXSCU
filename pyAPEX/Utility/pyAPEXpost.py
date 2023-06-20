import os
import pandas as pd
import numpy as np
from Utility.apex_utility import print_progress_bar
from Utility.easypy import easypy as ep
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

from SALib.analyze import sobol
from SALib.analyze import fast


class pyAPEXpost:
    @staticmethod
    def get_pe_files(attribute, folder='Output'):
        if (attribute == 'WYLD'):
            file_pe = os.path.join(folder, 'Statistics_runoff.csv')
        elif (attribute == 'YSD'):
            file_pe = os.path.join(folder, 'Statistics_sediment.csv')
        else:
            file_pe = os.path.join(folder, f'Statistics_sediment_{attribute}.csv')
        file_parameter = os.path.join(folder, 'APEXPARM.csv')
        return file_pe, file_parameter

    def get_stats(file_name):
        df_pem = pd.read_csv(file_name)
        df_pem.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
        return df_pem

    def get_best_stats(df, scale, vec_criteria):
        if scale == 'daily':
            COD, NSE, PBAIS = 'CODDC', 'NSEDC', 'PBIASDC'
        elif scale == 'monthly':
            COD, NSE, PBAIS = 'CODMC', 'NSEMC', 'PBIASMC'
        elif scale == 'yearly':
            COD, NSE, PBAIS = 'CODYC', 'NSEYC', 'PBIASYC'
        df_pem_daily = df[
            (df[COD] >= vec_criteria[0]) & (df[NSE] >= vec_criteria[1]) & (df[PBAIS].abs() <= vec_criteria[2])]
        df_daily = df_pem_daily[
            ['RunId', 'CODDC', 'RMSEDC', 'NRMSEDC', 'NSEDC', 'PBIASDC', 'IOADC', 'OF1DC', 'OF2DC',
             'CODDV', 'RMSEDV', 'NRMSEDV', 'NSEDV', 'PBIASDV', 'IOADV', 'OF1DV', 'OF2DV']]
        df_monthly = df_pem_daily[
            ['RunId', 'CODMC', 'RMSEMC', 'NRMSEMC', 'NSEMC', 'PBIASMC', 'IOAMC', 'OF1MC', 'OF2MC',
             'CODMV', 'RMSEMV', 'NRMSEMV', 'NSEMV', 'PBIASMV', 'IOAMV', 'OF1MV', 'OF2MV']]
        df_yearly = df_pem_daily[
            ['RunId', 'CODYC', 'RMSEYC', 'NRMSEYC', 'NSEYC', 'PBIASYC', 'IOAYC', 'OF1YC', 'OF2YC',
             'CODYV', 'RMSEYV', 'NRMSEYV', 'NSEYV', 'PBIASYV', 'IOAYV', 'OF1YV', 'OF2YV']]
        return df_daily, df_monthly, df_yearly, df_pem_daily

    def get_best_stats_by_metric(df, scale, vec_criteria, citeria_metric):
        if scale == 'daily':
            COD, NSE, PBAIS = 'CODDC', 'NSEDC', 'PBIASDC'
        elif scale == 'monthly':
            COD, NSE, PBAIS = 'CODMC', 'NSEMC', 'PBIASMC'
        elif scale == 'yearly':
            COD, NSE, PBAIS = 'CODYC', 'NSEYC', 'PBIASYC'
        if citeria_metric == 'COD':
            df_pem = df[df[COD] >= vec_criteria[0]]
        elif citeria_metric == 'NSE':
            df_pem = df[df[NSE] >= vec_criteria[1]]
        elif citeria_metric == 'PBIAS':
            df_pem = df[df[PBAIS].abs() <= vec_criteria[2]]
        df_daily = df_pem[
            ['RunId', 'CODDC', 'RMSEDC', 'NRMSEDC', 'NSEDC', 'PBIASDC', 'IOADC', 'OF1DC', 'OF2DC',
             'CODDV', 'RMSEDV', 'NRMSEDV', 'NSEDV', 'PBIASDV', 'IOADV', 'OF1DV', 'OF2DV']]
        df_monthly = df_pem[
            ['RunId', 'CODMC', 'RMSEMC', 'NRMSEMC', 'NSEMC', 'PBIASMC', 'IOAMC', 'OF1MC', 'OF2MC',
             'CODMV', 'RMSEMV', 'NRMSEMV', 'NSEMV', 'PBIASMV', 'IOAMV', 'OF1MV', 'OF2MV']]
        df_yearly = df_pem[
            ['RunId', 'CODYC', 'RMSEYC', 'NRMSEYC', 'NSEYC', 'PBIASYC', 'IOAYC', 'OF1YC', 'OF2YC',
             'CODYV', 'RMSEYV', 'NRMSEYV', 'NSEYV', 'PBIASYV', 'IOAYV', 'OF1YV', 'OF2YV']]
        return df_daily, df_monthly, df_yearly, df_pem

    def import_output(dir_data, attribute, set_name, crops, run_id, scale, pe_based, dir_save):
        if crops == None:
            file_read = f'{set_name}_{run_id:07}.csv'
            file_path = os.path.join(dir_data, file_read)
            df_data = pd.read_csv(file_path)
            if set_name == 'annual':
                df_data.index = df_data.YR
                df_data = df_data.drop(['Unnamed: 0', 'YR'], axis=1)
            else:
                df_data.Date = pd.to_datetime(df_data.Date)
                df_data.index = df_data.Date
                df_data = df_data.drop('Date', axis=1)
            df_data.to_csv(os.path.join('Output', f'{attribute}/{set_name}_{scale}_{pe_based}.csv'))
            df_data.to_csv(os.path.join(dir_save, f'{set_name}_{scale}_{pe_based}.csv'))
            df_data.to_csv(os.path.join('Output', f'{attribute}/{set_name}_{scale}_{run_id:07}.csv'))
            df_data.to_csv(os.path.join(dir_save, f'{set_name}_{scale}_{run_id:07}.csv'))
            return df_data
        else:
            file_read = f'{set_name}_{run_id:07}.csv'
            file_path = os.path.join(dir_data, file_read)
            df_data = pd.read_csv(file_path)
            if set_name == 'annual':
                df_data.index = df_data.YR
                df_data = df_data.drop(['Unnamed: 0', 'YR'], axis=1)
            else:
                df_data.Date = pd.to_datetime(df_data.Date)
                df_data.index = df_data.Date
                df_data = df_data.drop('Date', axis=1)
            df_data.to_csv(os.path.join('Output', f'{attribute}/{set_name}_{scale}_{pe_based}.csv'))
            df_data.to_csv(os.path.join(dir_save, f'{set_name}_{scale}_{pe_based}.csv'))
            df_data.to_csv(os.path.join('Output', f'{attribute}/{set_name}_{scale}_{run_id:07}.csv'))
            df_data.to_csv(os.path.join(dir_save, f'{set_name}_{scale}_{run_id:07}.csv'))
            df_list = [df_data]
            for crop in crops:
                file_read = f'{set_name}_{run_id:07}_{crop}.csv'
                file_path = os.path.join(dir_data, file_read)
                df_data = pd.read_csv(file_path)
                df_data.index = df_data.Date
                df_data = df_data.drop('Date', axis=1)
                df_data.to_csv(os.path.join('Output', f'{attribute}/{set_name}_{scale}_{crops}_{pe_based}.csv'))
                df_data.to_csv(os.path.join(dir_save, f'{set_name}_{scale}_{pe_based}.csv'))
                df_data.to_csv(os.path.join('Output', f'{attribute}/{set_name}_{scale}_{crops}_{run_id:07}.csv'))
                df_data.to_csv(os.path.join(dir_save, f'{set_name}_{scale}_{run_id:07}.csv'))
                df_list.append(df_data)
            return df_list

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

    def match_data(df_obs, df_mod):
        end_date_measure = df_obs.index[-1]
        id_end_measure = np.where(df_mod.index == end_date_measure)[0][0]
        df_model = df_mod.iloc[:(id_end_measure + 1):, :]
        df_sim = df_mod.iloc[(id_end_measure + 1):, :]
        return df_model, df_sim

    def convert_annual(df_obs_daily):
        df_obs = df_obs_daily.resample('Y').mean()
        df_obs.Year = df_obs.Year.astype(int)
        year_vec = df_obs.Year.values
        df_obs.index = year_vec
        df_obs = df_obs[['runoff']]
        df_obs_sediment = df_obs_daily.resample('Y').sum()
        df_obs_sediment = df_obs_sediment[['sediment']]
        df_obs['sediment'] = df_obs_sediment.sediment.values
        df_obs.index = year_vec
        return df_obs

    def match_data_annual(df_obs, df_mod):
        end_date_measure = df_obs.index[-1]
        id_end_measure = np.where(df_mod.index == end_date_measure)[0][0]
        df_model = df_mod.iloc[:(id_end_measure + 1):, :]
        df_sim = df_mod.iloc[(id_end_measure + 1):, :]
        return df_model, df_sim

    def partition_data(df_obs, obs_attribute, df_mod, mod_attribute, years_warm_up=4, cal_year=11):
        year_start = df_mod.Y[0]
        cal_start = year_start + years_warm_up
        cal_end = cal_start + cal_year
        val_end = df_obs.Year[-1]
        df_obs_cal = df_obs[(df_obs.Year >= cal_start) & (df_obs.Year <= cal_end)]
        df_obs_val = df_obs[(df_obs.Year > cal_end) & (df_obs.Year <= val_end)]
        df_model_cal_data = df_mod[(df_mod.Y >= cal_start) & (df_mod.Y <= cal_end)]
        df_model_val_data = df_mod[(df_mod.Y > cal_end) & (df_mod.Y <= val_end)]
        df_cal = pd.concat([df_obs_cal[obs_attribute], df_model_cal_data[mod_attribute]], axis=1)
        df_cal.columns = ['Observed', 'Modeled']
        df_val = pd.concat([df_obs_val[obs_attribute], df_model_val_data[mod_attribute]], axis=1)
        df_val.columns = ['Observed', 'Modeled']
        return df_cal, df_val

    def partition_data_annual(df_obs, obs_attribute, df_mod, mod_attribute, years_warm_up=4, cal_year=11):
        year_start = df_mod.index[0]
        cal_start = year_start + years_warm_up
        cal_end = cal_start + cal_year
        val_end = df_obs.index[-1]
        df_obs_cal = df_obs[(df_obs.index >= cal_start) & (df_obs.index <= cal_end)]
        df_obs_val = df_obs[(df_obs.index > cal_end) & (df_obs.index <= val_end)]
        df_model_cal_data = df_mod[(df_mod.index >= cal_start) & (df_mod.index <= cal_end)]
        df_model_val_data = df_mod[(df_mod.index > cal_end) & (df_mod.index <= val_end)]
        df_cal = pd.concat([df_obs_cal[obs_attribute], df_model_cal_data[mod_attribute]], axis=1)
        df_cal.columns = ['Observed', 'Modeled']
        df_val = pd.concat([df_obs_val[obs_attribute], df_model_val_data[mod_attribute]], axis=1)
        df_val.columns = ['Observed', 'Modeled']
        return df_cal, df_val

    def find_best(df_best, metric, stats='min'):
        value_vector = df_best[metric].abs().values
        if stats == 'min':
            best_value = value_vector.min()
        else:
            best_value = value_vector.max()
        idbest = np.where(value_vector == best_value)[0][0]
        # print(f'Best {metric} value, %.3f' % best_value)
        best_run = df_best.iloc[idbest, 0]
        # print(f'Best iteration  with respect to best {metric}, %1d' % best_run)
        # print('----------------------------------------------------------------')
        best_stats = df_best.iloc[idbest, :]
        # print(best_stats)
        return best_value, best_run, best_stats

    def get_best_params(file_name, ids_bests, scale):
        df_param = pd.read_csv(file_name)
        df_param.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
        df_param_of = df_param[df_param['RunId'] == ids_bests[0]]
        df_param_nse = df_param[df_param['RunId'] == ids_bests[1]]
        df_param_pbias = df_param[df_param['RunId'] == ids_bests[2]]
        df_param_cod = df_param[df_param['RunId'] == ids_bests[3]]
        df_param_best = pd.concat([df_param_of, df_param_nse, df_param_pbias, df_param_cod])
        df_param_best.index = ['ObjectiveFunction', 'NSE', 'PBIAS', 'COD']
        df_param_best['SCALE'] = scale
        return df_param_best

    def get_param_bests(file_name, id_vec, scale):
        df_param = pd.read_csv(file_name)
        df_param.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
        df_param_criteria = pd.DataFrame(columns=df_param.columns)
        df_param_t = df_param.T
        df_param_t.columns = df_param_t.loc['RunId', :]
        df_param_t.columns = df_param_t.columns.astype(int)
        df_param_criteria_t = df_param_t[id_vec]
        df_param_criteria = df_param_criteria_t.T
        df_param_criteria.index = np.arange(0, df_param_criteria.shape[0])
        return df_param_criteria

    def import_save(dir_data, attribute, croplist, ids, scale, dir_save):
        from Utility.pyAPEXpost import pyAPEXpost as ap
        metrics = ['OF', 'NSE', 'PBIAS', 'COD']

        df_outlet_of = ap.import_output(dir_data, attribute,
                                        set_name='daily_outlet',
                                        crops=croplist,
                                        run_id=ids[0],
                                        scale=scale,
                                        pe_based=metrics[0],
                                        dir_save=dir_save)

        df_basin_of = ap.import_output(dir_data, attribute,
                                       set_name='daily_basin',
                                       crops=croplist,
                                       run_id=ids[0],
                                       scale=scale,
                                       pe_based=metrics[0],
                                       dir_save=dir_save)

        df_annual_of = ap.import_output(dir_data, attribute,
                                        set_name='annual',
                                        crops=croplist,
                                        run_id=ids[0],
                                        scale=scale,
                                        pe_based=metrics[0],
                                        dir_save=dir_save)

        df_outlet_nse = ap.import_output(dir_data, attribute,
                                         set_name='daily_outlet',
                                         crops=croplist,
                                         run_id=ids[1],
                                         scale=scale,
                                         pe_based=metrics[1],
                                         dir_save=dir_save)

        df_basin_nse = ap.import_output(dir_data, attribute,
                                        set_name='daily_basin',
                                        crops=croplist,
                                        run_id=ids[1],
                                        scale=scale,
                                        pe_based=metrics[1],
                                        dir_save=dir_save)

        df_annual_nse = ap.import_output(dir_data, attribute,
                                         set_name='annual',
                                         crops=croplist,
                                         run_id=ids[1],
                                         scale=scale,
                                         pe_based=metrics[1],
                                         dir_save=dir_save)

        df_outlet_pbias = ap.import_output(dir_data, attribute,
                                           set_name='daily_outlet',
                                           crops=croplist,
                                           run_id=ids[2],
                                           scale=scale,
                                           pe_based=metrics[2],
                                           dir_save=dir_save)

        df_basin_pbias = ap.import_output(dir_data, attribute,
                                          set_name='daily_basin',
                                          crops=croplist,
                                          run_id=ids[2],
                                          scale=scale,
                                          pe_based=metrics[2],
                                          dir_save=dir_save)

        df_annual_pbias = ap.import_output(dir_data, attribute,
                                           set_name='annual',
                                           crops=croplist,
                                           run_id=ids[2],
                                           scale=scale,
                                           pe_based=metrics[2],
                                           dir_save=dir_save)

        df_outlet_cod = ap.import_output(dir_data, attribute,
                                         set_name='daily_outlet',
                                         crops=croplist,
                                         run_id=ids[3],
                                         scale=scale,
                                         pe_based=metrics[3],
                                         dir_save=dir_save)

        df_daily_basin_cod = ap.import_output(dir_data, attribute,
                                              set_name='daily_basin',
                                              crops=croplist,
                                              run_id=ids[3],
                                              scale=scale,
                                              pe_based=metrics[3],
                                              dir_save=dir_save)

        df_annual_cod = ap.import_output(dir_data, attribute,
                                         set_name='annual',
                                         crops=croplist,
                                         run_id=ids[3],
                                         scale=scale,
                                         pe_based=metrics[3],
                                         dir_save=dir_save)

        outlet = (df_outlet_of, df_outlet_nse, df_outlet_pbias, df_outlet_cod)
        basin = (df_basin_of, df_basin_nse, df_basin_pbias, df_daily_basin_cod)
        annual = (df_annual_of, df_annual_nse, df_annual_pbias, df_annual_cod)
        return outlet, basin, annual

    def summarize_stats(df, scale, metrics=['OF2DC', 'NSEDC', 'PBIASDC', 'CODDC'], stats=['min', 'max', 'min', 'max']):
        from Utility.pyAPEXpost import pyAPEXpost as ap
        # Objective function based
        best_of_value, id_of_run, best_of_stats = ap.find_best(df, metric=metrics[0], stats=stats[0])
        # NSE based bast set:
        best_nse_value, id_nse_run, best_nse_stats = ap.find_best(df, metric=metrics[1], stats=stats[1])
        # PBIAS based bast set:
        best_pbias_value, id_pbias_run, best_pbias_stats = ap.find_best(df, metric=metrics[2], stats=stats[2])
        # COD based bast set:
        best_cod_value, id_cod_run, best_cod_stats = ap.find_best(df, metric=metrics[3], stats=stats[3])
        best_stats = pd.concat([pd.DataFrame(best_of_stats).T,
                                pd.DataFrame(best_nse_stats).T,
                                pd.DataFrame(best_pbias_stats).T, pd.DataFrame(best_cod_stats).T], axis=0)
        best_stats.RunId = best_stats.RunId.astype(int)
        best_stats.index = ['Objective Function', 'NSE', 'PBIAS', 'COD']
        best_stats['SCALE'] = scale
        best_stats.columns = ['RunId', 'CODC', 'RMSEC', 'NRMSEC', 'NSEC', 'PBIASC', 'IOAC', 'OF1C', 'OF2C',
                              'CODV', 'RMSEV', 'NRMSEV', 'NSEV', 'PBIASV', 'IOAV', 'OF1V', 'OF2V', 'SCALE']
        return best_stats, (id_of_run, id_nse_run, id_pbias_run, id_cod_run)

    def compile_stats(df, criteria, scale):
        from Utility.pyAPEXpost import pyAPEXpost as ap
        df_daily, df_monthly, df_yearly, df_all = ap.get_best_stats(df, scale, criteria)
        df_stats = pd.concat([df_daily, df_monthly, df_yearly], axis=1)
        df_stats['SCALE'] = scale
        df_stats = df_stats.T.drop_duplicates().T
        return df_stats, df_daily, df_monthly, df_yearly, df_all

    def compile_stats_by_metrics(df, criteria, scale, metric):
        from Utility.pyAPEXpost import pyAPEXpost as ap
        df_daily, df_monthly, df_yearly, df_all = ap.get_best_stats_by_metric(df, scale, criteria, metric)
        df_stats = pd.concat([df_daily, df_yearly, df_yearly], axis=1)
        df_stats['SCALE'] = scale
        df_stats = df_stats.T.drop_duplicates().T
        return df_stats, df_daily, df_monthly, df_yearly, df_all

    def finalize_outlet_result(dir_read, df_obs, obs_attribute, mod_attribute, scale, metric):
        from Utility.pyAPEXpost import pyAPEXpost as ap
        df_raw = pd.read_csv(f'{dir_read}/daily_outlet_{scale}_{metric}.csv', index_col=0)
        df_raw.index = pd.to_datetime(df_raw.index)
        df_model, df_sim = ap.match_data(df_obs, df_raw)
        df_cal, df_val = ap.partition_data(df_obs=df_obs, obs_attribute=obs_attribute,
                                           df_mod=df_model, mod_attribute=mod_attribute,
                                           years_warm_up=4, cal_year=11)
        df_model = df_model[(df_model.index >= df_cal.index[0])]
        df_model['STAGE'], df_sim['STAGE'] = 'Calibration', 'Simulation'
        df_cal['STAGE'], df_val['STAGE'] = 'Calibration', 'Validation'
        df_model.STAGE[(df_model.index >= df_val.index[0]) &
                       (df_model.index <= df_val.index[-1])] = 'Validation'
        df_model = pd.concat([df_model, df_sim], axis=0)
        df_cal_val = pd.concat([df_cal, df_val], axis=0)
        df_model['SCALE'] = scale
        df_cal_val['SCALE'] = scale
        return df_model, df_cal_val

    def finalize_basin_result(dir_read, df_obs, df_ref, obs_attribute, mod_attribute, scale, metric):
        from Utility.pyAPEXpost import pyAPEXpost as ap
        df_raw = pd.read_csv(f'{dir_read}/daily_basin_{scale}_{metric}.csv', index_col=0)
        df_raw.index = pd.to_datetime(df_raw.index)
        df_cal = df_raw[(df_raw.index >= df_ref.index[df_ref.STAGE == 'Calibration'][0]) &
                        (df_raw.index <= df_ref.index[df_ref.STAGE == 'Calibration'][-1])]
        df_val = df_raw[(df_raw.index >= df_ref.index[df_ref.STAGE == 'Validation'][0]) &
                        (df_raw.index <= df_ref.index[df_ref.STAGE == 'Validation'][-1])]
        df_sim = df_raw[df_raw.index > df_ref.index[df_ref.STAGE == 'Validation'][-1]]
        df_cal['STAGE'], df_val['STAGE'], df_sim['STAGE'] = 'Calibration', 'Validation', 'Simulation'
        df_model = pd.concat([df_cal, df_val, df_sim], axis=0)
        df_model['SCALE'] = scale
        return df_model

    def finalize_annual_result(dir_read, df_obs, df_ref, obs_attribute, mod_attribute, scale, metric):
        from Utility.pyAPEXpost import pyAPEXpost as ap
        df_raw = pd.read_csv(f'{dir_read}/annual_{scale}_{metric}.csv', index_col=0)
        df_cal = df_raw[(df_raw.index >= df_ref.index[df_ref.STAGE == 'Calibration'][0].year) &
                        (df_raw.index <= df_ref.index[df_ref.STAGE == 'Calibration'][-1].year)]
        df_val = df_raw[(df_raw.index >= df_ref.index[df_ref.STAGE == 'Validation'][0].year) &
                        (df_raw.index <= df_ref.index[df_ref.STAGE == 'Validation'][-1].year)]
        df_sim = df_raw[df_raw.index > df_ref.index[df_ref.STAGE == 'Validation'][-1].year]
        df_cal['STAGE'], df_val['STAGE'], df_sim['STAGE'] = 'Calibration', 'Validation', 'Simulation'
        df_model = pd.concat([df_cal, df_val, df_sim], axis=0)
        df_model['SCALE'] = scale
        return df_model

    def get_FAST_index(param_list, list_bound, Y):
        problem = {'num_vars': len(param_list),
                   'names': param_list,
                   'bounds': list_bound
                   }
        Sif = fast.analyze(problem, Y, print_to_console=False)
        total = Sif.to_df()
        df_fast_total = pd.DataFrame(total.ST)
        df_fast_total.columns = ['Sensitivity Index']
        df_fast_total['Order'] = 'Total'
        df_fast_total['Method'] = 'FAST'
        df_fast_First = pd.DataFrame(total.S1)
        df_fast_First.columns = ['Sensitivity Index']
        df_fast_First['Order'] = 'First'
        df_fast_First['Method'] = 'FAST'
        df_fast_First['PARAM'] = df_fast_First.index
        df_fast_total['PARAM'] = df_fast_total.index
        return df_fast_First, df_fast_total

    def getSOBOL_index(param_list, list_bound, Y):
        problem = {'num_vars': len(param_list),
                   'names': param_list,
                   'bounds': list_bound
                   }
        Si = sobol.analyze(problem, Y, print_to_console=False)
        total, first, second = Si.to_df()
        df_Sobol_total = pd.DataFrame(total.ST)
        df_Sobol_total.columns = ['Sensitivity Index']
        df_Sobol_total['Order'] = 'Total'
        df_Sobol_total['Method'] = 'SOBOL'
        df_Sobol_first = pd.DataFrame(first.S1)
        df_Sobol_first.columns = ['Sensitivity Index']
        df_Sobol_first['Order'] = 'First'
        df_Sobol_first['Method'] = 'SOBOL'
        df_Sobol_first['PARAM'] = df_Sobol_first.index
        df_Sobol_total['PARAM'] = df_Sobol_total.index
        return df_Sobol_first, df_Sobol_total

    def sensitivity_index(x, y):
        del_x, del_y = np.diff(x), np.diff(y)
        s = (del_y / del_x) * y[:-1]
        df_s = pd.DataFrame(s)
        df_s.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_s = df_s.dropna()
        s = df_s.values
        if len(s) == 0:
            si = 0
        else:
            si = np.mean(s)
        return si

    def standarizedRegressionCoefficient1(x, y, intercept=True):
        df = pd.DataFrame({'x': x.ravel(), 'y': y.ravel()})
        df = df.dropna()
        sd_x = np.std(x)
        sd_y = np.std(y)
        x, y = df.x.values, df.y.values
        X = x.reshape((-1, 1))
        model = LinearRegression(fit_intercept=intercept)
        model.fit(X, y)
        coef = float(model.coef_)
        src = (sd_x / sd_y) * coef
        return src

    def standarizedRegressionCoefficient(x, y, intercept=True):
        df_x = pd.DataFrame({'x': x.ravel()})
        df_y = pd.DataFrame({'y': y.ravel()})
        sd_x = df_x.std()
        sd_y = df_y.std()
        formula = f'y~{df_x.columns[0]}'
        model = smf.ols(formula, data=df_x).fit()
        coefficients = model.params[1:]
        src = float(sd_x.values / sd_y.values) * float(coefficients.values)
        return src

    def standarizedRegressionCoefficientTotal1(df_x, df_y, intercept=False):
        sd_x = df_x.std()
        sd_y = df_y.std()
        model = LinearRegression(fit_intercept=intercept)
        model.fit(df_x, df_y)
        coef = model.coef_
        src = (sd_x / sd_y) * coef
        return src

    def standarizedRegressionCoefficientTotal(df_x, df_y, intercept=False):
        sd_x = df_x.std()
        sd_y = df_y.std()
        y = df_y.values
        formula = f'y~{df_x.columns[0]}'
        for col in df_x.columns[1:]:
            formula = f'{formula}+{col}'
        model = smf.ols(formula, data=df_x).fit()
        coefficients = model.params[1:]
        src = (sd_x / sd_y) * coefficients
        return src