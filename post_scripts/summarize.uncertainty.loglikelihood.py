import warnings
import pandas as pd
import numpy as np
import os
from utility import print_progress_bar
from utility import nash, nancorr, pbias
from configobj import ConfigObj

warnings.filterwarnings('ignore')
print('/014')


def get_configuration(is_grazing, field):
    if is_grazing:
        scenario = 'pyAPEX_g'
    else:
        scenario = 'pyAPEX_n'
    file_config = f'../{field}/{scenario}/pyAPEX/runtime.ini'
    config = ConfigObj(file_config)
    return config


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


def partition_data(df_obs, obs_attribute, df_mod, mod_attribute, years_warm_up=4, cal_year=11):
    try:
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
    except Exception as e:
        print(e)
        print('File is empty')
        return np.nan, np.nan
    return df_cal, df_val


def get_file_name(run_id, is_grazing, field, location):
    if is_grazing:
        scenario = 'pyAPEX_g'
    else:
        scenario = 'pyAPEX_n'
    file_loc = f'../{field}/{scenario}/pyAPEX/OutputUncertainty/'
    if (location == 'basin') | (location == 'outlet'):
        file_read = f'daily_{location}_{run_id:07}.csv.csv'
    else:
        file_read = f'{location}_{run_id:07}.csv'
    return file_loc + file_read


def read_model_output(file, attribute):
    data_read = pd.read_csv(file)
    df = data_read[['Date', 'Y', 'M', 'D', attribute]]
    df.index = df.Date
    df = df.drop('Date', axis=1)
    df.index = pd.to_datetime(df.index)
    return df


def read_measured(field, is_grazing, file):
    if is_grazing:
        scenario = 'pyAPEX_g'
    else:
        scenario = 'pyAPEX_n'
    obs_path = f'../{field}/{scenario}/pyAPEX/Program/'
    df = get_measure(obs_path, file)
    df = df.drop('Date', axis=1)
    df.index = pd.to_datetime(df.index)
    return df


def get_glue(field, is_grazing):
    # configuration data
    config = get_configuration(is_grazing=is_grazing, field=field)
    file_observe = config['file_observe']
    max_un, step = int(config['max_range_uncertaintity']), float(config['increment_uncertainty'])
    range_vec = np.arange(-max_un, max_un + step, step)
    n_sim = len(range_vec)
    # read measured data
    df_obs = read_measured(field=field, is_grazing=is_grazing, file=file_observe)
    MSE_vec = []
    NSE_vec = []
    COD_vec = []
    PBIAS_vec = []
    range_set = []
    likelihood_mse_year_mat = pd.DataFrame()
    likelihood_nse_year_mat = pd.DataFrame()
    likelihood_cod_year_mat = pd.DataFrame()
    likelihood_pbias_year_mat = pd.DataFrame()
    mse_year_mat = pd.DataFrame()
    cod_year_mat = pd.DataFrame()
    nse_year_mat = pd.DataFrame()
    pbias_year_mat = pd.DataFrame()
    # for run in range(4119, 4123):
    print_progress_bar(0, n_sim, prefix='', suffix='', decimals=1, length=100, fill='█')
    for run in range(n_sim):
        # read model output
        file_path = get_file_name(run_id=run + 1, is_grazing=is_grazing, field=field, location='outlet')
        df_mod = read_model_output(file=file_path, attribute='WYLD')
        # merge and separate measurement and simulated data
        df_cal, df_val = partition_data(df_obs, obs_attribute='runoff', df_mod=df_mod, mod_attribute='WYLD',
                                        years_warm_up=4,
                                        cal_year=11)
        try:
            df_val = df_val.dropna()
            df_data = pd.concat([df_cal, df_val], axis=0)
            df_data.insert(0, 'Year', df_data.index.year)

            # calculate yearly MSE
            year_vec = df_data.Year.unique()
            mse_vec_year = []
            nse_vec_year = []
            cod_vec_year = []
            pbias_vec_year = []
            for year in year_vec:
                data_year = df_data[df_data.Year == year]
                X, Y = data_year.Observed.values, data_year.Modeled.values
                MSE = np.sum((X - Y) ** 2) / len(X)
                mse_vec_year.append(MSE)
                nse_vec_year.append(nash(X, Y))
                cod_vec_year.append(nancorr(X, Y))
                pbias_vec_year.append(np.abs(pbias(X, Y)))
                del MSE, X, Y
            # calculating annual likelihood
            L_theta_mse_vec_year = []
            L_theta_nse_vec_year = []
            L_theta_cod_vec_year = []
            L_theta_pbias_vec_year = []
            for j in range(len(year_vec)):
                L_theta_mse_vec_year.append(np.exp(-mse_vec_year[j] / np.min(mse_vec_year)))
                L_theta_nse_vec_year.append(np.exp(-nse_vec_year[j] / np.max(nse_vec_year)))
                L_theta_pbias_vec_year.append(np.exp(-cod_vec_year[j] / np.max(cod_vec_year)))
                L_theta_cod_vec_year.append(np.exp(-pbias_vec_year[j] / np.min(pbias_vec_year)))
            df_mse = pd.DataFrame(mse_vec_year, index=year_vec)
            df_nse = pd.DataFrame(nse_vec_year, index=year_vec)
            df_cod = pd.DataFrame(cod_vec_year, index=year_vec)
            df_pbias = pd.DataFrame(pbias_vec_year, index=year_vec)
            df_mse.columns, df_nse.columns = [str(run + 1)], [str(run + 1)]
            df_cod.columns, df_pbias.columns = [str(run + 1)], [str(run + 1)]
            mse_year_mat = pd.concat([mse_year_mat, df_mse], axis=1)
            nse_year_mat = pd.concat([nse_year_mat, df_nse], axis=1)
            cod_year_mat = pd.concat([cod_year_mat, df_cod], axis=1)
            pbias_year_mat = pd.concat([pbias_year_mat, df_pbias], axis=1)
            df_Likelihood_mse = pd.DataFrame(L_theta_mse_vec_year, index=year_vec)
            df_Likelihood_nse = pd.DataFrame(L_theta_nse_vec_year, index=year_vec)
            df_Likelihood_cod = pd.DataFrame(L_theta_cod_vec_year, index=year_vec)
            df_Likelihood_pbias = pd.DataFrame(L_theta_pbias_vec_year, index=year_vec)
            df_Likelihood_mse.columns, df_Likelihood_nse.columns = [str(run + 1)], [str(run + 1)]
            df_Likelihood_cod.columns, df_Likelihood_mse.columns = [str(run + 1)], [str(run + 1)]
            likelihood_mse_year_mat = pd.concat([likelihood_mse_year_mat, df_Likelihood_mse], axis=1)
            likelihood_nse_year_mat = pd.concat([likelihood_nse_year_mat, df_Likelihood_nse], axis=1)
            likelihood_cod_year_mat = pd.concat([likelihood_cod_year_mat, df_Likelihood_cod], axis=1)
            likelihood_pbias_year_mat = pd.concat([likelihood_pbias_year_mat, df_Likelihood_pbias], axis=1)
            # calculate MSE over the simulation period
            X, Y = df_data.Observed.values, df_data.Modeled.values
            MSE_vec.append(np.sum((X - Y) ** 2) / len(X))
            NSE_vec.append(nash(X, Y))
            COD_vec.append(nancorr(X, Y))
            PBIAS_vec.append(np.abs(pbias(X, Y)))
            range_set.append(range_vec[run])
            # print(processing {run + 1}')
            print_progress_bar(run, n_sim, prefix=f'{run+1}', suffix='', decimals=1, length=100, fill='█')
        except Exception as e:
            print(e)
            continue

    if is_grazing:
        idx = 'g'
    else:
        idx = 'n'
    mse_year_mat.to_csv(f'../post_analysis/Results/{field}_{idx}_Uncertainty_annual_mse.csv')
    nse_year_mat.to_csv(f'../post_analysis/Results/{field}_{idx}_Uncertainty_annual_nse.csv')
    cod_year_mat.to_csv(f'../post_analysis/Results/{field}_{idx}_Uncertainty_annual_cod.csv')
    pbias_year_mat.to_csv(f'../post_analysis/Results/{field}_{idx}_Uncertainty_annual_pbias.csv')
    likelihood_mse_year_mat.to_csv(f'../post_analysis/Results/{field}_{idx}_Uncertainty_annual_likelihood_mse.csv')
    likelihood_nse_year_mat.to_csv(f'../post_analysis/Results/{field}_{idx}_Uncertainty_annual_likelihood_nse.csv')
    likelihood_cod_year_mat.to_csv(f'../post_analysis/Results/{field}_{idx}_Uncertainty_annual_likelihood_cod.csv')
    likelihood_pbias_year_mat.to_csv(f'../post_analysis/Results/{field}_{idx}_Uncertainty_annual_likelihood_pbias.csv')
    L_mse_Vec = []
    L_nse_Vec = []
    L_cod_Vec = []
    L_pbias_Vec = []
    for i in range(len(MSE_vec)):
        L_mse_Vec.append(np.exp(-MSE_vec[i] / np.nanmin(MSE_vec)))
        L_nse_Vec.append(np.exp(-NSE_vec[i] / np.nanmax(NSE_vec)))
        L_cod_Vec.append(np.exp(-COD_vec[i] / np.nanmax(COD_vec)))
        L_pbias_Vec.append(np.exp(-PBIAS_vec[i] / np.nanmin(PBIAS_vec)))
    df_out = pd.DataFrame({'Percent': range_set, 'MSE': MSE_vec, 'Likelihood_MSE': L_mse_Vec, 'NSE': NSE_vec,
                           'Likelihood_NSE': L_nse_Vec, 'COD': COD_vec,
                           'Likelihood_COD': L_cod_Vec, 'PBIAS': PBIAS_vec, 'Likelihood_PBIAS': L_pbias_Vec})
    df_out.to_csv(f'../post_analysis/Results/{field}_{idx}_Uncertainty_range.csv')


get_glue(field='Farm_1', is_grazing=False)
get_glue(field='Farm_1', is_grazing=True)
get_glue(field='Farm_8', is_grazing=False)
get_glue(field='Farm_8', is_grazing=True)
