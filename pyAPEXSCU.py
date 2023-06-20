# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:12:23 2022

@author: Mahesh.Maskey, Brian Stucky (USDA)
"""
import os
import pandas as pd
import numpy as np
import subprocess
import random
import fortranformat as ff
from Utility.apex_utility import read_sensitive_params, copy_rename_file, get_scanario_name
from Utility.apex_utility import modify_list
from Utility.apex_utility import get_acy
from Utility.apex_utility import get_daily_sad
from Utility.apex_utility import get_daily_dps
from Utility.apex_utility import get_daily_dws
from Utility.apex_utility import read_param_file
from Utility.overwrite_param import overwrite_param
from Utility.apex_utility import savedata_rel1
from Utility.apex_utility import organize2save
from Utility.apex_utility import do_validate_fill
from Utility.apex_utility import calculate_nutrients
from Utility.easypy import easypy as ep
from Utility.pyAPEXpost import pyAPEXpost as ap
from datetime import datetime


class simAPEX:
    def __init__(
            self, config, src_dir, winepath, in_obj, model_mode, scale=None,
            isall=True
    ):
        """
        self.config: A configuration object.
        src_dir: The location of the source repository for managing APEX runs.
        winepath: Path to an executable Wine container image.
        in_obj: An inAPEX object.
        """
        self.config = config
        self.src_dir = src_dir
        self.winepath = winepath
        self.run_name = self.config['run_name']
        self.file_limits = self.src_dir / 'Utility' / self.config['file_limits']

        self.dir_calibration = in_obj.dir_calibration
        self.dir_sensitivity = os.path.join(os.path.dirname(in_obj.dir_sensitivity), 'OutputSensitivity')
        self.dir_uncertainty = os.path.join(os.path.dirname(in_obj.dir_uncertainty), 'OutputUncertainty')
        in_obj.dir_sensitivity, in_obj.dir_uncertainty = self.dir_sensitivity, self.dir_uncertainty
        self.senstitivty_out_file = os.path.join(self.dir_sensitivity, 'Output_summary.txt')
        self.uncertainty_outflie = os.path.join(self.dir_uncertainty, 'Output_summary.txt')

        self.get_range()
        if (model_mode == "calibration"):
            n_discrete = int(self.config['n_discrete'])
            n_simul = int(self.config['n_simulation'])
            self.generate_param_set(n_discrete, isall)
        else:
            dir_res = self.config['dir_calibrate_res']
            file_pem = self.config['file_pem']
            self.file_pem = dir_res + '/' + file_pem
            # reads the performance statistics file to get the best params
            self.get_stats()
            file_parm = self.config['file_param']
            # reads the parameter file from the calibration results
            self.file_parm = dir_res + '/' + file_parm
            self.read_params()
            if (model_mode == "sensitivity"):
                # file for basic info
                if os.path.exists(self.dir_sensitivity) is False:
                    os.makedirs(self.dir_sensitivity)
                f = open(self.senstitivty_out_file, 'w')
                f.close()
                self.get_best4sa(scale='daily')
                self.generate_sensitive_params(isall)
            elif (model_mode == "uncertainty"):
                if os.path.exists(self.dir_uncertainty) is False:
                    os.makedirs(self.dir_uncertainty)
                f = open(self.uncertainty_outflie, 'w')
                f.close()
                self.get_params4ua(scale)
                self.generate_uncertaintity_params(isall)

        self.recommended_parameters = self.recc_params
        self.n_params = len(self.recc_params)
        self.pem = ['CODAD', 'RMSEAD', 'NRMSEAD', 'NSEAD', 'PBIASAD', 'IOAAD', 'OF1AD', 'OF2AD',
                    'CODDC', 'RMSEDC', 'NRMSEDC', 'NSEDC', 'PBIASDC', 'IOADC', 'OF1DC', 'OF2DC',
                    'CODDV', 'RMSEDV', 'NRMSEDV', 'NSEDV', 'PBIASDV', 'IOADV', 'OF1DV', 'OF2DV',
                    'CODAM', 'RMSEAM', 'NRMSEAM', 'NSEAM', 'PBIASAM', 'IOAAM', 'OF1AM', 'OF2AM',
                    'CODMC', 'RMSEMC', 'NRMSEMC', 'NSEMC', 'PBIASMC', 'IOAMC', 'OF1MC', 'OF2MC',
                    'CODMV', 'RMSEMV', 'NRMSEMV', 'NSEMV', 'PBIASMV', 'IOAMV', 'OF1MV', 'OF2MV',
                    'CODAY', 'RMSEAY', 'NRMSEAY', 'NSEAY', 'PBIASAY', 'IOAAY', 'OF1AY', 'OF2AY',
                    'CODYC', 'RMSEYC', 'NRMSEYC', 'NSEYC', 'PBIASYC', 'IOAYC', 'OF1YC', 'OF2YC',
                    'CODYV', 'RMSEYV', 'NRMSEYV', 'NSEYV', 'PBIASYV',  'IOAYV', 'OF1YV', 'OF2YV']
        if (model_mode == "calibration"):

            id_start = int(self.config['n_start'])
            n_simul = int(self.config['n_simulation'])
        else:
            id_start = int(self.config['n_start'])
            id_start = 0
            n_simul = self.parameters_matrix.shape[0]

        self.assign_output_df()
        if (model_mode == "calibration"):
            df_PEM_runoff, df_PEM_sediment_YSD, df_PEM_sediment_USLE = pd.DataFrame(columns=self.pem), pd.DataFrame(
                columns=self.pem), pd.DataFrame(columns=self.pem)
            df_PEM_sediment_MUSL, df_PEM_sediment_REMX, df_PEM_sediment_MUSS = pd.DataFrame(
                columns=self.pem), pd.DataFrame(columns=self.pem), pd.DataFrame(columns=self.pem)
            df_PEM_sediment_MUST, df_PEM_sediment_RUS2, df_PEM_sediment_RUSL = pd.DataFrame(
                columns=self.pem), pd.DataFrame(columns=self.pem), pd.DataFrame(columns=self.pem)
            self.df_p_set = pd.DataFrame()
        else:
            if 'RunId' in self.df_calpem_sets:
                df_PEM_runoff = pd.DataFrame(self.df_calpem_sets).T
                df_PEM_runoff.index = df_PEM_runoff.RunId.values
                df_PEM_runoff = df_PEM_runoff.drop(['RunId'], axis=1)
            else:
                df_PEM_runoff = pd.DataFrame(self.df_calpem_sets).T
            df_PEM_sediment_YSD, df_PEM_sediment_USLE = pd.DataFrame(columns=self.pem), pd.DataFrame(columns=self.pem)
            df_PEM_sediment_MUSL, df_PEM_sediment_REMX, df_PEM_sediment_MUSS = pd.DataFrame(
                columns=self.pem), pd.DataFrame(columns=self.pem), pd.DataFrame(columns=self.pem)
            df_PEM_sediment_MUST, df_PEM_sediment_RUS2, df_PEM_sediment_RUSL = pd.DataFrame(
                columns=self.pem), pd.DataFrame(columns=self.pem), pd.DataFrame(columns=self.pem)

            self.df_p_set = self.pbest
        # start simulation 
        t0 = datetime.now()
        for i in range(id_start, n_simul):
            try:
                t1 = datetime.now()
                if (model_mode == "calibration"):
                    if isall:
                        self.pick_param(allparam=True, i=i)
                    else:
                        self.pick_param(allparam=False, i=i)
                    # params = np.array([v.replace(',', '') for v in params], dtype=np.float64)
                else:
                    self.p = self.parameters_matrix[i, :]
                self.p = overwrite_param(in_obj.param_file, in_obj.simparam_file, self.p)
                modify_list(in_obj.list_file, in_obj.simparam_file)

                t2 = datetime.now()
                print('Calling APEXgraze')
                if os.name == 'nt':
                    p = subprocess.run(['APEXgraze.exe'], capture_output=True, text=True)
                    print(p.stdout)
                else:
                    p = subprocess.run([self.winepath, 'APEXgraze.exe'], capture_output=True, text=True)
                    print(p.stderr)
                    print(p.stdout)
                # Read run name from APEXRUN.DAT file
                curr_directory = os.getcwd()
                self.scenario_name = get_scanario_name(curr_directory)
                #Saving standard output file together with runs for final summary
                copy_rename_file(curr_directory, self.scenario_name, itr_id=i, extension='OUT', in_obj=in_obj,
                                 model_mode=model_mode)

                # rename run_name_[iteration].out e.g., 001RUN_0000106.out
                print(
                    f'Completed simulation {(i + 1)}/{n_simul} in {round((datetime.now() - t2).total_seconds(), 3)} seconds')
                modify_list(in_obj.list_file, in_obj.param_file)
                df_SAD = get_daily_sad(self.run_name)
                df_SAD = calculate_nutrients(df_SAD)
                df_DPS = get_daily_dps(self.run_name)
                df_DWS = get_daily_dws(self.run_name)
                df_annual = get_acy(self.run_name)
                is_pesticide = self.config['is_pesticide']
                warm_years = int(self.config['warm_years'])
                calib_years = int(self.config['calib_years'])
                print('Saving model performance statistics')
                df_PEM_runoff = do_validate_fill(model_mode, df_SAD, df_PEM_runoff, in_obj, 'WYLD', 'runoff', i,
                                                 warm_years, calib_years)
                if is_pesticide:
                    df_PEM_sediment_YSD = do_validate_fill(model_mode, df_DPS, df_PEM_sediment_YSD, in_obj, 'YSD',
                                                           'sediment', i, warm_years, calib_years)
                df_PEM_sediment_USLE = do_validate_fill(model_mode, df_SAD, df_PEM_sediment_USLE, in_obj, 'USLE',
                                                        'sediment', i, warm_years, calib_years)
                df_PEM_sediment_MUSL = do_validate_fill(model_mode, df_SAD, df_PEM_sediment_MUSL, in_obj, 'MUSL',
                                                        'sediment', i, warm_years, calib_years)
                df_PEM_sediment_REMX = do_validate_fill(model_mode, df_SAD, df_PEM_sediment_REMX, in_obj, 'REMX',
                                                        'sediment', i, warm_years, calib_years)
                df_PEM_sediment_MUSS = do_validate_fill(model_mode, df_SAD, df_PEM_sediment_MUSS, in_obj, 'MUSS',
                                                        'sediment', i, warm_years, calib_years)
                df_PEM_sediment_MUST = do_validate_fill(model_mode, df_SAD, df_PEM_sediment_MUST, in_obj, 'MUST',
                                                        'sediment', i, warm_years, calib_years)
                df_PEM_sediment_RUS2 = do_validate_fill(model_mode, df_SAD, df_PEM_sediment_RUS2, in_obj, 'RUS2',
                                                        'sediment', i, warm_years, calib_years)
                df_PEM_sediment_RUSL = do_validate_fill(model_mode, df_SAD, df_PEM_sediment_RUSL, in_obj, 'RUSL',
                                                        'sediment', i, warm_years, calib_years)
                file_outlet = 'daily_outlet_' + str(i + 1).zfill(7) + '.csv'
                df_outlet = df_DWS[['Y', 'M', 'D', 'RFV', 'WYLD', 'TMX', 'TMN', 'PET', 'Q', 'CN',
                                    'SSF', 'PRK', 'IRGA', 'USLE', 'MUSL', 'REMX',
                                    'MUSS', 'MUST', 'RUS2', 'RUSL']]
                if is_pesticide:
                    df_outlet['YSD'] = df_DPS['YSD']
                savedata_rel1(df_outlet, file_outlet, model_mode, in_obj)

                file_basin = 'daily_basin_' + str(i + 1).zfill(7) + '.csv'
                df_basin = df_SAD[['Y', 'M', 'D', 'CPNM', 'LAI', 'BIOM', 'STL',
                                   'STD', 'STDL', 'PRCP', 'WYLD', 'TMX', 'TMN',
                                   'PET', 'ET', 'Q', 'CN', 'SSF', 'PRK', 'QDR',
                                   'IRGA', 'USLE', 'MUSL', 'REMX', 'MUSS', 'MUST',
                                   'RUS2', 'RUSL', 'YN', 'YP', 'QN', 'QP', 'QDRN',
                                   'QPRP', 'SSFN', 'RSFN', 'QRFN', 'QRFP', 'QDRP',
                                   'DPRK', 'TN', 'TP']]
                savedata_rel1(df_basin, file_basin, model_mode, in_obj)

                file_annual = 'annual_' + str(i + 1).zfill(7) + '.csv'
                df_year = df_annual[['YR', 'CPNM', 'YLDG', 'YLDF', 'BIOM', 'WS',
                                     'NS', 'PS', 'TS', 'AS', 'SS']]
                savedata_rel1(df_year, file_annual, model_mode, in_obj)

                crops = df_basin.CPNM.unique()
                if len(crops) > 1:
                    for cp in crops:
                        file_basin = 'daily_basin_' + str(i + 1).zfill(7) + '_' + cp + '.csv'
                        df_basin_c = df_basin[df_basin['CPNM'] == cp]
                        savedata_rel1(df_basin_c, file_basin, model_mode, in_obj)
                        file_annual = 'annual_' + str(i + 1).zfill(7) + '_' + cp + '.csv'
                        df_year_c = df_year[df_year['CPNM'] == cp]
                        savedata_rel1(df_year_c, file_annual, model_mode, in_obj)

                df_p = pd.DataFrame(self.p)
                df_p = df_p.T
                df_p.columns = self.param_list
                if 'RunId' in self.df_p_set:
                    self.df_p_set.index = self.df_p_set.RunId.values
                    self.df_p_set = self.df_p_set.drop(['RunId'], axis=1)
                self.df_p_set = organize2save(self.df_p_set, df_p, i, axis=0)
                print('Saving parmeters in APEXPARM.DAT')
                savedata_rel1(self.df_p_set, 'APEXPARM', model_mode, in_obj)
                print('Parameters')
                print(self.p)
                print('---------------------------------------------------------------')
                print(f'Completed run no. {i + 1} in {round((datetime.now() - t2).total_seconds(), 3)} seconds')
                print('---------------------------------------------------------------\n')
                print(
                    f'Completed simulation {(i + 1)}/{n_simul} in {round((datetime.now() - t1).total_seconds(), 3)} seconds')
            except Exception as e:
                print(e)
                print(f'error occurs in simulation {i}')
                continue
        print(f'\nCompleted {n_simul - id_start} runs in {round((datetime.now() - t0).total_seconds(), 3)} seconds')

    def get_best4sa(self, scale):
        criteria = [float(self.config['COD_criteria']), float(self.config['NSE_criteria']),
                    float(self.config['PBAIS_criteria'])]
        f = open(self.senstitivty_out_file, 'a')
        f.writelines('---------Calibration and validation resukt summary')
        f.writelines('\n---------------------------------------------------------------')
        f.writelines('Calibratiion criteria according to Moriasi et a. (20015)\n')
        f.writelines(f'Nash Sutcliffe efficiency, NSE     {criteria[1]}\n')
        f.writelines(f'Percent Bias, PBIAS:               {criteria[2]}\n')
        f.writelines(f'Coefficient of determination, COD: {criteria[0]}\n')
        f.writelines('\n---------------------------------------------------------------')
        stats_out = ap.compile_stats(self.stats, criteria, scale)
        if scale == 'daily':
            best_value, id_run, best_stats = ap.find_best(stats_out[1], metric='OF2DC', stats='min')
            _, _, self.df_calpem_sets = ap.find_best(stats_out[4], metric='OF2DC', stats='min')
        elif scale == 'monthly':
            best_value, id_run, best_stats = ap.find_best(stats_out[2], metric='OF2MC', stats='min')
            _, _, self.df_calpem_sets = ap.find_best(stats_out[4], metric='OF2MC', stats='min')
        else:
            best_value, id_run, best_stats = ap.find_best(stats_out[3], metric='OF2YC', stats='min')
            _, _, self.df_calpem_sets = ap.find_best(stats_out[4], metric='OF2YC', stats='min')
        df_pem_criteria = stats_out[0]
        df_pem_criteria.to_csv(f'{self.dir_sensitivity}/selected_Statistcs_runoff.csv')
        run_vec = df_pem_criteria.RunId
        df_stats_daily = stats_out[1]
        df_stats_monthly = stats_out[2]
        df_stats_yearly = stats_out[3]
        df_stats = pd.concat([df_stats_daily, df_stats_monthly, df_stats_yearly], axis=0)
        self.stats_citerion = df_stats
        self.best_stat = best_stats
        self.best_obj_value = best_value
        self.id_best = id_run

        f.writelines('Number of solutions: {:7d}'.format(df_pem_criteria.shape[0]))
        f.writelines('Best Objective function value: {:.4f}'.format(best_value))
        f.writelines('\nBest iterstion {:7d}'.format(id_run))
        f.writelines('\n---------------------------------------------------------------')
        f.writelines('\nModel performnce for the best set at calibration\n')
        f.writelines('\nCOD:       {:0.3f}'.format(best_stats[1]))
        f.writelines('\nRMSE:      {:0.3f}'.format(best_stats[2]))
        f.writelines('\nnRMSE:     {:0.3f}'.format(best_stats[3]))
        f.writelines('\nNSE:       {:0.3f}'.format(best_stats[4]))
        f.writelines('\nPBIAS:     {:0.3f}'.format(best_stats[5]))
        f.writelines('\nIOA:       {:0.3f}'.format(best_stats[6]))
        f.writelines('\nObjValue1: {:0.3f}'.format(best_stats[7]))
        f.writelines('\nObjValue2: {:0.3f}'.format(best_stats[8]))
        f.writelines('\nModel performnce for the best set at validation\n')
        f.writelines('\nCOD:       {:0.3f}'.format(best_stats[9]))
        f.writelines('\nRMSE:      {:0.3f}'.format(best_stats[10]))
        f.writelines('\nnRMSE:     {:0.3f}'.format(best_stats[11]))
        f.writelines('\nNSE:       {:0.3f}'.format(best_stats[12]))
        f.writelines('\nPBIAS:     {:0.3f}'.format(best_stats[13]))
        f.writelines('\nIOA:       {:0.3f}'.format(best_stats[14]))
        f.writelines('\nObjValue1: {:0.3f}'.format(best_stats[15]))
        f.writelines('\nObjValue2: {:0.3f}'.format(best_stats[16]))
        f.writelines('\n---------------------------------------------------------------')
        f.writelines('\nRange of model performnce for the runs within criteria')
        f.writelines('\nNotes: Last two three charecters refer to sets and time scales:')
        f.writelines('\n*AD: all daily set; *DC: daily set for calibration, *DV: daily sets for validation')
        f.writelines('\n*D: daily, *M: monthly, *Y: yearly, C: calibration, V: validation')
        f.writelines('\n---------------------------------------------------------------')
        for col in stats_out[4].columns[1:-2]:
            f.writelines(f'\n{col}: {(round(stats_out[4][col].min(), 3))}-{(round(stats_out[4][col].max(), 3))}')
        pbest = self.param[self.param['RunId'] == id_run]
        self.pbest = pbest
        f.writelines('\n---------------------------------------------------------------')
        f.writelines('\nBest model parameter set from calibration')
        for col in pbest.columns[1:]:
            f.writelines(f'\n{col}: {(round(pbest[col].min(), 8))}-{(round(pbest[col].max(), 8))}')
        df_param_criteria = ap.get_param_bests(self.file_parm, run_vec, scale)
        df_param_criteria.to_csv(f'{self.dir_sensitivity}/selected_APEXPARM.csv')
        f.writelines('\n---------------------------------------------------------------')
        f.writelines('\n Range of model parameters for the runs within criteria')
        f.writelines('\n---------------------------------------------------------------')
        for col in df_param_criteria.columns[1:]:
            f.writelines(
                f'\n{col}: {(round(df_param_criteria[col].min(), 8))}-{(round(df_param_criteria[col].max(), 8))}')
        f.close()
        return self

    def get_stats(self):
        df_pem = pd.read_csv(self.file_pem)
        df_pem.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
        self.stats = df_pem
        return self

    def read_params(self):
        df_param = pd.read_csv(self.file_parm)
        df_param.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
        self.param = df_param
        return self

    def get_params4ua(self, scale):
        criteria = [float(self.config['COD_criteria']), float(self.config['NSE_criteria']),
                    float(self.config['PBAIS_criteria'])]
        f = open(self.uncertainty_outflie, 'a')
        f.writelines('---------Calibration and validation resukt summary')
        f.writelines('\n---------------------------------------------------------------')
        f.writelines('Calibratiion criteria according to Moriasi et a. (20015)\n')
        f.writelines(f'Nash Sutcliffe efficiency, NSE     {criteria[1]}\n')
        f.writelines(f'Percent Bias, PBIAS:               {criteria[2]}\n')
        f.writelines(f'Coefficient of determination, COD: {criteria[0]}\n')
        f.writelines('\n---------------------------------------------------------------')
        stats_out = ap.compile_stats(self.stats, criteria, scale)
        if scale == 'daily':
            best_value, id_run, best_stats = ap.find_best(stats_out[1], metric='OF2DC', stats='min')
            _, _, self.df_calpem_sets = ap.find_best(stats_out[4], metric='OF2DC', stats='min')
            df_stats4ua = stats_out[1]
        elif scale == 'monthly':
            best_value, id_run, best_stats = ap.find_best(stats_out[2], metric='OF2MC', stats='min')
            _, _, self.df_calpem_sets = ap.find_best(stats_out[4], metric='OF2MC', stats='min')
            df_stats4ua = stats_out[2]
        else:
            best_value, id_run, best_stats = ap.find_best(stats_out[3], metric='OF2YC', stats='min')
            _, _, self.df_calpem_sets = ap.find_best(stats_out[4], metric='OF2YC', stats='min')
            df_stats4ua = stats_out[3]
        df_stats_daily = stats_out[1]
        df_stats_daily.columns = ['RunId', 'CODC', 'RMSEC', 'NRMSEC', 'NSEC', 'PBIASC',
                                  'IOAC', 'OF1C', 'OF2C', 'CODV', 'RMSEV', 'NRMSEV', 'NSEV',
                                  'PBIASV', 'IOAV', 'OF1V', 'OF2V']
        df_stats_daily['SCALE'] = 'daily'
        df_stats_monthly = stats_out[2]
        df_stats_monthly.columns = ['RunId', 'CODC', 'RMSEC', 'NRMSEC', 'NSEC', 'PBIASC',
                                    'IOAC', 'OF1C', 'OF2C', 'CODV', 'RMSEV', 'NRMSEV', 'NSEV',
                                    'PBIASV', 'IOAV', 'OF1V', 'OF2V']
        df_stats_monthly['SCALE'] = 'monthly'
        df_stats_yearly = stats_out[3]
        df_stats_yearly.columns = ['RunId', 'CODC', 'RMSEC', 'NRMSEC', 'NSEC', 'PBIASC',
                                   'IOAC', 'OF1C', 'OF2C', 'CODV', 'RMSEV', 'NRMSEV', 'NSEV',
                                   'PBIASV', 'IOAV', 'OF1V', 'OF2V']
        df_stats_yearly['SCALE'] = 'yearly'
        df_stats = pd.concat([df_stats_daily, df_stats_monthly, df_stats_yearly], axis=0)
        self.stats_citerion = df_stats
        self.best_stat = best_stats
        self.best_obj_value = best_value
        self.id_best = id_run
        f.writelines('Number of solutions: {:7d}'.format(df_stats4ua.shape[0]))
        f.writelines('Best Objective function value: {:.4f}'.format(best_value))
        f.writelines('\nBest iterstion {:7d}'.format(id_run))
        f.writelines('\n---------------------------------------------------------------')
        f.writelines('\nModel performnce for the best set at calibration\n')
        f.writelines('\nCOD:       {:0.3f}'.format(best_stats[1]))
        f.writelines('\nRMSE:      {:0.3f}'.format(best_stats[2]))
        f.writelines('\nnRMSE:     {:0.3f}'.format(best_stats[3]))
        f.writelines('\nNSE:       {:0.3f}'.format(best_stats[4]))
        f.writelines('\nPBIAS:     {:0.3f}'.format(best_stats[5]))
        f.writelines('\nIOA:       {:0.3f}'.format(best_stats[6]))
        f.writelines('\nObjValue1: {:0.3f}'.format(best_stats[7]))
        f.writelines('\nObjValue2: {:0.3f}'.format(best_stats[8]))
        f.writelines('\nModel performnce for the best set at validation\n')
        f.writelines('\nCOD:       {:0.3f}'.format(best_stats[9]))
        f.writelines('\nRMSE:      {:0.3f}'.format(best_stats[10]))
        f.writelines('\nnRMSE:     {:0.3f}'.format(best_stats[11]))
        f.writelines('\nNSE:       {:0.3f}'.format(best_stats[12]))
        f.writelines('\nPBIAS:     {:0.3f}'.format(best_stats[13]))
        f.writelines('\nIOA:       {:0.3f}'.format(best_stats[14]))
        f.writelines('\nObjValue1: {:0.3f}'.format(best_stats[15]))
        f.writelines('\nObjValue2: {:0.3f}'.format(best_stats[16]))
        f.writelines('\n---------------------------------------------------------------')
        f.writelines('\nRange of model performnce for the runs within criteria')
        f.writelines('\nNotes: Last two three charecters refer to sets and time scales:')
        f.writelines('\n*AD: all daily set; *DC: daily set for calibration, *DV: daily sets for validation')
        f.writelines('\n*D: daily, *M: monthly, *Y: yearly, C: calibration, V: validation')
        f.writelines('\n---------------------------------------------------------------')
        self.stats = df_stats4ua
        id_sets = df_stats4ua.RunId.values
        for col in stats_out[4].columns[1:-2]:
            f.writelines(f'\n{col}: {(round(stats_out[4][col].min(), 3))}-{(round(stats_out[4][col].max(), 3))}')
        pbest = self.param[self.param['RunId'] == id_run]
        self.pbest = pbest
        f.writelines('\n---------------------------------------------------------------')
        f.writelines('\nBest model parameter set from calibration')
        for col in pbest.columns[1:]:
            f.writelines(f'\n{col}: {(round(pbest[col].min(), 8))}-{(round(pbest[col].max(), 8))}')
        df_ua_params = ap.get_param_bests(self.file_parm, id_sets, scale)
        df_ua_params.to_csv(f'{self.dir_uncertainty}/selected_APEXPARM.csv')
        f.writelines('\n---------------------------------------------------------------')
        f.writelines('\n Range of model parameters for the runs within criteria')
        f.writelines('\n---------------------------------------------------------------')
        for col in df_ua_params.columns[1:]:
            f.writelines(f'\n{col}: {(round(df_ua_params[col].min(), 8))}-{(round(df_ua_params[col].max(), 8))}')
        f.close()
        self.ua_params = df_ua_params
        return self

    def get_range(self):
        # import csv file containing range of parameters with recommended values for specific project
        df_param_limit = pd.read_csv(self.file_limits, index_col=0, encoding="ISO-8859-1")
        self.param_discription = df_param_limit.columns.to_list()
        self.param_list = df_param_limit.iloc[0, :].to_list()
        array_param_list = df_param_limit.iloc[1:, :].to_numpy()
        self.array_param_list = array_param_list.astype(np.float64)
        mat_param_list = np.asmatrix(array_param_list)
        self.mat_param_list = np.squeeze(np.asarray(mat_param_list))
        return self

    def generate_param_set(self, n_sim, isall):
        self.recc_params = self.mat_param_list[2, :]
        if isall:
            minp = self.mat_param_list[0, :]
            self.min_params = np.array([v.replace(',', '') for v in minp], dtype=np.float64)
            maxp = self.mat_param_list[1, :]
            self.max_params = np.array([v.replace(',', '') for v in maxp], dtype=np.float64)
            n_params = len(self.max_params)
        else:
            id_sensitive = read_sensitive_params(self.src_dir)
            mat_sensitive_limt = self.mat_param_list[:, id_sensitive]
            minp = mat_sensitive_limt[0, :]
            self.min_params = np.array([v.replace(',', '') for v in minp], dtype=np.float64)
            maxp = mat_sensitive_limt[1, :]
            self.max_params = np.array([v.replace(',', '') for v in maxp], dtype=np.float64)
            recc_sensitive = mat_sensitive_limt[2, :]
            n_params = len(recc_sensitive)

        diff = self.max_params - self.min_params
        inc = diff / n_sim
        mat_params = np.zeros((n_sim + 1, n_params))
        if isall:
            mat_params[0, :] = self.recc_params
        else:
            mat_params[0, :] = recc_sensitive

        for i in range(n_sim):
            for j in range(n_params):
                mat_params[i + 1, j] = self.min_params[j] + i * inc[j]
        self.parameters_matrix = mat_params
        return self

    def pick_param(self, i, allparam=True):
        nset, nparam = self.parameters_matrix.shape
        p = np.zeros((nparam))
        random.seed(i)
        id_rands = ep.ranom_intmatrix(nparam, 1, limit=(0, nset - 1))
        for ip in range(nparam):
            p[ip] = self.parameters_matrix[int(id_rands[ip]), ip]
        if allparam:
            p = p
        else:
            p_all = self.recc_params
            id_sensitive = read_sensitive_params(self.src_dir)
            p_all[id_sensitive] = p
            p = p_all
        self.p = p
        return self.p

    def generate_sensitive_params(self, isall_try, p=None):
        maxp = int(self.config['max_range'])
        inc = float(self.config['increment'])
        minp = -maxp
        deltas = np.arange(minp, maxp + 1, inc)
        nset = len(deltas)
        p = self.pbest.T.values[1:].ravel()
        if p is None:
            recc_params = np.array(self.mat_param_list[2, :])
        else:
            recc_params = p
        self.recc_params = recc_params
        lb_params = self.mat_param_list[0, :].astype(float)
        ub_params = self.mat_param_list[1, :].astype(float)
        diff = ub_params - lb_params
        nparams = len(recc_params)
        if isall_try:
            # Changing all the parameters with in the range at once
            mat_params = ep.nanmatrix(nset, nparams)
            for i in range(nparams):
                if i <= 69:
                    mat_params[:, i] = recc_params[i]
                else:
                    for j in range(nset):
                        p_sen = recc_params[i] + diff[i] * deltas[j] * 0.01
                        if p_sen <= lb_params[i]:
                            p_sen = lb_params[i]
                        elif p_sen >= ub_params[i]:
                            p_sen = ub_params[i]
                        else:
                            p_sen = p_sen
                        mat_params[j, i] = p_sen
            # Increasing or decreasing all the parameters individually
            id_params = np.arange(70, nparams)[:-4]
            for id_ in id_params:
                mat_id = ep.nanmatrix(nset, nparams)
                mat_id[:, :] = recc_params
                p_vec = []
                for j in range(nset):
                    p_sen = recc_params[id_] + diff[id_] * deltas * 0.01
                    if p_sen <= lb_params[id_]:
                        p_sen = lb_params[id_]
                    elif p_sen >= ub_params[id_]:
                        p_sen = ub_params[id_]
                    else:
                        p_sen = p_sen
                    p_vec.append(p_sen)
                mat_id[:, id_] = p_vec
                mat_params = np.concatenate((mat_params, mat_id), axis=0)
                del mat_id
        else:
            mat_params = ep.nanmatrix(nset, nparams)
            idsensitive = read_sensitive_params(self.src_dir)
            recc_params = p
            mat_params[:, :] = recc_params
            min_params = lb_params[idsensitive]
            max_params = ub_params[idsensitive]
            diff = max_params - min_params
            # Increasing or decreasing all the selected parameters with in the range at once
            for i in range(len(idsensitive)):
                for j in range(nset):
                    p_sen = recc_params[idsensitive[i]] + diff[i] * deltas[j] * 0.01
                    if p_sen <= min_params[i]:
                        p_sen = min_params[i]
                    elif p_sen >= max_params[i]:
                        p_sen = max_params[i]
                    else:
                        p_sen = p_sen
                    mat_params[j, idsensitive[i]] = p_sen
            del i
            # Increasing or decreasing all the selected parameters individually
            for i in range(len(idsensitive)):
                mat_id = ep.nanmatrix(nset, nparams)
                mat_id[:, :] = recc_params
                p_vec = []
                for j in range(nset):
                    p_sen = recc_params[idsensitive[i]] + diff[i] * deltas[j] * 0.01
                    if p_sen <= min_params[i]:
                        p_sen = min_params[i]
                    elif p_sen >= max_params[i]:
                        p_sen = max_params[i]
                    else:
                        p_sen = p_sen
                    p_vec.append(p_sen)
                mat_id[:, idsensitive[i]] = p_vec
                mat_params = np.concatenate((mat_params, mat_id), axis=0)
                del mat_id
            self.parameters_matrix = mat_params
            # from numpy import savetxt
            # file_text = os.path.join(os.path.dirname(self.senstitivty_out_file), 'APEXPARM_SA.txt')
            # savetxt(file_text, mat_params, delimiter=',')
        return self

    def generate_uncertaintity_params(self, isall_try):
        maxp = int(self.config['max_range_uncertaintity'])
        inc = float(self.config['increment_uncertainty'])
        minp = -maxp
        deltas = np.arange(minp, maxp + inc, inc)
        nset = len(deltas)
        p = self.pbest.T.values[1:].ravel()
        self.recc_params = p
        df_ua_params = self.ua_params
        ua_params = df_ua_params.iloc[:, 1:]
        mu_vec = ua_params.mean(axis=0).values
        std_vec = ua_params.std(axis=0).values
        lb_params = self.mat_param_list[0, :].astype(float)
        ub_params = self.mat_param_list[1, :].astype(float)
        nparams = len(mu_vec)
        if isall_try:
            # Changing all the parameters with in the range at once
            mat_params = ep.nanmatrix(nset, nparams)
            for i in range(nparams):
                if i <= 69:
                    mat_params[:, i] = mu_vec[i]
                else:
                    for j in range(nset):
                        p_un = mu_vec[i] + std_vec[i] * deltas[j]
                        if p_un <= lb_params[i]:
                            p_un = lb_params[i]
                        elif p_un >= ub_params[i]:
                            p_un = ub_params[i]
                        else:
                            p_un = p_un
                        mat_params[j, i] = p_un
        else:
            mat_params = ep.nanmatrix(nset, nparams)
            idsensitive = read_sensitive_params(self.src_dir)
            mat_params[:, :] = p
            min_params = lb_params[idsensitive]
            max_params = ub_params[idsensitive]
            # Increasing or decreasing all the selected parameters with in the range at once
            for i in range(len(idsensitive)):
                for j in range(nset):
                    p_un = mu_vec[idsensitive[i]] + std_vec[idsensitive[i]] * deltas[j]
                    if p_un <= min_params[i]:
                        p_un = min_params[i]
                    elif p_un >= max_params[i]:
                        p_un = max_params[i]
                    else:
                        p_un = p_un
                    mat_params[j, idsensitive[i]] = p_un
        self.parameters_matrix = mat_params
        return self

    def assign_output_df(self):
        self.df_WYLD, self.df_LAI, self.df_BIOM = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.df_PRCP, self.df_ET, self.df_PET = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.df_YSD, self.df_USLE = pd.DataFrame(), pd.DataFrame()
        self.df_MUSL, self.df_REMX = pd.DataFrame(), pd.DataFrame()
        self.df_MUSS, self.df_MUST = pd.DataFrame(), pd.DataFrame()
        self.df_RUS2, self.df_RUSL = pd.DataFrame(), pd.DataFrame()
        self.df_STL, self.df_STD, self.df_STDL = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        self.df_YN, self.df_YP = pd.DataFrame(), pd.DataFrame()
        self.df_QN, self.df_QP = pd.DataFrame(), pd.DataFrame()
        self.df_QDRN, self.df_QDRP = pd.DataFrame(), pd.DataFrame()
        self.df_QRFN, self.df_QRFP = pd.DataFrame(), pd.DataFrame()
        self.df_SSFN, self.df_RSFN = pd.DataFrame(), pd.DataFrame()
        self.df_TN, self.df_TP = pd.DataFrame(), pd.DataFrame()
        self.df_DPRK = pd.DataFrame()
        self.df_YLDG, self.df_YLDF = pd.DataFrame(), pd.DataFrame()
        self.df_WS, self.df_TS = pd.DataFrame(), pd.DataFrame()
        self.df_NS, self.df_PS = pd.DataFrame(), pd.DataFrame()
        return self
