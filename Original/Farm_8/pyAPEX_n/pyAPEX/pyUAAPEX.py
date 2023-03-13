# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 09:00:28 2022

@author: Mahesh.Maskey
"""


import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import warnings
from configobj import ConfigObj
from pathlib import Path
from Utility.apex_utility import read_sensitive_params, print_progress_bar
from Utility.pyAPEXpost import pyAPEXpost as ap
from Utility.apex_utility import split_data
print('\014')

class unaAPEX:
    def __init__(self, src_dir, config, cluster_dir, site, scenario, scale, obs_attribute, mod_attribute, out_dir, isfull=True):
        
        """  
        
 

        """
        self.config = config
        self.obs_attribute = obs_attribute
        self.mod_attribute = mod_attribute
        self.site = site
        self.scenario = scenario
        self.src_dir = src_dir 
        self.criteria = [float(config['COD_criteria']), float(config['NSE_criteria']), float(config['PBAIS_criteria'])]
        self.cluster_dir = cluster_dir 
        self.metrics = ['OF', 'NSE', 'PBIAS', 'COD'] 
        self.cal_dir = config['dir_calibration']
        self.una_dir = config['dir_uncertainty']
        self.out_dir = out_dir 
        self.site = site
        self.get_pe_files()  
        self.get_params()
        self.assign_read_dir()
        self.read_pem(scale)      
        self.df_obs = ap.get_measure (data_dir = 'Program', file_name = 'calibration_data.csv')
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir) 
        self.outlet_daily = self.import_best(location='daily_outlet', crops=None)
        self.basin_daily = self.import_best(location='daily_basin', crops=None)
        self.annual = self.import_best(location='annual', crops=None)
        self.compile_stats_param(scale)

        # import major output
        df_WYLD = self.import_output(location='outlet', variable='WYLD', isfull=isfull)
        df_WYLD.to_csv(f'{out_dir}/Daily_WYLD.csv')  
        df_RUS2 = self.import_output(location='outlet', variable='RUS2', isfull=isfull)
        df_RUS2.to_csv(f'{out_dir}/Daily_RUS2.csv')          
        df_YSD = self.import_output(location='outlet', variable='YSD', isfull=isfull)
        df_YSD.to_csv(f'{out_dir}/Daily_YSD.csv')          
        df_LAI = self.import_output(location='basin', variable='LAI', isfull=isfull)
        df_LAI.to_csv(f'{out_dir}/Daily_LAI.csv') 
        df_BIOM = self.import_output(location='basin', variable='BIOM', isfull=isfull)
        df_BIOM.to_csv(f'{out_dir}/Daily_BIOM.csv') 
        df_PRCP = self.import_output(location='basin', variable='PRCP', isfull=isfull)
        df_PRCP.to_csv(f'{out_dir}/Daily_PRCP.csv') 
        df_STL = self.import_output(location='basin', variable='STL', isfull=isfull)
        df_STL.to_csv(f'{out_dir}/Daily_STL.csv') 
        df_STD = self.import_output(location='basin', variable='STD', isfull=isfull)
        df_STD.to_csv(f'{out_dir}/Daily_STD.csv') 
        df_STDL = self.import_output(location='basin', variable='STDL', isfull=isfull)
        df_STDL.to_csv(f'{out_dir}/Daily_STDL.csv')          
        df_ET = self.import_output(location='basin', variable='ET', isfull=isfull)
        df_ET.to_csv(f'{out_dir}/Daily_ET.csv')
        df_PET = self.import_output(location='basin', variable='PET', isfull=isfull)
        df_PET.to_csv(f'{out_dir}/Daily_PET.csv') 
        df_DPRK = self.import_output(location='basin', variable='DPRK', isfull=isfull)
        df_DPRK.to_csv(f'{out_dir}/Daily_DPRK.csv')       
        dfTP = self.import_output(location='basin', variable='TP', isfull=isfull)
        dfTP.to_csv(f'{out_dir}/Daily_TP.csv') 
        df_TN = self.import_output(location='basin', variable='TN', isfull=isfull)
        df_TN.to_csv(f'{out_dir}/Daily_TN.csv')         
        df_YLDG = self.import_output(location='annual', variable='YLDG', isfull=isfull)
        df_YLDG.to_csv(f'{out_dir}/Annual_YLDG.csv') 
        df_YLDF = self.import_output(location='annual', variable='YLDF', isfull=isfull)
        df_YLDF.to_csv(f'{out_dir}/Annual_YLDF.csv')         
        df_BIOM = self.import_output(location='annual', variable='BIOM', isfull=isfull)
        df_BIOM.to_csv(f'{out_dir}/Annual_BIOM.csv') 
        df_WS = self.import_output(location='annual', variable='WS', isfull=isfull)
        df_WS.to_csv(f'{out_dir}/Annual_WS.csv')    
        df_NS = self.import_output(location='annual', variable='NS', isfull=isfull)
        df_NS.to_csv(f'{out_dir}/Annual_NS.csv') 
        df_PS = self.import_output(location='annual', variable='PS', isfull=isfull)
        df_PS.to_csv(f'{out_dir}/Annual_PS.csv')
        df_TS = self.import_output(location='annual', variable='TS', isfull=isfull)
        df_TS.to_csv(f'{out_dir}/Annual_TS.csv')            
 
        
    def get_pe_files(self):
        if (self.mod_attribute=='WYLD'):
            file_pe = os.path.join(self.una_dir, 'Statistics_runoff.csv')
        elif (self.mod_attribute=='YSD'):
            file_pe = os.path.join(self.una_dir, 'Statistics_sediment.csv')
        else:
            file_pe = os.path.join(self.una_dir, f'Statistics_{self.attribute}.csv')
        file_parameter = os.path.join(self.una_dir, 'APEXPARM.csv')
        self.file_pem = file_pe
        self.file_param = file_parameter
        return self        

    def get_params(self):
        df_params = pd.read_csv(self.file_param)
        df_params.rename(columns = {'Unnamed: 0':'RunId'}, inplace = True)
        self.param_list =df_params.columns
        self.params_all = df_params            
        self.pbest = df_params.iloc[0,:]
        self.params = df_params.iloc[1:, ]
        return self    

    def read_pem(self, scale):
        # read statistics
        df_pem =  pd.read_csv(self.file_pem)
        df_pem.rename(columns = {'Unnamed: 0':'RunId'}, inplace = True)
        id_best = int(df_pem.iloc[0, 0])
        df_pem = df_pem.iloc[1:, :]
        pebest = df_pem.iloc[0, :]
        if scale=='daily':
            COD, NSE, PBAIS = 'CODDC', 'NSEDC', 'APBIASDC'
        elif scale=='monthly':
            COD, NSE, PBAIS = 'CODMC', 'NSEMC', 'APBIASMC'
        elif scale=='yearly':
            COD, NSE, PBAIS = 'CODYC', 'NSEYC', 'APBIASYC'
        df_pem_within = df_pem[(df_pem[COD]>self.criteria[0])&(df_pem[NSE]>self.criteria[1])&(df_pem[PBAIS]<self.criteria[2])]        
        self.pem = df_pem
        self.id_best = id_best
        self.pebest = pebest
        self.pem4criteria = df_pem_within
        return self  

    def assign_read_dir(self):
        if self.scenario=='non_grazing':
            scenario1 = 'pyAPEX_n'
        else: 
            scenario1 = 'pyAPEX_g'
        self.read_dir = os.path.join(self.cluster_dir, 
                                     self.site, scenario1, 'Output')
        return self    
    
    def import_best(self, location, crops):
        run_id = self.id_best
        if crops==None:
            file_read = f'{location}_{run_id:07}.csv'
            file_path = os.path.join(self.read_dir, file_read)
            df_data = pd.read_csv(file_path)
            if location== 'annual':
               df_data.index = df_data.YR 
               df_data = df_data.drop(['Unnamed: 0', 'YR'], axis=1)
            else:
                df_data.Date = pd.to_datetime(df_data.Date)
                df_data.index = df_data.Date
                df_data = df_data.drop('Date', axis=1)
            df_data.to_csv(os.path.join(self.out_dir, f'{location}_best.csv'))
            df_data.to_csv(os.path.join(self.out_dir, file_read))
            return df_data
        else:
            file_read = f'{location}_{run_id:07}.csv'
            file_path = os.path.join(self.read_dir, file_read)
            df_data = pd.read_csv(file_path)
            if location== 'annual':
               df_data.index = df_data.YR 
               df_data = df_data.drop(['Unnamed: 0', 'YR'], axis=1)
            else:
                df_data.Date = pd.to_datetime(df_data.Date)
                df_data.index = df_data.Date
                df_data = df_data.drop('Date', axis=1)
            df_data.to_csv(os.path.join(self.out_dir, f'{location}_best.csv'))
            df_data.to_csv(os.path.join(self.out_dir, file_read))
            df_list = [df_data]
            for crop in crops:
                file_read = f'{location}_{run_id:07}_{crop}.csv'
                file_path = os.path.join(self.read_dir, file_read)
                df_data = pd.read_csv(file_path)
                df_data.index = df_data.Date
                df_data = df_data.drop('Date', axis=1)
                df_data.to_csv(os.path.join(self.out_dir, f'{location}_best.csv'))
                df_data.to_csv(os.path.join(self.out_dir, file_read))
                df_list.append(df_data)
            return df_list  
        
    def compile_stats_param(self, scale):
        out_compile = ap.compile_stats(self.pem, self.criteria, scale)
        self.df_stats_daily = out_compile[0]
        self.df_daily_daily = out_compile[1]
        self.df_daily_monthly = out_compile[2]
        self.df_daily_yearly = out_compile[3]
        self.ids_best_daily = self.df_stats_daily.RunId.values.astype(int)
        self.df_best_params = pd.DataFrame()
        for i in range(len(self.ids_best_daily)):
            df = self.params[self.params['RunId']==self.ids_best_daily[i]]
            self.df_best_params = pd.concat([self.df_best_params, df], axis=0)     
        return self   
        
    def assign_dates(self, df_mod, n_warm, n_calib_year):
        start_year = df_mod.Year.values[0]
        cal_start = start_year + n_warm
        cal_end = cal_start + n_calib_year
        val_end = self.df_obs.Year[-1]
        val_end_date = self.df_obs.Date[-1]
        return start_year, cal_start, cal_end, val_end, val_end_date
        
    def import_output(self, location, variable, isfull):
        print (f'\nExporting {variable} data')
        if isfull:
            n_runs = self.params.shape[0]
        else:
            n_runs = self.df_best_params.shape[0]
        n_warm =  int(self.config['warm_years'])
        n_calib_year = int(self.config['calib_years'])
        df_out = pd.DataFrame()                   
        print_progress_bar(0, n_runs, prefix='Progress', suffix='Complete', length=50, fill='█')
        for k in range(n_runs):
            if isfull:
                i = k+1
            else:
                i = self.ids_best_daily[k]
            try:
                if location=='outlet':
                    file_outlet =os.path.join(self.una_dir, f'daily_outlet_{i:07}.csv.csv')
                    data = pd.read_csv(file_outlet)
                    data.index = data.Date
                    data.index = pd.to_datetime(data.index)             
                    data=data.drop(['Date', 'Y', 'M', 'D'], axis=1)
                    data.insert(0, 'Year', data.index.year, True)
                    start_year, cal_start, cal_end, val_end, val_end_date =  self.assign_dates(data, n_warm, n_calib_year)
                    df_data_cal, df_data_val = split_data(start_year, data, n_warm, n_calib_year)
                    df_data_val = df_data_val[df_data_val.index<=val_end_date]
                elif location=='basin':
                    file_basin =os.path.join(self.una_dir, f'daily_basin_{i:07}.csv.csv')
                    data = pd.read_csv(file_basin)
                    data.index = data.Date
                    data.index = pd.to_datetime(data.index)
                    data=data.drop(['Date', 'Y', 'M', 'D'], axis=1)
                    data.insert(0, 'Year', data.index.year, True)
                    start_year, cal_start, cal_end, val_end, val_end_date =  self.assign_dates(data, n_warm, n_calib_year)
                    df_data_cal, df_data_val = split_data(start_year, data, n_warm, n_calib_year)                
                    df_data_val = df_data_val[df_data_val.index<=val_end_date]
                else:
                    file_annual =os.path.join(self.una_dir, f'annual_{i:07}.csv.csv')
                    data = pd.read_csv(file_annual)        
                    data.index = data.YR
                    data=data.drop(['Unnamed: 0', 'YR'], axis=1)
                    data.insert(0, 'Year', data.index, True)
                    start_year, cal_start, cal_end, val_end, val_end_date =  self.assign_dates(data, n_warm, n_calib_year)
                    df_data_cal, df_data_val = split_data(start_year, data, n_warm, n_calib_year)
                    df_data_val = df_data_val[df_data_val.index<=val_end]
                df_data_cal.insert(df_data_cal.shape[1], 'Stage', 'Calibration', True)
                df_data_val.insert(df_data_val.shape[1], 'Stage', 'Validation', True)
            except Exception as e:
                print(e)
                print(f'error occurs in simulation {i}')
                continue 
            df_data = pd.concat([df_data_cal, df_data_val], axis=0)
            stage_vec = df_data.Stage.values
            df_data = pd.DataFrame(df_data[variable])
            df_data.columns = [f'{i}']
            df_out = pd.concat([df_out, df_data], axis=1)
           
            print_progress_bar(k, n_runs, prefix=f'Progress: {i}', suffix='Complete', length=50, fill='█')
        df_out.insert(0, 'Stage', stage_vec, True)            
        return df_out        