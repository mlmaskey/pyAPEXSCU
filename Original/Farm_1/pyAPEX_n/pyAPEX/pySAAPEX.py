# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:44:44 2022

@author: Mahesh.Maskey
"""


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from configobj import ConfigObj
import seaborn as sns
from Utility.pyAPEXpost import print_progress_bar
from Utility.pyAPEXpost import pyAPEXpost as ap
from Utility.apex_utility import read_sensitive_params
from Utility.apex_utility import split_data
from Utility.easypy import easypy as ep 
warnings.filterwarnings('ignore') 
config = ConfigObj('runtime.ini')
print('\014')

class senanaAPEX:
    def __init__(self, src_dir, config, out_dir, attribute,  metric='OF'):
        
        """  
        
 

        """
        
        self.config = config
        self.src_dir = src_dir 
        self.file_limits = self.src_dir / 'Utility' / self.config['file_limits']   
        self.attribute = attribute
        self.folder = config['dir_sensitivity']
        self.get_range ()      
        self.df_obs = ap.get_measure (data_dir = 'Program', file_name = 'calibration_data.csv')
        id_sensitive = read_sensitive_params(self.src_dir)
        for i in range(len(id_sensitive)):
            id_sensitive[i] = id_sensitive[i]+1
        param_range = self.param_range[:, id_sensitive]
        self.list_bound = list(param_range[:2, :].T)
        self.get_pe_files()  
        self.get_params(id_sensitive, isall=False)
        metric_list = ['RunId', 'CODDC', 'RMSEDC', 'NRMSEDC','NSEDC', 'PBIASDC', 'OF2DC']
        self.read_pem(metric_list)
        self.params.insert(self.params.shape[1], 'PARAM', 'all')
        self.pem4criteria.insert(self.pem4criteria.shape[1], 'PARAM', 'all')
        df_pem = self.pem4criteria
        df_params = self.params
        site = config['Site']
        scenario = config['Scenario']
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        # reading criteria set in config   
        COD = float(config['COD_criteria'])
        NSE = float(config['NSE_criteria'])
        PBIAS = float(config['PBAIS_criteria'])
        
        df_sen_summary = pd.DataFrame()
        df_sens = pd.DataFrame()
        self.metric = metric
        maxp = int(config['max_range'])
        inc = float(config['increment'])
        minp = -maxp
        deltas = np.arange(minp, maxp, inc)
        
       
    
        nparam = len(self.param_list)
        nstep = len(deltas)
        n_simul = len(deltas)*(nparam+1)
        id_vec = np.arange(nstep, n_simul,nstep)
        for i in range(len(id_vec)):
            df_pem.PARAM[id_vec[i]:id_vec[i]+nstep]=self.param_list[i]
            df_params.PARAM[id_vec[i]:id_vec[i]+nstep]=self.param_list[i]                   
        df_count = pd.DataFrame()                                              
        for metric in ['OF', 'NSE', 'PBIAS', 'COD']:
            # calculate standard deviation
            df_x = df_params.copy()
            df_x = df_x[df_x.PARAM=='all']
            df_x = df_x.drop(['RunId', 'PARAM'], axis=1)
            df_y = df_pem[df_pem.PARAM=='all']
            df_y = df_y[metric]
            src_total = ap.standarizedRegressionCoefficientTotal(df_x, df_y, intercept=False) 
            df_param_mat = pd.DataFrame()
            df_metric = pd.DataFrame()
            df_src = pd.DataFrame()
            df_pem_combined = df_pem[df_pem.PARAM=='all']
            df_pem_criteria = df_pem_combined[(df_pem_combined.COD>COD)&(df_pem_combined.NSE>NSE)&(df_pem_combined.PBIAS<PBIAS)]
            dfcount1 = pd.DataFrame({'COD': len(df_pem_combined.COD>=COD), 'NSE': len(df_pem_combined.NSE>=NSE), 
                                    'PBIAS':len(df_pem_combined.PBIAS<=PBIAS),'TOTAL':df_pem_criteria.shape[0],
                                    'METRIC': metric}, index=['all'])            
            for j in range(len(self.param_list)):
                param=self.param_list[j]
                param_name = f'PARAM [{id_sensitive[j]-70}]'
                df_pm_j = df_pem[df_pem.PARAM==param]
                df_pem_criteria = df_pm_j[(df_pm_j.COD>COD)&(df_pm_j.NSE>NSE)&(df_pm_j.PBIAS<PBIAS)]
                dfcount = pd.DataFrame({'COD': len(df_pm_j.COD>=COD), 'NSE': len(df_pm_j.NSE>=NSE), 
                                        'PBIAS':len(df_pm_j.PBIAS<=PBIAS), 'TOTAL':df_pem_criteria.shape[0],
                                        'METRIC': metric}, index=[param_name])
                df_count = pd.concat([df_count, dfcount], axis=0) 
                df_pa_j = df_params[df_params.PARAM==param]
                df_param_indv = pd.DataFrame(df_pa_j[param].values)
                df_param_mat = pd.concat([df_param_mat,df_param_indv], axis=1)
                df_metric_indv = pd.DataFrame(df_pm_j[metric].values)
                df_metric = pd.concat([df_metric,df_metric_indv], axis=1)
                df_sen = pd.DataFrame({'PARAM': df_pa_j[self.param_list[j]], 
                                       'METRIC': df_pm_j[metric]})
                df_sen.METRIC = ep.get_outlier_na(df_sen.METRIC)
                df_sen.insert(1, 'PARAM_CHANGE', ep.find_change_percent(df_sen.PARAM.values, self.pbest[param]))
                df_sen.insert(3, 'CHANGE_METRIC', ep.find_change_percent(df_sen.METRIC.values, self.pebest[metric]))
                df_sen.insert(4, 'ABSOLUTE_CHANGE', deltas)
                df_sen.insert(5, 'PARAM_NAME', param)
                df_sen.insert(6, 'NAME', param_name)
                df_sen_non_nan = df_sen.dropna()
                r2, p_value = ep.corr_test(df_sen_non_nan.PARAM_CHANGE, df_sen_non_nan.CHANGE_METRIC)
                x = df_params[df_params.PARAM==param][param].values
                y = df_pem[df_pem.PARAM==param][metric].values
                Si = ap.sensitivity_index(x, y)
                df_j = pd.DataFrame({'Nmet': df_pem_criteria.shape[0], 
                                     'minCOD': df_pem_criteria.COD.min(), 
                                     'maxCOD': df_pem_criteria.COD.max(), 
                                     'NCOD': len(df_pem_criteria.COD>COD),
                                     'minNSE': df_pem_criteria.NSE.min(), 
                                     'maxNSE': df_pem_criteria.NSE.max(), 
                                     'NNSE': len(df_pem_criteria.NSE>NSE),
                                     'minPBIAS': df_pem_criteria.PBIAS.min(), 
                                     'maxPBIAS': df_pem_criteria.PBIAS.max(), 
                                     'NPBIAS': len(df_pem_criteria.PBIAS>PBIAS),
                                     'MINp': df_pa_j[self.param_list[j]].min(),
                                     'MAXp': df_pa_j[self.param_list[j]].max(),
                                     'Rsquared': r2,
                                     'p-value':p_value,
                                     'SenstivityIndex': Si},
                                     index= [param_name])
                #calculate individual stadraized Regression coefficient
                src = ap.standarizedRegressionCoefficient(df_pa_j[param].values, df_pm_j[metric].values, intercept=True)
                dfsrc = pd.DataFrame({'SRC': src, 'Metric': metric}, index= [param_name])
                df_src= pd.concat([df_src,dfsrc], axis=0)
                df_j['PARAM'] = param
                df_sen_summary = pd.concat([df_sen_summary, df_j], axis=0)
                df_sens = pd.concat([df_sens, df_sen_non_nan], axis=0)
            df_count = pd.concat([df_count, dfcount1], axis=0) 
            df_param_mat.columns = self.param_list
            df_param_all= pd.concat([df_params[df_params.PARAM=='all'][self.param_list],df_param_mat], axis=0)
            df_metric.columns = self.param_list
            df_count.to_csv(f'{out_dir}/Param_within_{metric}.csv')
            # prepare data for heat map
            param_list=[]
            for i, param in enumerate (self.param_list):
                param_name = f'PARAM [{id_sensitive[i]-70}]'
                param_list.append(param_name)
            
            if metric=='OF':
                cbartitle = f'Percent change in objective function value'
            else:
                cbartitle = f'Percent change in {metric} value'        
        
            df_plotdata = (df_metric - self.pebest[metric])*100/self.pebest[metric]
            df_plotdata.columns = param_list
            df_plotdata.index=[round(item, 2) for item in deltas]
            df_plotdata.to_csv(f'{out_dir}/Heatmapdata_{metric}.csv')
            self.Heatmapdata_ = df_plotdata
            fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)            
            ax = sns.heatmap(df_plotdata, cmap='bwr',
                             vmin=df_plotdata.min().min(), 
                             vmax= df_plotdata.max().max(),
                             cbar_kws={'label': cbartitle})
            ax.set_ylabel('Percent change in parameters')
            plt.savefig(f'{out_dir}/Heatmap_{metric}.png', dpi=600, bbox_inches="tight") 
                                            
            # Sensitivitiy indices
            df_src.insert(1, 'SRC_Total', src_total.values, True)
             
            Y = df_pem[metric].values
            list_bound = list(df_sen_summary[['MINp', 'MAXp']])
            
            df_SI_First, df_SI_total =  ap.getSOBOL_FASTindex(param_list, list_bound, Y)
            df_SRC_First = pd.DataFrame({'Senstivity Index': df_src.SRC.values, 'Order': 'First', 'Method': 'SRC', 'PARAM':param_list}, index=param_list)
            df_SRC_total = pd.DataFrame({'Senstivity Index': df_src.SRC_Total.values, 'Order': 'Total', 'Method': 'SRC', 'PARAM':param_list}, index=param_list)
            df_SI_First= pd.concat([df_SI_First,df_SRC_First], axis=0)
            df_SI_total= pd.concat([df_SI_total,df_SRC_total], axis=0)
            fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=False, sharey=True, tight_layout=True)
            sns.barplot(data=df_SI_First, y ='PARAM', x= 'Senstivity Index', hue='Method', ci=None, ax=axes[0])
            # handles, labels = ax1.get_legend_handles_labels()
            axes[0].grid(True)
            axes[0].set_ylabel('Parameters')
            axes[0].set_title('First order sensitivity indeex')
            sns.barplot(data=df_SI_total, y ='PARAM', x= 'Senstivity Index', hue='Method', ci=None, ax=axes[1])
            # handles, labels = ax1.get_legend_handles_labels()
            axes[1].grid(True)
            axes[1].set_ylabel('Parameters')
            axes[1].set_title('Total sensitivity indeex')
            plt.savefig(f'{out_dir}/Index_{metric}.png', dpi=600, bbox_inches="tight")        
            df_param_all.to_csv(f'{out_dir}/Param_{metric}.csv')
            df_SI_First.to_csv(f'{out_dir}/Index_first_{metric}.csv')
            df_SI_total.to_csv(f'{out_dir}/Index_Total_{metric}.csv')
            df_sen_summary.to_csv(f'{out_dir}/Stat_Summary_{metric}.csv')

        # import major output
        df_WYLD = self.import_output(location='outlet', variable='WYLD')
        df_WYLD.to_csv(f'{out_dir}/Daily_WYLD.csv')  
        df_RUS2 = self.import_output(location='outlet', variable='RUS2')
        df_RUS2.to_csv(f'{out_dir}/Daily_RUS2.csv')          
        df_YSD = self.import_output(location='outlet', variable='YSD')
        df_YSD.to_csv(f'{out_dir}/Daily_YSD.csv')          
        df_LAI = self.import_output(location='basin', variable='LAI')
        df_LAI.to_csv(f'{out_dir}/Daily_LAI.csv') 
        df_BIOM = self.import_output(location='basin', variable='BIOM')
        df_BIOM.to_csv(f'{out_dir}/Daily_BIOM.csv') 
        df_PRCP = self.import_output(location='basin', variable='PRCP')
        df_PRCP.to_csv(f'{out_dir}/Daily_PRCP.csv') 
        df_STL = self.import_output(location='basin', variable='STL')
        df_STL.to_csv(f'{out_dir}/Daily_STL.csv') 
        df_STD = self.import_output(location='basin', variable='STD')
        df_STD.to_csv(f'{out_dir}/Daily_STD.csv') 
        df_STDL = self.import_output(location='basin', variable='STDL')
        df_STDL.to_csv(f'{out_dir}/Daily_STDL.csv')          
        df_ET = self.import_output(location='basin', variable='ET')
        df_ET.to_csv(f'{out_dir}/Daily_ET.csv')
        df_PET = self.import_output(location='basin', variable='PET')
        df_PET.to_csv(f'{out_dir}/Daily_PET.csv') 
        df_DPRK = self.import_output(location='basin', variable='DPRK')
        df_DPRK.to_csv(f'{out_dir}/Daily_DPRK.csv')       
        dfTP = self.import_output(location='basin', variable='TP')
        dfTP.to_csv(f'{out_dir}/Daily_TP.csv') 
        df_TN = self.import_output(location='basin', variable='TN')
        df_TN.to_csv(f'{out_dir}/Daily_TN.csv')         
        df_YLDG = self.import_output(location='annual', variable='YLDG')
        df_YLDG.to_csv(f'{out_dir}/Annual_YLDG.csv') 
        df_YLDF = self.import_output(location='annual', variable='YLDF')
        df_YLDF.to_csv(f'{out_dir}/Annual_YLDF.csv')         
        df_BIOM = self.import_output(location='annual', variable='BIOM')
        df_BIOM.to_csv(f'{out_dir}/Annual_BIOM.csv') 
        df_WS = self.import_output(location='annual', variable='WS')
        df_WS.to_csv(f'{out_dir}/Annual_WS.csv')    
        df_NS = self.import_output(location='annual', variable='NS')
        df_NS.to_csv(f'{out_dir}/Annual_NS.csv') 
        df_PS = self.import_output(location='annual', variable='PS')
        df_PS.to_csv(f'{out_dir}/Annual_PS.csv')
        df_TS = self.import_output(location='annual', variable='TS')
        df_TS.to_csv(f'{out_dir}/Annual_TS.csv')        
        
    def get_range (self):
         # import csv file containing range of parameters with recommended values for specific project
        df_param_limit = pd.read_csv(self.file_limits,  index_col=0, encoding="ISO-8859-1")
        array_param_list = df_param_limit.iloc[1:, :].to_numpy()
        array_param_list = array_param_list.astype(np.float)
        mat_param_list = np.asmatrix(array_param_list)
        mat_param_list = np.squeeze(np.asarray(mat_param_list))
        self.param_range = mat_param_list
        return self        
    
    def get_pe_files(self):
        if (self.attribute=='WYLD'):
            file_pe = os.path.join(self.folder, 'Statistics_runoff.csv')
        elif (self.attribute=='YSD'):
            file_pe = os.path.join(self.folder, 'Statistics_sediment.csv')
        else:
            file_pe = os.path.join(self.folder, f'Statistics_{self.attribute}.csv')
        file_parameter = os.path.join(self.folder, 'APEXPARM.csv')
        self.file_pem = file_pe
        self.file_param = file_parameter
        return self
    
    def get_params(self, id_sensitive, isall):
        df_params = pd.read_csv(self.file_param)
        df_params.rename(columns = {'Unnamed: 0':'RunId'}, inplace = True)
        self.param_list =df_params.columns
        self.params_all = df_params
        if isall:
            self.params = df_params.copy()
        else:               
            self.param_list = self.param_list[id_sensitive]
            df_param = df_params[self.param_list]
            df_param.insert(0, 'RunId', df_params.RunId)                
        self.pbest = df_param.iloc[0,:]
        self.params = df_param.iloc[1:, ]
        return self

    def read_pem(self, metric_list):
        # read statistics
        df_pem = ap.get_stats(self.file_pem)
        df_pem_obs = df_pem[metric_list]
        df_pem_obs.columns = ['RunId', 'COD', 'RMSE', 'NRMSE','NSE', 'PBIAS', 'OF']
        pebest = df_pem_obs.iloc[0, :]
        df_pem_obs = df_pem_obs.iloc[1:, ]
        self.pem = df_pem
        self.pem4criteria = df_pem_obs
        self.pebest = pebest
        return self
    
    def assign_dates(self, df_mod, n_warm, n_calib_year):
        start_year = df_mod.Year.values[0]
        cal_start = start_year + n_warm
        cal_end = cal_start + n_calib_year
        val_end = self.df_obs.Year[-1]
        val_end_date = self.df_obs.Date[-1]
        return start_year, cal_start, cal_end, val_end, val_end_date
    
    def import_output(self, location, variable):
        print (f'\nExporting {variable} data')
        n_runs = self.pem4criteria.shape[0]
        n_warm =  int(self.config['warm_years'])
        n_calib_year = int(self.config['calib_years'])
        df_out = pd.DataFrame()                   
        print_progress_bar(0, n_runs+1, prefix='Progress', suffix='Complete', length=50, fill='█')
        for i in range(1, n_runs+1):
            if location=='outlet':
                file_outlet =os.path.join(self.folder, f'daily_outlet_{i:07}.csv.csv')
                data = pd.read_csv(file_outlet)
                data.index = data.Date
                data.index = pd.to_datetime(data.index)             
                data=data.drop(['Date', 'Y', 'M', 'D'], axis=1)
                data.insert(0, 'Year', data.index.year, True)
                start_year, cal_start, cal_end, val_end, val_end_date =  self.assign_dates(data, n_warm, n_calib_year)
                df_data_cal, df_data_val = split_data(start_year, data, n_warm, n_calib_year)
                df_data_val = df_data_val[df_data_val.index<=val_end_date]
            elif location=='basin':
                file_basin =os.path.join(self.folder, f'daily_basin_{i:07}.csv.csv')
                data = pd.read_csv(file_basin)
                data.index = data.Date
                data.index = pd.to_datetime(data.index)
                data=data.drop(['Date', 'Y', 'M', 'D'], axis=1)
                data.insert(0, 'Year', data.index.year, True)
                start_year, cal_start, cal_end, val_end, val_end_date =  self.assign_dates(data, n_warm, n_calib_year)
                df_data_cal, df_data_val = split_data(start_year, data, n_warm, n_calib_year)                
                df_data_val = df_data_val[df_data_val.index<=val_end_date]
            else:
                file_annual =os.path.join(self.folder, f'annual_{i:07}.csv.csv')
                data = pd.read_csv(file_annual)        
                data.index = data.YR
                data=data.drop(['Unnamed: 0', 'YR'], axis=1)
                data.insert(0, 'Year', data.index, True)
                start_year, cal_start, cal_end, val_end, val_end_date =  self.assign_dates(data, n_warm, n_calib_year)
                df_data_cal, df_data_val = split_data(start_year, data, n_warm, n_calib_year)
                df_data_val = df_data_val[df_data_val.index<=val_end]
            df_data_cal.insert(df_data_cal.shape[1], 'Stage', 'Calibration', True)
            df_data_val.insert(df_data_val.shape[1], 'Stage', 'Validation', True)
            df_data = pd.concat([df_data_cal, df_data_val], axis=0)
            stage_vec = df_data.Stage.values
            df_data = pd.DataFrame(df_data[variable])
            df_data.columns = [f'{i}']
            df_out = pd.concat([df_out, df_data], axis=1)
            print_progress_bar(i, n_runs+1, prefix=f'Progress: {i}', suffix='Complete', length=50, fill='█')
        df_out.insert(0, 'Stage', stage_vec, True)            
        return df_out
    



        



