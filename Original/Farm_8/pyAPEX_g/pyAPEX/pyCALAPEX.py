# ## Calibration of APEX model for surface runoff data

# ### Import libraries

import os
import pandas as pd
import numpy as np
import warnings
from configobj import ConfigObj
from Utility.pyAPEXpost import pyAPEXpost as ap
warnings.filterwarnings('ignore') 
config = ConfigObj('runtime.ini')

# ### File locations and input parameters
#  Main input

# cluster_dir = 'G:/PostDocResearch/USDA-ARS/Project/OklahomaWRE/Cluster_data/'
# mod_attribute = ['WYLD', 'USLE', 'MUSL', 'REMX', 'MUSS', 'RUS2', 'RUSL', 'YSD']
class calAPEX:
    def __init__(self, src_dir, config, cluster_dir, site, scenario, obs_attribute, mod_attribute, out_dir):
        
        """  
        
 

        """
        self.config = config
        self.obs_attribute = obs_attribute
        self.site = site
        self.scenario = scenario
        self.mod_attribute = mod_attribute
        self.src_dir = src_dir 
        self.cluster_dir = cluster_dir 
        self.metrics = ['OF', 'NSE', 'PBIAS', 'COD'] 
        self.cal_dir = config['dir_calibration']
        self.out_dir = out_dir
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)            
        self.get_pe_files() # Set the locations of modeled and process data
        # #### Set the criteria 
        if obs_attribute=='runoff':
            self.criteria = [float(config['COD_criteria']), float(config['NSE_criteria']), float(config['PBAIS_criteria'])]
        elif obs_attribute=='sediment':
            self.criteria = [float(config['COD_criteria_sediment']), float(config['NSE_criteria_sediment']), float(config['PBAIS_criteria_sediment'])]            
        elif obs_attribute=='Soil_erosion':
             self.criteria = [float(config['COD_criteria_sediment']), float(config['NSE_criteria_sediment']), float(config['PBAIS_criteria_sediment'])]            
        self.get_stats() # ### Read statistics
        self.daily_stats()
        self.monthly_stats()
        self.yearly_stats()
        # combine all the stats in long form
        self.df_stats = pd.concat([self.stats_daily, self.stats_monthly, self.stats_yearly], axis=0)
        self.count_stats()
        self.df_stats.to_csv(f'{self.out_dir}/best_stats.csv')
        #### Merge stata summer and export
        try:
            self.extract_objective_function(scale='daily')
            self.extract_objective_function(scale='monthly')
            self.extract_objective_function(scale='yearly')
        except:
            self.get_stats_by_metric(scale='daily')
            self.get_stats_by_metric(scale='monthly')
            self.get_stats_by_metric(scale='yearly')
            self.extract_objective_function(scale='daily')
            self.extract_objective_function(scale='monthly')
            self.extract_objective_function(scale='yearly')            
        self.df_stats = pd.concat([self.best_stats_daily, self.best_stats_monthly, self.best_stats_yearly], axis=0)
        self.df_stats.to_csv(f'{self.out_dir}/summary_stats.csv')
        # Import parameters
        self.import_parameter()
        # concate parameter summary and export
        self.df_param_best = pd.concat([self.df_param_best_daily, self.df_param_best_monthly, self.df_param_best_yearly], axis=0)
        self.df_param_best.to_csv(f'{self.out_dir}/summary_APEXPARM.csv') 
        # ### Import modeled data
        self.assign_read_dir()
        self.import_modeled_data(scale='daily', croplist=None)
        self.import_modeled_data(scale='monthly', croplist=None)
        self.import_modeled_data(scale='yearly', croplist=None)       
        # Read observed data
        self.df_observed = ap.get_measure (data_dir = 'Program', file_name = 'calibration_data.csv')
        
        # Seperate data for model and simulation           
        ## Finalize model data for outlet
        # for objective function
        df_model_outlet_daily, df_calibration_daily = self.final_data(scale='daily', metric='OF', location='outlet')
        df_model_outlet_monthly, df_calibration_monthly = self.final_data(scale='monthly', metric='OF', location='outlet')
        df_model_outlet_yearly, df_calibration_yearly = self.final_data(scale='yearly', metric='OF', location='outlet')
        self.df_model_out_OF = pd.concat([df_model_outlet_daily, df_model_outlet_monthly, df_model_outlet_yearly], axis = 0)
        del df_model_outlet_daily, df_model_outlet_monthly, df_model_outlet_yearly
        self.df_model_out_OF.to_csv(f'{self.out_dir}/model_result_outlet_OF.csv')         
        self.df_model_calibration = pd.concat([df_calibration_daily, df_calibration_monthly, df_calibration_yearly], axis = 0)
        self.df_model_calibration.to_csv(f'{self.out_dir}/model_calibration_OF.csv')  
        df_model_basin_daily = self.final_data(scale='daily', metric='OF', location='basin')
        df_model_basin_monthly = self.final_data(scale='monthly', metric='OF', location='basin')
        df_model_basin_yearly = self.final_data(scale='yearly', metric='OF', location='basin')       
        self.df_model_basin_OF = pd.concat([df_model_basin_daily, df_model_basin_monthly, df_model_basin_yearly], axis = 0)
        self.df_model_basin_OF.to_csv(f'{self.out_dir}/model_result_basin_OF.csv')         
        del df_model_basin_daily, df_model_basin_monthly, df_model_basin_yearly
        df_model_annual_daily = self.final_data(scale='daily', metric='OF', location='annual')       
        df_model_annual_monthly = self.final_data(scale='monthly', metric='OF', location='annual')       
        df_model_annual_yearly = self.final_data(scale='yearly', metric='OF', location='annual')       
        self.df_model_annual_OF = pd.concat([df_model_annual_daily, df_model_annual_monthly, df_model_annual_yearly], axis = 0)
        self.df_model_annual_OF.to_csv(f'{self.out_dir}/model_result_annual_OF.csv')              
        del df_model_annual_daily, df_model_annual_monthly, df_model_annual_yearly

        # for Nash-Sutcliffe efficiency
        df_model_outlet_daily, df_calibration_daily = self.final_data(scale='daily', metric='NSE', location='outlet')
        df_model_outlet_monthly, df_calibration_monthly = self.final_data(scale='monthly', metric='NSE', location='outlet')
        df_model_outlet_yearly, df_calibration_yearly = self.final_data(scale='yearly', metric='NSE', location='outlet')
        self.df_model_out_NSE = pd.concat([df_model_outlet_daily, df_model_outlet_monthly, df_model_outlet_yearly], axis = 0)
        del df_model_outlet_daily, df_model_outlet_monthly, df_model_outlet_yearly
        self.df_model_out_NSE.to_csv(f'{self.out_dir}/model_result_outlet_NSE.csv')         
        self.df_model_calibration = pd.concat([df_calibration_daily, df_calibration_monthly, df_calibration_yearly], axis = 0)
        self.df_model_calibration.to_csv(f'{self.out_dir}/model_calibration_NSE.csv')  
        df_model_basin_daily = self.final_data(scale='daily', metric='NSE', location='basin')
        df_model_basin_monthly = self.final_data(scale='monthly', metric='NSE', location='basin')
        df_model_basin_yearly = self.final_data(scale='yearly', metric='NSE', location='basin')       
        self.df_model_basin_NSE = pd.concat([df_model_basin_daily, df_model_basin_monthly, df_model_basin_yearly], axis = 0)
        self.df_model_basin_NSE.to_csv(f'{self.out_dir}/model_result_basin_NSE.csv')         
        del df_model_basin_daily, df_model_basin_monthly, df_model_basin_yearly
        df_model_annual_daily = self.final_data(scale='daily', metric='NSE', location='annual')       
        df_model_annual_monthly = self.final_data(scale='monthly', metric='NSE', location='annual')       
        df_model_annual_yearly = self.final_data(scale='yearly', metric='NSE', location='annual')       
        self.df_model_annual_NSE = pd.concat([df_model_annual_daily, df_model_annual_monthly, df_model_annual_yearly], axis = 0)
        self.df_model_annual_NSE.to_csv(f'{self.out_dir}/model_result_annual_NSE.csv')              
        del df_model_annual_daily, df_model_annual_monthly, df_model_annual_yearly        

        # for Coefficient of determination
        df_model_outlet_daily, df_calibration_daily = self.final_data(scale='daily', metric='COD', location='outlet')
        df_model_outlet_monthly, df_calibration_monthly = self.final_data(scale='monthly', metric='COD', location='outlet')
        df_model_outlet_yearly, df_calibration_yearly = self.final_data(scale='yearly', metric='COD', location='outlet')
        self.df_model_out_COD = pd.concat([df_model_outlet_daily, df_model_outlet_monthly, df_model_outlet_yearly], axis = 0)
        del df_model_outlet_daily, df_model_outlet_monthly, df_model_outlet_yearly
        self.df_model_out_COD.to_csv(f'{self.out_dir}/model_result_outlet_COD.csv')         
        self.df_model_calibration = pd.concat([df_calibration_daily, df_calibration_monthly, df_calibration_yearly], axis = 0)
        self.df_model_calibration.to_csv(f'{self.out_dir}/model_calibration_COD.csv')  
        df_model_basin_daily = self.final_data(scale='daily', metric='COD', location='basin')
        df_model_basin_monthly = self.final_data(scale='monthly', metric='COD', location='basin')
        df_model_basin_yearly = self.final_data(scale='yearly', metric='COD', location='basin')       
        self.df_model_basin_COD = pd.concat([df_model_basin_daily, df_model_basin_monthly, df_model_basin_yearly], axis = 0)
        self.df_model_basin_COD.to_csv(f'{self.out_dir}/model_result_basin_COD.csv')         
        del df_model_basin_daily, df_model_basin_monthly, df_model_basin_yearly
        df_model_annual_daily = self.final_data(scale='daily', metric='COD', location='annual')       
        df_model_annual_monthly = self.final_data(scale='monthly', metric='COD', location='annual')       
        df_model_annual_yearly = self.final_data(scale='yearly', metric='COD', location='annual')       
        self.df_model_annual_COD = pd.concat([df_model_annual_daily, df_model_annual_monthly, df_model_annual_yearly], axis = 0)
        self.df_model_annual_COD.to_csv(f'{self.out_dir}/model_result_annual_COD.csv')              
        del df_model_annual_daily, df_model_annual_monthly, df_model_annual_yearly

        # for Perecnt bias
        df_model_outlet_daily, df_calibration_daily = self.final_data(scale='daily', metric='PBIAS', location='outlet')
        df_model_outlet_monthly, df_calibration_monthly = self.final_data(scale='monthly', metric='PBIAS', location='outlet')
        df_model_outlet_yearly, df_calibration_yearly = self.final_data(scale='yearly', metric='PBIAS', location='outlet')
        self.df_model_out_PBIAS = pd.concat([df_model_outlet_daily, df_model_outlet_monthly, df_model_outlet_yearly], axis = 0)
        del df_model_outlet_daily, df_model_outlet_monthly, df_model_outlet_yearly
        self.df_model_out_PBIAS.to_csv(f'{self.out_dir}/model_result_outlet_PBIAS.csv')         
        self.df_model_calibration = pd.concat([df_calibration_daily, df_calibration_monthly, df_calibration_yearly], axis = 0)
        self.df_model_calibration.to_csv(f'{self.out_dir}/model_calibration_PBIAS.csv')  
        df_model_basin_daily = self.final_data(scale='daily', metric='PBIAS', location='basin')
        df_model_basin_monthly = self.final_data(scale='monthly', metric='PBIAS', location='basin')
        df_model_basin_yearly = self.final_data(scale='yearly', metric='PBIAS', location='basin')       
        self.df_model_basin_PBIAS = pd.concat([df_model_basin_daily, df_model_basin_monthly, df_model_basin_yearly], axis = 0)
        self.df_model_basin_PBIAS.to_csv(f'{self.out_dir}/model_result_basin_PBIAS.csv')         
        del df_model_basin_daily, df_model_basin_monthly, df_model_basin_yearly
        df_model_annual_daily = self.final_data(scale='daily', metric='PBIAS', location='annual')       
        df_model_annual_monthly = self.final_data(scale='monthly', metric='PBIAS', location='annual')       
        df_model_annual_yearly = self.final_data(scale='yearly', metric='PBIAS', location='annual')       
        self.df_model_annual_PBIAS = pd.concat([df_model_annual_daily, df_model_annual_monthly, df_model_annual_yearly], axis = 0)
        self.df_model_annual_PBIAS.to_csv(f'{self.out_dir}/model_result_annual_PBIAS.csv')              
        del df_model_annual_daily, df_model_annual_monthly, df_model_annual_yearly        

    def get_pe_files(self):    
        if (self.mod_attribute=='WYLD'):
            file_pe = os.path.join(self.cal_dir, 'Statistics_runoff.csv')
        elif (self.mod_attribute=='YSD'):
            file_pe = os.path.join(self.cal_dir, 'Statistics_sediment.csv')
        else:
            file_pe = os.path.join(self.cal_dir, f'Statistics_Soil_erosion_{self.mod_attribute}.csv')
        file_parameter = os.path.join(self.cal_dir, 'APEXPARM.csv')
        self.file_pem = file_pe
        self.file_param = file_parameter
        return self
        
    def assign_read_dir(self):
        if self.scenario=='non_grazing':
            scenario1 = 'pyAPEX_n'
        else: 
            scenario1 = 'pyAPEX_g'
        self.read_dir = os.path.join(self.cluster_dir, 
                                     self.site, scenario1, 'Output')
        return self
    
    def create_output_dir(self):
        self.out_dir = os.path.join(self.out_dir, self.obs_attribute)
        if os.path.exists(self.out_dir) is False:
            os.makedirs(self.out_dir)
        return self
    
    def count_stats(self):
        criteria = self.criteria
        nCOD_daily = np.sum(self.df_pem.CODDC>=criteria[0])
        nNSE_daily = np.sum(self.df_pem.NSEDC>=criteria[1])
        nPBIAS_daily = np.sum(self.df_pem.APBIASDC<=criteria[2])
        nCOD_monthly = np.sum(self.df_pem.CODMC>=criteria[0])
        nNSE_monthly = np.sum(self.df_pem.NSEMC>=criteria[1])
        nPBIAS_monthly = np.sum(self.df_pem.APBIASMC<=criteria[2])        
        nCOD_yearly = np.sum(self.df_pem.CODYC>=criteria[0])
        nNSE_yearly = np.sum(self.df_pem.NSEYC>=criteria[1])
        nPBIAS_yearly = np.sum(self.df_pem.APBIASYC<=criteria[2])   
        self.dict_count_stats = {'CODD': nCOD_daily, 'NSED': nNSE_daily, 'PBIASD': nPBIAS_daily,
                                       'CODM': nCOD_monthly, 'NSEM': nNSE_monthly, 'PBIASM': nPBIAS_monthly,
                                       'CODY': nCOD_yearly, 'NSEY':nNSE_yearly, 'PBIASY': nPBIAS_yearly}
        return self
        
    
    def get_stats(self):
        df_pem = pd.read_csv(self.file_pem)
        df_pem.rename(columns = {'Unnamed: 0':'RunId'}, inplace = True)
        df_pem.insert(6, 'APBIASAD', np.abs(df_pem.PBIASAD), True)
        df_pem.insert(15, 'APBIASDC', np.abs(df_pem.PBIASDC), True)
        df_pem.insert(24, 'APBIASDV', np.abs(df_pem.PBIASDV), True)
        df_pem.insert(33, 'APBIASAM', np.abs(df_pem.PBIASAM), True)
        df_pem.insert(42, 'APBIASMC', np.abs(df_pem.PBIASMC), True)
        df_pem.insert(51, 'APBIASMV', np.abs(df_pem.PBIASMV), True)
        df_pem.insert(60, 'APBIASAY', np.abs(df_pem.PBIASAY), True)
        df_pem.insert(69, 'APBIASYC', np.abs(df_pem.PBIASYC), True)
        df_pem.insert(78, 'APBIASYV', np.abs(df_pem.PBIASYV), True)
        self.df_pem = df_pem
        return self
    
    def daily_stats(self):
        # get best stats at daily scale
        stats_daily = ap.compile_stats(df=self.df_pem, criteria=self.criteria, scale='daily')
        self.stats_daily = stats_daily[0]
        self.daily2daily = stats_daily[1]
        self.daily2monthly = stats_daily[2] 
        self.daily2yearly = stats_daily[3]
        self.daily_all = stats_daily[4]
        return self
    
    def get_stats_by_metric(self, scale='daily'):
        # get best stats at daily scale
        stats_COD = ap.compile_stats_by_metrics(df=self.df_pem, criteria=self.criteria, scale=scale, metric='COD')
        stats_NSE = ap.compile_stats_by_metrics(df=self.df_pem, criteria=self.criteria, scale=scale, metric='NSE')
        stats_PBIAS = ap.compile_stats_by_metrics(df=self.df_pem, criteria=self.criteria, scale=scale, metric='PBIAS')
        if scale=='daily':
            self.stats_daily_COD = stats_COD[0]
            self.daily2daily_COD = stats_COD[1]
            self.daily2monthly_COD = stats_COD[2] 
            self.daily2yearly_COD = stats_COD[3]
            self.daily_all_COD = stats_COD[4]
            self.stats_daily_NSE = stats_NSE[0]
            self.daily2daily_NSE = stats_NSE[1]
            self.daily2monthly_NSE = stats_NSE[2] 
            self.daily2yearly_NSE = stats_NSE[3]
            self.daily_all_NSE = stats_NSE[4]
            self.stats_daily_PBIAS = stats_PBIAS[0]
            self.daily2daily_PBIAS = stats_PBIAS[1]
            self.daily2monthly_PBIAS = stats_PBIAS[2] 
            self.daily2yearly_PBIAS = stats_PBIAS[3]
            self.daily_all_PBIAS = stats_PBIAS[4]            
        if scale=='monthly':
            self.stats_monthly_COD = stats_COD[0]
            self.monthly2daily_COD = stats_COD[1]
            self.monthly2monthly_COD = stats_COD[2] 
            self.monthly2yearly_COD = stats_COD[3]
            self.monthly_all_COD = stats_COD[4]
            self.stats_monthly_NSE = stats_NSE[0]
            self.monthly2daily_NSE = stats_NSE[1]
            self.monthly2monthly_NSE = stats_NSE[2] 
            self.monthly2yearly_NSE = stats_NSE[3]
            self.monthly_all_NSE = stats_NSE[4]
            self.stats_monthly_PBIAS = stats_PBIAS[0]
            self.monthly2daily_PBIAS = stats_PBIAS[1]
            self.monthly2monthly_PBIAS = stats_PBIAS[2] 
            self.monthly2yearly_PBIAS = stats_PBIAS[3]
            self.monthly_all_PBIAS = stats_PBIAS[4]             
        if scale=='yearly':
            self.stats_yearly_COD = stats_COD[0]
            self.yearly2daily_COD = stats_COD[1]
            self.yearly2monthly_COD = stats_COD[2] 
            self.yearly2yearly_COD = stats_COD[3]
            self.yearly_all_COD = stats_COD[4]
            self.stats_yearly_NSE = stats_NSE[0]
            self.yearly2daily_NSE = stats_NSE[1]
            self.yearly2monthly_NSE = stats_NSE[2] 
            self.yearly2yearly_NSE = stats_NSE[3]
            self.yearly_all_NSE = stats_NSE[4]
            self.stats_yearly_PBIAS = stats_PBIAS[0]
            self.yearly2daily_PBIAS = stats_PBIAS[1]
            self.yearly2monthly_PBIAS = stats_PBIAS[2] 
            self.yearly2yearly_PBIAS = stats_PBIAS[3]
            self.yearly_all_PBIAS = stats_PBIAS[4]            
            
        return self
    
    def monthly_stats(self):
        # get best stats at monthly scale
        stats_monthly = ap.compile_stats(df=self.df_pem, criteria=self.criteria, scale='monthly')
        self.stats_monthly = stats_monthly[0]
        self.monthly2daily = stats_monthly[1]
        self.monthly2monthly = stats_monthly[2] 
        self.monthly2yearly = stats_monthly[3]
        self.monthly_all = stats_monthly[4]
        return self

    def yearly_stats(self):
        # get best stats at yearly scale
        stats_yearly = ap.compile_stats(df=self.df_pem, criteria=self.criteria, scale='daily')
        self.stats_yearly = stats_yearly[0]
        self.yearly2daily = stats_yearly[1]
        self.yearly2monthly = stats_yearly[2] 
        self.yearly2yearly = stats_yearly[3]
        self.yearly_all = stats_yearly[4]
        return self
    
    def extract_objective_function(self, scale):
        if scale == 'daily':
            df = self.daily2daily
            metrics = ['OF2DC', 'NSEDC', 'APBIASDC', 'CODDC']
        elif scale=='monthly':
            df = self.monthly2monthly
            metrics = ['OF2MC', 'NSEMC', 'APBIASMC', 'CODMC']
        else:
            df = self.yearly2yearly
            metrics = ['OF2YC', 'NSEYC', 'APBIASYC', 'CODYC']
        stats = ['min', 'max', 'min', 'max']
        best_stats, ids_best = ap.summarize_stats(df, scale, metrics, stats)
        if scale == 'daily':
            self.best_stats_daily, self.ids_best_daily =  best_stats, ids_best 
        elif scale=='monthly':
            self.best_stats_monthly, self.ids_best_monthly =  best_stats, ids_best
        else:
            self.best_stats_yearly, self.ids_best_yearly =  best_stats, ids_best
        return self

    def import_parameter(self):
        # daily
        self.df_param_best_daily = ap.get_best_params(file_name=self.file_param,
                                                      ids_bests=self.ids_best_daily,
                                                      scale='daily')
        # monthly
        self.df_param_best_monthly= ap.get_best_params(file_name=self.file_param, 
                                                  ids_bests=self.ids_best_monthly, 
                                                  scale='monthly')                                       
        # yearly
        self.df_param_best_yearly= ap.get_best_params(file_name=self.file_param, 
                                                  ids_bests=self.ids_best_yearly, 
                                                  scale='yearly')
        return self
            
    def import_modeled_data(self, scale, croplist=None):
        if scale=='daily':
            ids_best = self.ids_best_daily
        elif scale=='monthly':
            ids_best = self.ids_best_monthly
        else:
            ids_best = self.ids_best_yearly
            
        outlet, basin, annual = ap.import_save(dir_data=self.read_dir,
                                               attribute=self.obs_attribute,
                                               croplist=croplist, 
                                               ids=ids_best,
                                               scale=scale, 
                                       dir_save=self.out_dir)
        if scale=='daily':
            self.outlet_daily, self.basin_daily, self.annual_daily = outlet, basin, annual
        elif scale=='monthly':
            self.outlet_monthly, self.basin_monthly, self.annual_monthly = outlet, basin, annual
        else:
            self.outlet_yearly, self.basin_yearly, self.annual_yearly = outlet, basin, annual
        return self
    
    def final_data(self, scale, metric, location):
        if location=='outlet':
            outlet_result = ap.finalize_outlet_result (self.out_dir, 
                                                       self.df_observed, 
                                                       self.obs_attribute, 
                                                       self.mod_attribute,  
                                                       scale, metric)
            df_model_daily = outlet_result[0]
            df_calibration_daily =  outlet_result[1]
            return df_model_daily, df_calibration_daily 
        elif location=='basin':
            df_outlet_daily, _ = self.final_data(scale, metric, location='outlet')
            df_model_daily = ap.finalize_basin_result (self.out_dir,
                                                       self.df_observed, 
                                                       df_outlet_daily, 
                                                       self.obs_attribute, 
                                                       self.mod_attribute,
                                                       scale, metric)
            return df_model_daily
        elif location=='annual':
            df_outlet_daily, _ = self.final_data(scale, metric, location='outlet')
            df_model_yearly = ap.finalize_annual_result (self.out_dir, 
                                                                self.df_observed, 
                                                                df_outlet_daily,
                                                               self.obs_attribute,
                                                               self.mod_attribute,
                                                               scale, metric)
            return df_model_yearly
            
        
            
        


                                        



