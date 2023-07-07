# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:16:44 2022

@author: Mahesh.Maskey, Brian Stucky (USDA)
"""
import os
import shutil
import subprocess
from datetime import datetime
import fortranformat as ff
import numpy as np
import pandas as pd
from Utility.apex_utility import calculate_nutrients
from Utility.apex_utility import get_acy
from Utility.apex_utility import get_daily_dps
from Utility.apex_utility import get_daily_dws
from Utility.apex_utility import get_daily_sad
from Utility.apex_utility import modify_list
from Utility.apex_utility import read_param_file
from Utility.apex_utility import get_scanario_name
from Utility.apex_utility import txt2list
from Utility.apex_utility import validate_model
from Utility.overwrite_param import overwrite_param


class inAPEX:
    def __init__(self, scenario):
        """
        prog_dir: The location of the APEX simulation software.
        """
        # Specify the paths for all required input files.
        self.vec_month = None
        self.vec_date = None
        self.vec_year = None
        self.stop_date = None
        self.start_date = None
        self.last_year = None
        self.start_day = None
        self.start_month = None
        self.start_year = None
        self.n_years = None
        # self.control_file = f'{prog_dir}/APEXCONT.DAT'
        self.control_file = 'APEXCONT.DAT'
        # self.list_file = f'{prog_dir}/APEXFILE.DAT'
        self.list_file = 'APEXFILE.DAT'
        # self.param_file = f'{prog_dir}/APEXPARM.DAT'
        self.param_file = 'APEXPARM.DAT'
        self.new_param_file = f'{scenario}_APEXPARM.DAT'
        # self.run_file = f'{prog_dir}/APEXRUN.DAT'
        self.run_file = 'APEXRUN.DAT'
        self.weather_file_list = 'WDLYLIST.DAT'
        self.weather_file = 'WEATHER.DLY'
        self.file_observe = 'calibration_data.csv'
        self.get_run_name()
        self.get_control_period()
        # Search subarea file in APEXFILE list
        file_list = txt2list(self.list_file)
        df_file_list = pd.DataFrame(file_list)
        area_file = df_file_list.iloc[np.where(df_file_list.iloc[:, 0] == 'FSUBA')[0][0], 1]
        # area_file_path = os.path.join(prog_dir, area_file)
        area_list = txt2list(area_file)
        # self.file_subarea = os.path.join(prog_dir, area_list[0][1])
        self.file_subarea = area_list[0][1]
        self.watershed_area = self.get_watershed_area()
        self.n_dates = len(self.vec_date)
        self.n_month = len(self.vec_month)
        self.n_years = len(self.vec_year)

    def get_run_name(self):
        lines = read_param_file(self.run_file)
        read_format = ff.FortranRecordReader('(A10, 6I7)')
        line_read = read_format.read(lines[0])
        run_name = line_read[0].split(' ')[0]
        self.run_name = run_name

    def get_control_period(self):
        # Extracts the simulation period, start date from APEXCONT.DAT file
        # and computes the date vector in three time scales: days, month, and year
        lines = read_param_file(self.control_file)
        read_format = ff.FortranRecordReader('(20I6)')
        line_read = read_format.read(lines[0])
        self.n_years = line_read[0]
        self.start_year = line_read[1]
        self.start_month = line_read[2]
        self.start_day = line_read[3]
        self.last_year = self.start_year + self.n_years - 1
        self.start_date = pd.to_datetime(str(self.start_year) + '/' + str(self.start_month) + '/' + str(self.start_day))
        self.stop_date = pd.to_datetime(str(self.last_year) + '/12/31')
        self.vec_date = pd.date_range(self.start_date, self.stop_date, freq='d')
        self.vec_month = pd.date_range(self.start_date, self.stop_date, freq='M')
        self.vec_year = pd.date_range(self.start_date, self.stop_date, freq='Y')
        return self

    def get_watershed_area(self):
        # Reads the subarea file and extracts the watershed area in hectare.
        lines = read_param_file(self.file_subarea)
        read_format = ff.FortranRecordReader('(10F8.3)')
        line_read = read_format.read(lines[3])
        return line_read[0]


class simAPEX:
    def __init__(self, src_dir, winepath, in_obj, attribute, is_pesticide, scenario, id_case, warm_years, calib_years):
        """
        src_dir: The location of the source repository for managing APEX runs.
        winepath: Path to an executable Wine container image.
        in_obj: An inAPEX object.
        """
        self.src_dir = src_dir
        self.winepath = winepath
        self.run_name = in_obj.run_name
        self.cal_param_file = self.src_dir / 'Output' / attribute / 'summary_APEXPARM.csv'

        # create folder to save selected output attributes
        self.dir_output = os.path.join(os.path.split(os.getcwd())[0], f'Output_{scenario}')
        self.weather_file = in_obj.weather_file
        self.new_weather_file = os.path.join(os.path.abspath('../../CLIMATE'), f'{scenario}.dly')
        if os.path.exists(self.dir_output) is False:
            os.makedirs(self.dir_output)
        self.file_pem = os.path.join(self.dir_output, f'stats_{attribute}')
        # read calibrated parameter set
        self.id_case = id_case
        self.read_param()

        self.n_params = len(self.param)
        self.assign_output_df()
        self.p = self.param
        self.p = overwrite_param(in_obj.param_file, in_obj.new_param_file, self.p)
        modify_list(in_obj.list_file, in_obj.new_param_file)
        self.copy_weather_file()
        t0 = datetime.now()
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
        # Saving standard output file together with runs for final summary
        # rename run_name_[iteration].out e.g., 001RUN_0000106.out
        self.copy_rename_file(curr_directory, extension='OUT')
        print(f'Completed simulation in {round((datetime.now() - t0).total_seconds(), 3)} seconds')
        modify_list(in_obj.list_file, in_obj.param_file)
        df_SAD = get_daily_sad(self.run_name)
        df_SAD = calculate_nutrients(df_SAD)
        df_DPS = get_daily_dps(self.run_name)
        df_DWS = get_daily_dws(self.run_name)
        df_annual = get_acy(self.run_name)
        print('Saving model performance statistics')
        self.do_validate_fill(df_SAD, in_obj, 'WYLD', 'runoff', warm_years, calib_years)
        if is_pesticide:
            self.do_validate_fill(df_DPS, in_obj, 'YSD', 'sediment', warm_years, calib_years)
        self.do_validate_fill(df_SAD, in_obj, 'USLE', 'sediment', warm_years, calib_years)
        self.do_validate_fill(df_SAD, in_obj, 'MUSL', 'sediment', warm_years, calib_years)
        self.do_validate_fill(df_SAD, in_obj, 'REMX', 'sediment', warm_years, calib_years)
        self.do_validate_fill(df_SAD, in_obj, 'MUSS', 'sediment', warm_years, calib_years)
        self.do_validate_fill(df_SAD, in_obj, 'MUST', 'sediment', warm_years, calib_years)
        self.do_validate_fill(df_SAD, in_obj, 'RUS2', 'sediment', warm_years, calib_years)
        self.do_validate_fill(df_SAD, in_obj, 'RUSL', 'sediment', warm_years, calib_years)

        file_outlet = f'daily_outlet.csv'
        df_outlet = df_DWS[['Y', 'M', 'D', 'RFV', 'WYLD', 'TMX', 'TMN', 'PET', 'Q', 'CN', 'SSF', 'PRK', 'IRGA', 'USLE',
                            'MUSL', 'REMX', 'MUSS', 'MUST', 'RUS2', 'RUSL']]
        if is_pesticide:
            df_outlet['YSD'] = df_DPS['YSD']
        df_outlet.to_csv(f'{self.dir_output}/{file_outlet}')
        file_basin = f'daily_basin.csv'
        df_basin = df_SAD[['Y', 'M', 'D', 'CPNM', 'LAI', 'BIOM', 'STL', 'STD', 'STDL', 'PRCP', 'WYLD', 'TMX', 'TMN',
                           'PET', 'ET', 'Q', 'CN', 'SSF', 'PRK', 'QDR', 'IRGA', 'USLE', 'MUSL', 'REMX', 'MUSS', 'MUST',
                           'RUS2', 'RUSL', 'YN', 'YP', 'QN', 'QP', 'QDRN', 'QPRP', 'SSFN', 'RSFN', 'QRFN', 'QRFP',
                           'QDRP', 'DPRK', 'TN', 'TP']]
        df_outlet.to_csv(f'{self.dir_output}/{file_basin}')
        file_annual = f'annual.csv'
        df_year = df_annual[['YR', 'CPNM', 'YLDG', 'YLDF', 'BIOM', 'WS', 'NS', 'PS', 'TS', 'AS', 'SS']]
        df_year.to_csv(f'{self.dir_output}/{file_annual}')

        crops = df_basin.CPNM.unique()
        if len(crops) > 1:
            for cp in crops:
                file_basin = f'daily_basin_{self.run_name}_{cp}.csv'
                df_basin_c = df_basin[df_basin['CPNM'] == cp]
                df_basin_c.to_csv(f'{self.dir_output}/{file_basin}.csv')
                file_annual = f'daily_basin_{self.run_name}_{cp}.csv'
                df_year_c = df_year[df_year['CPNM'] == cp]
                df_year_c.to_csv(f'{self.dir_output}/{file_annual}.csv')

        df_p = pd.DataFrame(self.p)
        df_p = df_p.T
        df_p.columns = self.param_description[1:-1]
        self.df_p = df_p
        if 'RunId' in self.df_p:
            self.df_p.index = self.df_p.RunId.values
            self.df_p = self.df_p.drop(['RunId'], axis=1)
        print('Saving parameters in APEXPARM.DAT')
        self.df_p.to_csv(f'{self.dir_output}/APEXPARM.csv')
        print('Parameters')
        print(self.p)
        print('---------------------------------------------------------------')
        print(f'\nCompleted runs in {round((datetime.now() - t0).total_seconds(), 3)} seconds')

    def get_stats(self):
        df_pem = pd.read_csv(self.file_pem)
        df_pem.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
        self.stats = df_pem
        return self

    def read_param(self):
        # import csv file containing range of parameters with recommended values for specific project
        df_param = pd.read_csv(self.cal_param_file, index_col=0, encoding="ISO-8859-1")
        self.param_description = df_param.columns.to_list()
        df_param_cal = df_param.iloc[:, 1:-1]
        id_case = self.id_case-1
        self.df_param = df_param
        self.df_param_cal = df_param_cal
        param = df_param_cal.iloc[id_case, :].values
        param_id = df_param.iloc[id_case, id_case]
        param_info = df_param_cal.index[id_case]
        self.param = param
        self.param_id = param_id
        self.param_info = param_info
        print(f'Current parameters based on {param_info}')
        return self

    def copy_weather_file(self):
        shutil.copy2(self.new_weather_file, self.weather_file)
        print(f'{self.weather_file} was replaced by {self.new_weather_file}')
        return self

    def copy_rename_file(self, curr_directory, extension):
        outfile = f'{self.run_name}.{extension}'
        source_file = os.path.join(curr_directory, f'{self.run_name}.{extension}')
        dist_file = os.path.join(self.dir_output, outfile)
        shutil.copy2(source_file, dist_file)
        print(f'{source_file} is copied into {self.dir_output} as {outfile}')

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

    def do_validate_fill(self, df, in_obj, attribute, observation, wy, cy):
        pem = ['CODAD', 'RMSEAD', 'NRMSEAD', 'NSEAD', 'PBIASAD', 'APBIASAD', 'IOAAD', 'OF1AD', 'OF2AD',
               'CODDC', 'RMSEDC', 'NRMSEDC', 'NSEDC', 'PBIASDC', 'APBIASDC', 'IOADC', 'OF1DC', 'OF2DC',
               'CODDV', 'RMSEDV', 'NRMSEDV', 'NSEDV', 'PBIASDV', 'APBIASDV', 'IOADV', 'OF1DV', 'OF2DV',
               'CODAM', 'RMSEAM', 'NRMSEAM', 'NSEAM', 'PBIASAM', 'APBIASAM', 'IOAAM', 'OF1AM', 'OF2AM',
               'CODMC', 'RMSEMC', 'NRMSEMC', 'NSEMC', 'PBIASMC', 'APBIASMC', 'IOAMC', 'OF1MC', 'OF2MC',
               'CODMV', 'RMSEMV', 'NRMSEMV', 'NSEMV', 'PBIASMV', 'APBIASMV', 'IOAMV', 'OF1MV', 'OF2MV',
               'CODAY', 'RMSEAY', 'NRMSEAY', 'NSEAY', 'PBIASAY', 'APBIASAY', 'IOAAY', 'OF1AY', 'OF2AY',
               'CODYC', 'RMSEYC', 'NRMSEYC', 'NSEYC', 'PBIASYC', 'APBIASYC', 'IOAYC', 'OF1YC', 'OF2YC',
               'CODYV', 'RMSEYV', 'NRMSEYV', 'NSEYV', 'PBIASYV', 'APBIASYV', 'IOAYV', 'OF1YV', 'OF2YV']
        file_observe = in_obj.file_observe
        wa = in_obj.watershed_area
        start_year = in_obj.start_year
        df_i = validate_model(df, attribute, observation, file_observe, pem, wa, start_year, wy, cy)
        if (attribute == 'WYLD') | (attribute == 'YSD'):
            file_name = 'Statistics_' + observation
        else:
            file_name = 'Statistics_Soil_erosion_' + attribute
        df_i.to_csv(f'{self.dir_output}/{file_name}.csv')
        return df_i
