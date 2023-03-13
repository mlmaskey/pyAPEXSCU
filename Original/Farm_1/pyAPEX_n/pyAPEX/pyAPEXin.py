# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:16:44 2022

@author: Mahesh.Maskey, Brian Stucky (USDA)
"""

import pandas as pd
import numpy as np
import fortranformat as ff
from Utility.apex_utility import read_param_file
from Utility.apex_utility import txt2list

class inAPEX:
    def __init__(self, config, prog_dir):
        """
        config: A run configuration object.
        prog_dir: The location of the APEX simulation softare.
        """
        # Specify the paths for all required input files.
        self.control_file = prog_dir / 'APEXCONT.DAT'
        self.list_file = prog_dir / 'APEXFILE.DAT'
        self.param_file = prog_dir / 'APEXPARM.DAT'
        self.simparam_file = prog_dir / 'simAPEXPARM.DAT'        
        self.file_observe = prog_dir / config['file_observe']
        self.sim_id_range = f'_{(config["n_start"] + 1):07}-{(config["n_simulation"]):07}'
        self.dir_calibration = prog_dir.parent / (config['dir_calibration'] + self.sim_id_range)
        self.dir_sensitivity = prog_dir.parent / (config['dir_sensitivity'] + self.sim_id_range)
        self.dir_uncertainty = prog_dir.parent / (config['dir_uncertainty'] + self.sim_id_range)

        self.get_control_period()
        # Search subarea file in APEXFILE list
        file_list = txt2list(self.list_file)
        df_file_list = pd.DataFrame(file_list)
        list_area = df_file_list.iloc[np.where(df_file_list.iloc[:, 0]=='FSUBA')[0][0], 1]
        area_list = txt2list(prog_dir / list_area)
        self.file_subarea = prog_dir / area_list[0][1]
        self.watershed_area = self.get_watershed_area()
        self.ndates = len(self.vec_date)
        self.nmonth = len(self.vec_month)
        self.nyears = len(self.vec_year)

    def get_control_period(self):
        # Extracts the simulation period, start date from APEXCONT.DAT file
        # and computes the date vector in three time scales: days, month, and year
        lines = read_param_file(self.control_file )
        read_format = ff.FortranRecordReader('(20I6)')
        line_read = read_format.read(lines[0])
        self.n_years = line_read[0]
        self.startyear = line_read[1]
        self.startmonth = line_read[2]
        self.startday = line_read[3]
        self.lastyear = self.startyear+self.n_years-1
        self.startdate = pd.to_datetime(str(self.startyear) + '/' +str(self.startmonth) + '/' + str(self.startday))
        self.stopdate = pd.to_datetime(str(self.lastyear) + '/12/31')
        self.vec_date = pd.date_range(self.startdate, self.stopdate, freq='d')
        self.vec_month = pd.date_range(self.startdate, self.stopdate, freq='M')
        self.vec_year = pd.date_range(self.startdate, self.stopdate, freq='Y')
        return self

    def get_watershed_area(self):
        # Reads the subarea file and extracts the watershed area in hactare.
        lines = read_param_file(self.file_subarea)
        read_format = ff.FortranRecordReader('(10F8.3)')
        line_read = read_format.read(lines[3])
        return line_read[0]

