# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 07:42:35 2022

@author: Mahesh.Maskey
"""


import os
import os.path
from pathlib import Path
from configobj import ConfigObj
from Utility.apex_utility import print_progress_bar
from Utility.apex_utility import copy_best_output
from Utility.apex_utility import read_summary
from pyCALAPEX import calAPEX
print('/014')

src_dir = Path(os.path.dirname(os.path.realpath(__file__)))
config = ConfigObj(str(src_dir / 'runtime.ini'))
site = 'Farm_1'
crops = None
scenario = 'grazing'
obs_attribute='runoff'
config['Site'] = site
config['Scenario'] = scenario
out_dir = 'Output/'+ obs_attribute
cluster_dir= f'/home/mahesh.maskey/SWMRU_APEX/OklahomaWRE/APEX_V2_output/'
# cluster_dir = '../Output'
sen_obj = calAPEX(src_dir, config=config, cluster_dir=cluster_dir,  site=site, scenario=scenario,
                  obs_attribute=obs_attribute, mod_attribute='WYLD', out_dir=out_dir)
print('Congratulation! Calibration done')

print('Importing the outputs within the criteria')


if scenario == 'non_grazing':
    scenario = 'pyAPEX_n'
else:
    scenario = 'pyAPEX_g'
    
cluster_dir= f'/home/mahesh.maskey/SWMRU_APEX/OklahomaWRE/APEX_V2_output/{site}/{scenario}/Output'
id_best_runs, local_dir = read_summary(file_name='best_stats.csv', obs_attribute='runoff', in_dir=None)


k = 0
print_progress_bar(k, len(id_best_runs), prefix='Complete', suffix='', decimals=1, length=50, fill='█')
for id_run in id_best_runs:
    copy_best_output(input_dir=cluster_dir, save_dir=local_dir, attribute='001RUN', iteration = id_run, extension= 'OUT', crop = None)
    copy_best_output(input_dir=cluster_dir, save_dir=local_dir, attribute='daily_outlet', iteration = id_run, extension= 'csv', crop = None)
    if crops is None:
        copy_best_output(input_dir=cluster_dir, save_dir=local_dir, attribute='daily_basin', iteration = id_run, extension= 'csv', crop = None)
        copy_best_output(input_dir=cluster_dir, save_dir=local_dir, attribute='annual', iteration = id_run, extension= 'csv', crop = None)
    else:
        for crop in crops:
            copy_best_output(input_dir=cluster_dir, save_dir=local_dir, attribute='daily_basin', iteration = id_run, extension= 'csv', crop = crop)
            copy_best_output(input_dir=cluster_dir, save_dir=local_dir, attribute='annual', iteration = id_run, extension= 'csv', crop = crop)      
    print_progress_bar(k, len(id_best_runs), prefix=f'{k}: {id_run}', suffix='Complete', decimals=1, length=50, fill='█')
    k = k + 1
