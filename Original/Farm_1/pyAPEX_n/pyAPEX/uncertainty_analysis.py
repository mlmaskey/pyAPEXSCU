# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 07:42:35 2022

@author: Mahesh.Maskey
"""


import os
import os.path
from pathlib import Path
from configobj import ConfigObj
from pyUAAPEX import unaAPEX
print('\014')

cluster_dir = 'G:/PostDocResearch/USDA-ARS/Project/OklahomaWRE/Cluster_data/'
src_dir = Path(os.path.dirname(os.path.realpath(__file__)))
config = ConfigObj(str(src_dir / 'runtime.ini'))
site = 'Farm_1'
scenario = 'non_grazing'
obs_attribute='runoff'
out_dir = f'../../../post_analysis/Uncertainty_analysis/{site}/{scenario}/{obs_attribute}'
scale = 'daily'
mod_attribute = 'WYLD'
una_obj = unaAPEX(src_dir, config, cluster_dir, site, scenario, scale, obs_attribute,mod_attribute, out_dir)