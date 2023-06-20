# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 07:42:35 2022

@author: Mahesh.Maskey
"""


import os
import os.path
from pathlib import Path
from configobj import ConfigObj
from pySAAPEX import senanaAPEX
print('\014')


src_dir = Path(os.path.dirname(os.path.realpath(__file__)))
config = ConfigObj(str(src_dir / 'runtime.ini'))
site = 'Farm_1'
scenario = 'grazing'
config['Site'] = site
config['Scenario'] = scenario
out_dir = f'../../../post_analysis/sensitivity_analysis/{site}/{scenario}'
sen_obj = senanaAPEX(src_dir, config, out_dir, attribute='runoff', metric='OF')