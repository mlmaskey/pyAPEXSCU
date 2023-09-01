# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 02:52:20 2023

@author: Michael Aiyedun
"""
import os
import pandas as pd
import numpy as np
import re
import datetime as dt
import matplotlib.pyplot as plt

start = dt.date(1950, 1, 1)
end = dt.date(2099, 12, 31)


for folder in os.listdir("."):
    if os.path.isdir(folder):
        if (folder!='.vs' and 'Output' in folder ): # made to account for visual studio
                ann_df=pd.read_csv(folder+'/annual.csv')
                basin_df=pd.read_csv(folder+'/daily_basin.csv')
                outlet_df=pd.read_csv(folder+'/daily_outlet.csv')
                name=re.sub('Output_', '',folder)
                os.mkdir('graphs_'+name)
                
                prcpvbio=pd.DataFrame({
                    'Precipitation': basin_df['PRCP'],
                    'BIOMASS': basin_df['BIOM']})
                
                
                plt.title(name+'_BIOMASS')
                plt.xlabel('Date Interval')
                plt.ylabel('BIOMASS')
                
                plt.plot(prcpvbio.index,prcpvbio['BIOMASS'], label='bio')
                #plt.plot(prcpvbio.index,prcpvbio['Precipitation'], label='prec')
                   
                plt.savefig('graphs_'+name+'/'+'PRCPvBIOM_graph_'+name+'.png')
                plt.close()
               
               
               
                prcpvwyld=pd.DataFrame({
                    'Precipitation': basin_df['PRCP'],
                    'Wateryeild': basin_df['WYLD']})
                plt.title(name+'_Water_Yeild')
                plt.xlabel('Date Interval')
                plt.ylabel('Water Yeild')
                plt.bar(prcpvwyld.index, prcpvwyld['Wateryeild'])
                
                plt.savefig('graphs_'+name+'/'+'PRCPvWYLD_graph_'+name+'.png')
                plt.close()
