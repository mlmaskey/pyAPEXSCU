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
import matplotlib.dates as mdates

#start = dt.date(1950, 1, 1)
#end = dt.date(2099, 12, 31)


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
                
                Bdate_df=pd.DataFrame()
                for i in range(len(basin_df)):
                    Bdate_df.loc[i,'Date']=dt.date(basin_df.loc[i,'Y'], basin_df.loc[i,'M'], basin_df.loc[i,'D'])
                
                
                plt.title(name+'_BIOMASS')
                plt.xlabel('Date Interval')
                plt.ylabel('BIOMASS')
                
                
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y'))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3650)) 
                #plt.gca().set_xbound(start, end)
                
                ymin=min(prcpvbio['BIOMASS'])
                ymax=max(prcpvbio['BIOMASS'])
                plt.ylim(ymin, 1.05 * ymax)
                plt.rcParams['figure.figsize'] = [40,40]
                plt.plot(Bdate_df['Date'],prcpvbio['BIOMASS'])
                #plt.bar(Bdate_df['Date'],prcpvbio['BIOMASS'])
                #plt.plot(prcpvbio.index,prcpvbio['Precipitation'], label='prec')
                   
                plt.savefig('graphs_'+name+'/'+'PRCPvBIOM_graph_'+name+'.png')
                plt.close()
               
               
               
                prcpvwyld=pd.DataFrame({
                    'Precipitation': basin_df['PRCP'],
                    'Wateryeild': basin_df['WYLD']})
                
                Odate_df=pd.DataFrame()
                for i in range(len(outlet_df)):
                    Odate_df.loc[i,'Date']=dt.date(basin_df.loc[i,'Y'], basin_df.loc[i,'M'], basin_df.loc[i,'D'])
                
                plt.title(name+'_Water_Yeild')
                plt.xlabel('Date Interval')
                plt.ylabel('Water Yeild')
                
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y'))
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3650)) 
                #plt.gca().set_xbound(start, end)
                
                
                ymin=min(prcpvwyld['Wateryeild'])
                ymax=max(prcpvwyld['Wateryeild'])
                plt.ylim(ymin, 1.05 * ymax)
                plt.rcParams['figure.figsize'] = [40,20]
                plt.bar(Odate_df['Date'],prcpvwyld['Wateryeild'])
                #plt.plot(Odate_df['Date'],prcpvwyld['Wateryeild'])
                plt.savefig('graphs_'+name+'/'+'PRCPvWYLD_graph_'+name+'.png')
                plt.close()
