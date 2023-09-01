# -*- coding: utf-8 -*-
"""
Created on Thu Jul 6 20:55:35 2023

@author: Michael Aiyedun
"""
import os
import pandas as pd
import numpy as np
import fortranformat as ff

        
"""
This script is built around converting data taken from https://gdo-dcp.ucllnl.org/downscaled_cmip_projections/dcpInterface.html#Projections:%20Subset%20Request and the given file format
ftp://gdo-dcp.ucllnl.org/pub/dcp/subset/<job_identifier>/ to access the specific data
The variables this script is looking for are Tasmax (temperature max at surface), Tasmin (temperature min at surface), and Pr (Precipitation) 
This script needs to be located in the same folder that holds the data's folder (as taken from the link) 
The package fortranformat needs to be intsalled via pip or conda command for this to work
If there is an error please check your data to see for potential incompatable data types in the csv files and change/remove them
"""

#Weather.dly format for the Apex graze model in fortran
write_format = ff.FortranRecordWriter('(i6,i4,i4,a6,f6.1,f6.1,f6.2)') 


for folder in os.listdir("."):
    if os.path.isdir(folder):
        if (folder!='.vs'): # made to account for visual studio
            """
the script goes through current folder the script is in to get data. 
folders that don't contain the data/don't match the data structure will need to be moved or edited to match the format.
(DLY_import.ply is here)[name of data folder]/[loca5]/tasmax'or'tasmin'or'pr.csv
            """

            tmaxdir=folder+'/loca5/tasmax.csv'
            tmindir=folder+'/loca5/tasmin.csv'
            prdir=folder+'/loca5/pr.csv'
            dlyname=folder+'_weather.dly'

            #creates blank file
            f_w = open(dlyname, "w")
            f_w.close()
            del f_w
        
            dftmax= pd.read_csv(tmaxdir, header=None)
            dftmax = dftmax.drop(4, axis=1)

            dftmin= pd.read_csv(tmindir, header=None)
            mincol=dftmin[4]
            dftmin = pd.DataFrame({'5':mincol})


            dfpr= pd.read_csv(prdir, header=None)
            prcol=dfpr[4]
            dfpr = pd.DataFrame({'6':prcol})

            df = pd.concat([dftmax,dftmin,dfpr], axis=1)
            #alt Dataframe to mirror ApexGRaze model
            altdf =pd.DataFrame({
                    'YEAR':df[0],
                    'MONTH':df[1],
                    'DAY':df[2],
                    'SRAD':'',
                    'TMAX':df[3],
                    'TMIN':df['5'],
                    'PRCP':df['6']
                    })
            # removed and adjusted due to NaN in SRAD issue
            #altdf.to_csv('temp.csv', index=None)
            #adf=pd.read_csv('temp.csv')
            #print(adf)

            for i in range(altdf.shape[0]):
                f_a =open(dlyname,'a')
                info = altdf.iloc[i,:]
                info =write_format.write(info) # formats the data frame to fortran
                f_a.writelines(info+'\n')
                f_a.close()
            print(dlyname+' created. \n') # move the new .dly files to location of where the APEXGraze reads them



