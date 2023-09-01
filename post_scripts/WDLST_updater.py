# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 01:38:17 2023

@author: Michael Aiyedun
"""
import os
import re

"""
This script is built around converting data taken from 
https://gdo-dcp.ucllnl.org/downscaled_cmip_projections/dcpInterface.html#Projections:%20Subset%20Request and the given file format
This script is to update the WDLYLIST.DAT file for the APEXGraze program. Copy 
The WDLYLIST.DAT file that is created to the ApexGraze file location
Note: this script makes a new WDLYLIST.DAT for the specific folders that are in
the same directory of this script so keep that in mind 
(WDLST_updater.ply is here)[name of data folder]/[loca5]/MetaData.txt
"""
geolocation =input('do all the data files have the same Elevation?\nPlease answer yes or no: ')
if(geolocation.lower()=='yes' or geolocation.lower()=='y'):

    
    #lat=input('Enter the Latitude: ')
    #lon=input('Enter the Longitude: ')
    
#Elevation isn't stored in the MetaData so this value has to be given before hand(?)
    evelat=input('Enter the Elevation: ') 
    
    f_w =open('WDLYLIST.DAT','w')
    f_w.close()
    del f_w
    ID = 1
    
    for folder in os.listdir("."):
        if os.path.isdir(folder):
            if (folder!='.vs'): # made to account for visual studio
                
                dlyname=folder+'_weather.dly'
                """
MetaData.txt is the only consistent file (with the same file name) from this data format that 
https://gdo-dcp.ucllnl.org/downscaled_cmip_projections/dcpInterface.html#Projections:%20Subset%20Request 
provides between all the files to find the Latitude and Longitude data values              
                """
                fil=open(folder+'/loca5/MetaData.txt')
                metadata=fil.readlines()
                location=metadata[13]
                location1=re.sub("[Location:             (]", '',location)
                location2=re.sub("[])\n]", '',location1)
                location3=re.sub("[ ]", '',location2)
                latlon=location3.split(',')
                lat =latlon[0]
                lon=latlon[1]
                fil.close()
                
                wdlist = '   ' + str(ID) +'   '+ dlyname +'   ' + lat + '   ' + lon + '   '+ evelat + '   '+folder
                f_a=open('WDLYLIST.DAT','a')
                f_a.writelines(wdlist+'\n')
                ID+=1
                f_a.close()
    print('WDLYLIST.DAT created. \n')
    
else:# just assume if not yes/y then it's no/n

    
    f_w =open('WDLYLIST.DAT','w')
    f_w.close()
    del f_w
    ID = 1
    
    for folder in os.listdir("."):
        if os.path.isdir(folder):
            if (folder!='.vs'): # made to account for visual studio
                dlyname=folder+'_weather.dly'
                
                
                #lat=input('Enter the Latitude: ')
                #lon=input('Enter the Longitude: ')
                
                fil=open(folder+'/loca5/MetaData.txt') 
                metadata=fil.readlines()
                location=metadata[13]
                location1=re.sub("[Location:             (]", '',location)
                location2=re.sub("[])\n]", '',location1)
                location3=re.sub("[ ]", '',location2)
                latlon=location3.split(',')
                lat =latlon[0]
                lon=latlon[1]
                fil.close()
 
                """
if the scale of just entering  the Elevation everytime is to much on a large scale project 
edit this section to read a file that contains all the evevation in order of the files where this script is placed
                """               
 
                evelat=input('Enter the Elevation: ') 

                wdlist = '   ' + str(ID) +'   '+ dlyname +'   ' + lat + '   ' + lon + '   '+ evelat + '   '+folder
                f_a=open('WDLYLIST.DAT','a')
                f_a.writelines(wdlist+'\n')
                ID+=1
                f_a.close()
    print('WDLYLIST.DAT created. \n')    


            