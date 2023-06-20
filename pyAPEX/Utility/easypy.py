# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:52:51 2022
Author: Mahesh Lal Maskey
Contact: mmaskey@ucdavis.edu/mahesh.maskey@usda.gov
"""
# print("\014") #ref: https://stackoverflow.com/questions/54943710/code-to-clear-console-and-variables-in-spyder

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
class easypy:
    @staticmethod
    def nanmatrix(nr:int, nc=None):
        '''
        Creates empty matrix with nan values
        Parameters
        ----------
        nr : int
            DESCRIPTION. Number of rows.
        nc : TYPE, optional
            DESCRIPTION. Number of columns. If  nc is not given it creates 
            square matrix with nan values
        Returns
        -------
        mat : numpy array
            DESCRIPTION. Empty matrix with nan values
        Example
        -------
        nanmat = easypy.nanmatrix(5, 3)
        '''
        if nc==None:
            nc = nr
            mat = np.zeros((nr, nc))
        else:
            mat = np.zeros((nr, nc))
        mat[:] = np.nan
        return mat
    
    def nansum(data:list):
        if isinstance(data,(pd.core.series.Series,np.ndarray)):
            data = data.tolist()         
        k = 0
        newdata = []
        for val in data:
            if np.isnan(val):
                k=k+1
            else:
                newdata.append(val)
        if k==0:
            sumdata = np.sum(data)
        else:
            sumdata = np.sum(newdata)                
        return sumdata

    
    def nanmean(data: list):
        '''
        Computes mean vaules of a list containing nan values
        Parameters
        ----------
        data : list
            DESCRIPTION. List containing numeric and nan values
        Returns
        -------
        mu : float
            DESCRIPTION. Returns arithmetc mean
        Example
        -------
        data = np.array([2, 8, np.nan, 1, 9])
        mu = easypy.nanmean(data)
        '''
        # Converts pandas series and numpy array into a list first
        if isinstance(data,(pd.core.series.Series,np.ndarray)):
            data = data.tolist()            
        k = 0
        newdata = []
        for val in data:
            if np.isnan(val):
                k=k+1
            else:
                newdata.append(val)
        if k==0:
            mu = np.mean(data)
        else:
            mu = np.mean(newdata)                
        return mu


    def nanmax(data: list):
        '''
        Computes maximum vaule of a list containing nan values
        Parameters
        ----------
        data : list
            DESCRIPTION. List containing numeric and nan values
        Returns
        -------
        x : float
            DESCRIPTION. Returns maximum value
        Example
        -------
        data = np.array([2, 8, np.nan, 1, 9])
        x = easypy.nanmax(data)
        '''
        # Converts pandas series and numpy array into a list first
        if isinstance(data,(pd.core.series.Series,np.ndarray)):
            data = data.tolist()            
        k = 0
        newdata = []
        for val in data:
            if np.isnan(val):
                k=k+1
            else:
                newdata.append(val)
        if k==0:
            x = np.max(data)
        else:
            x = np.max(newdata)                
        return x

    def nanmindata(data: list):
        '''
        Computes minimum vaules of a list containing nan values
        Parameters
        ----------
        data : list
            DESCRIPTION. List containing numeric and nan values
        Returns
        -------
        n : float
            DESCRIPTION. Returns minimum
        Example
        -------
        data = np.array([2, 8, np.nan, 1, 9])
        n = easypy.nanmin(data)
        '''
        # Converts pandas series and numpy array into a list first
        if isinstance(data,(pd.core.series.Series,np.ndarray)):
            data = data.tolist()            
        k = 0
        newdata = []
        for val in data:
            if np.isnan(val):
                k=k+1
            else:
                newdata.append(val)
        if k==0:
            n = np.min(data)
        else:
            n = np.min(newdata)                
        return n
    
    def nanstd(data: list):
        '''
        Computes standard deviation of a list containing nan values
        Parameters
        ----------
        data : list
            DESCRIPTION. List containing numeric and nan values
        Returns
        -------
        s : float
            DESCRIPTION. Returns standard deviation
        Example
        -------
        data = np.array([2, 8, np.nan, 1, 9])
        s = easypy.nanmean(data)
        '''
        # Converts pandas series and numpy array into a list first
        if isinstance(data,(pd.core.series.Series,np.ndarray)):
            data = data.tolist()            
        k = 0
        newdata = []
        for val in data:
            if np.isnan(val):
                k=k+1
            else:
                newdata.append(val)
        if k==0:
            s = np.std(data)
        else:
            s = np.std(newdata)                
        return s    
    
    
    def nancorr(x:list, y:list):
        '''
        Computes pearson correlation cofficient reardless of vectors 
        contaning NaN values
        Parameters
        ----------
        x : list
            DESCRIPTION. First vector
        y : list
            DESCRIPTION. Secon Vector

        Returns
        -------
        corrcoef : Pearson correlation coefficient R-square
            DESCRIPTION.
        Example
        ------
        x = [5.798, 1.185, 5.217, 8.222, 6.164, 4.118, 2.465, 6.663, 2.153, 0.205]
        y = [6.162, 6.27, 8.529, 0.127, 5.29, 1.443, 5.189, 8.244, 6.266, 9.582]
        easypy.nancorr(x, y)
        x = [57.8, 11.85, 52.17, 82.22, np.nan, 45.18, 24.65, np.nan, 21.53, 2.00]
        y = [61.62, 62.07, np.nan, 20.17, 51.29, 11.43, 52.19, 82.44, 62.66, 95.82]
        easypy.nancorr(x, y)

        '''
        if len(x)!=len(y):
            print(f'AssertionError: two vectors must be same length')
            return
        if isinstance(x,(pd.core.series.Series)):
            xnew = []
            for xv in x:
                xnew.append(xv[0])
        elif isinstance(y,(pd.core.series.Series)):
            ynew = []
            for yv in y:
                ynew.append(yv[0])
        else:
            xnew, ynew = x, y   
        if isinstance(x,(np.ndarray)):
            xnew = []
            for xv in x:
                xnew.append(xv)
        elif isinstance(y,(np.ndarray)):
            ynew = []
            for yv in y:
                ynew.append(yv)
        else:
            xnew, ynew = x, y             
        data = {'x': xnew, 'y': ynew}
        df = pd.DataFrame(data, index= range(len(x)))
        df = df.dropna()
        xnew1, ynew1 = df.x.to_numpy(), df.y.to_numpy()
        corr_matrix = np.corrcoef(xnew1, ynew1)
        corrcoef = corr_matrix[1, 0]**2
        return corrcoef
    
    def ranom_matrix (nr:int, nc:int):
        from Utility.easypy import easypy as ep
        import random
        a_mat = ep.nanmatrix(nr, nc)
        for i in range(nr):
            for j in range(nc):
                a_mat[i,j] = random.random()
        return a_mat

    def ranom_intmatrix (nr:int, nc:int, limit):
        from Utility.easypy import easypy as ep
        import random
        a_mat = ep.nanmatrix(nr, nc)
        for i in range(nr):
            for j in range(nc):
                a_mat[i,j] = random.randint(limit[0], limit[1])
        return a_mat
    
    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False   

    def get_outlier_na(x):
        mu = np.mean(x)
        sd = np.std(x)
        x[(x>mu+3*sd)|(x<mu-3*sd)] = np.nan
        return x
    
    def find_change_percent(x, xbest):
        change = (xbest-x)*100/xbest
        return change    

    def corr_test(X, Y):
        df = pd.DataFrame({'X': X.values, 'Y': Y.values})
        df = df.dropna()
        X, Y = df.X.values, df.Y.values
        r2_ = pearsonr(X, Y)
        r2 = r2_[0]
        p_value = r2_[1]
        return r2, p_value
