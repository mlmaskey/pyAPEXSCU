# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:25:04 2023

@author: Mahesh.Maskey
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import os
from utility import read_obs_data
from utility import get_plotdata

warnings.filterwarnings('ignore')

print('\014')
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)
# Farm 1  based on temporal optimization at different metrices
file_lists4farm1 = [
    '../post_analysis/Calibration/Farm_1/non_grazing/Soil_erosion_RUS2/daily_outlet_daily_PBIAS_daily.csv',
    '../post_analysis/Calibration/Farm_1/non_grazing/Soil_erosion_RUS2/daily_outlet_daily_PBIAS_monthly.csv',
    '../post_analysis/Calibration/Farm_1/non_grazing/Soil_erosion_RUS2/daily_outlet_daily_PBIAS_yearly.csv',
    '../post_analysis/Calibration/Farm_1/grazing/Soil_erosion_RUS2/daily_outlet_daily_PBIAS_daily.csv',
    '../post_analysis/Calibration/Farm_1/grazing/Soil_erosion_RUS2/daily_outlet_daily_PBIAS_monthly.csv',
    '../post_analysis/Calibration/Farm_1/grazing/Soil_erosion_RUS2/daily_outlet_daily_PBIAS_yearly.csv']

fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharey=True)
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[0], attribute='sediment', factor=0.5,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 0], legend=False)
axes[0, 0].set_xlabel('')
axes[0, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[1], attribute='sediment', factor=0.5,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 1], legend=False)
axes[0, 1].set_xlabel('')
axes[0, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[2], attribute='sediment', factor=0.5,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 2], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 2], legend=False)
axes[0, 2].set_xlabel('')
axes[0, 2].grid(True)
del cal_data, val_data

cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[0], attribute='sediment', factor=0.5,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 0], legend=False)
axes[1, 0].set_xlabel('')
axes[1, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[1], attribute='sediment', factor=0.5,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 1], legend=False)
axes[1, 1].set_xlabel('')
axes[1, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[2], attribute='sediment', factor=0.5,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 2], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 2], legend=False)
axes[1, 2].set_xlabel('')
axes[1, 2].grid(True)
del cal_data, val_data

cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[0], attribute='sediment', factor=0.5,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 0], legend=False)
axes[2, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[1], attribute='sediment', factor=0.5,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 1], legend=False)
axes[2, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[2], attribute='sediment', factor=0.5,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 2], legend='brief')
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 2], legend='brief')
del cal_data, val_data
axes[2, 2].grid(True)
axes[2, 2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Calibration_Soil_erosion_RUS2_Farm_1_non_grazing_correlation.png'),
            dpi=600, bbox_inches="tight")
plt.show()
plt.close('all')

fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharey=True)
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[3], attribute='sediment', factor=0.5,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 0], legend=False)
axes[0, 0].set_xlabel('')
axes[0, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[4], attribute='sediment', factor=0.5,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 1], legend=False)
axes[0, 1].set_xlabel('')
axes[0, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[5], attribute='sediment', factor=0.5,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 2], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 2], legend=False)
axes[0, 2].set_xlabel('')
axes[0, 2].grid(True)
del cal_data, val_data

cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[3], attribute='sediment', factor=0.5,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 0], legend=False)
axes[1, 0].set_xlabel('')
axes[1, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[4], attribute='sediment', factor=0.5,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 1], legend=False)
axes[1, 1].set_xlabel('')
axes[1, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[5], attribute='sediment', factor=0.5,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 2], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 2], legend=False)
axes[1, 2].set_xlabel('')
axes[1, 2].grid(True)
del cal_data, val_data

cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[3], attribute='sediment', factor=0.5,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 0], legend=False)
axes[2, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[4], attribute='sediment', factor=0.5,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 1], legend=False)
axes[2, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_1', file=file_lists4farm1[5], attribute='sediment', factor=0.5,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 2], legend='brief')
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 2], legend='brief')
del cal_data, val_data
axes[2, 2].grid(True)
axes[2, 2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Calibration_Soil_erosion_RUS2_Farm_1_grazing_correlation.png'),
            dpi=600, bbox_inches="tight")
plt.show()

# Farm 8  based on temporal optimization at different metrices
file_lists4farm8 = [
    '../post_analysis/Calibration/Farm_8/non_grazing/Soil_erosion_RUS2/daily_outlet_daily_COD_daily.csv',
    '../post_analysis/Calibration/Farm_8/non_grazing/Soil_erosion_RUS2/daily_outlet_daily_COD_monthly.csv',
    '../post_analysis/Calibration/Farm_8/non_grazing/Soil_erosion_RUS2/daily_outlet_daily_COD_yearly.csv',
    '../post_analysis/Calibration/Farm_8/grazing/Soil_erosion_RUS2/daily_outlet_daily_COD_daily.csv',
    '../post_analysis/Calibration/Farm_8/grazing/Soil_erosion_RUS2/daily_outlet_daily_COD_monthly.csv',
    '../post_analysis/Calibration/Farm_8/grazing/Soil_erosion_RUS2/daily_outlet_daily_COD_yearly.csv']

fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharey=True)
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[0], attribute='sediment', factor=1.0,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 0], legend=False)
axes[0, 0].set_xlabel('')
axes[0, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[1], attribute='sediment', factor=1.0,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 1], legend=False)
axes[0, 1].set_xlabel('')
axes[0, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[2], attribute='sediment', factor=1.0,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 2], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 2], legend=False)
axes[0, 2].set_xlabel('')
axes[0, 2].grid(True)
del cal_data, val_data

cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[0], attribute='sediment', factor=1.0,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 0], legend=False)
axes[1, 0].set_xlabel('')
axes[1, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[1], attribute='sediment', factor=1.0,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 1], legend=False)
axes[1, 1].set_xlabel('')
axes[1, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[2], attribute='sediment', factor=1.0,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 2], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 2], legend=False)
axes[1, 2].set_xlabel('')
axes[1, 2].grid(True)
del cal_data, val_data

cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[0], attribute='sediment', factor=1.0,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 0], legend=False)
axes[2, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[1], attribute='sediment', factor=1.0,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 1], legend=False)
axes[2, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[2], attribute='sediment', factor=1.0,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 2], legend='brief')
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 2], legend='brief')
del cal_data, val_data
axes[2, 2].grid(True)
axes[2, 2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Calibration_Soil_erosion_RUS2_Farm_8_non_grazing_correlation.png'),
            dpi=600, bbox_inches="tight")
plt.show()
plt.close('all')

fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharey=True)
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[3], attribute='sediment', factor=1.0,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 0], legend=False)
axes[0, 0].set_xlabel('')
axes[0, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[4], attribute='sediment', factor=1.0,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 1], legend=False)
axes[0, 1].set_xlabel('')
axes[0, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[5], attribute='sediment', factor=1.0,
                                  scale='daily')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[0, 2], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[0, 2], legend=False)
axes[0, 2].set_xlabel('')
axes[0, 2].grid(True)
del cal_data, val_data

cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[3], attribute='sediment', factor=1.0,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 0], legend=False)
axes[1, 0].set_xlabel('')
axes[1, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[4], attribute='sediment', factor=1.0,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 1], legend=False)
axes[1, 1].set_xlabel('')
axes[1, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[5], attribute='sediment', factor=1.0,
                                  scale='monthly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[1, 2], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[1, 2], legend=False)
axes[1, 2].set_xlabel('')
axes[1, 2].grid(True)
del cal_data, val_data

cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[3], attribute='sediment', factor=1.0,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 0], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 0], legend=False)
axes[2, 0].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[4], attribute='sediment', factor=1.0,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 1], legend=False)
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 1], legend=False)
axes[2, 1].grid(True)
del cal_data, val_data
cal_data, val_data = get_plotdata(site='Farm_8', file=file_lists4farm8[5], attribute='sediment', factor=1.0,
                                  scale='yearly')
sns.scatterplot(data=cal_data, x='Observed', y='Modeled', label='Calibration', ax=axes[2, 2], legend='brief')
sns.scatterplot(data=val_data, x='Observed', y='Modeled', label='Validation', ax=axes[2, 2], legend='brief')
del cal_data, val_data
axes[2, 2].grid(True)
axes[2, 2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Calibration_Soil_erosion_RUS2_Farm_8_grazing_correlation.png'),
            dpi=600, bbox_inches="tight")
plt.show()
