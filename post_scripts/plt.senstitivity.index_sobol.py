# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 18:25:24 2023

@author: Mahesh.Maskey
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
print('\014')
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)


def gather_data_index(file, method):
    data = pd.read_csv(file)
    data.rename(columns={'Unnamed: 0': 'Parameters'}, inplace=True)
    data_new = data[data.Method == method]
    return data_new


file_sensitivity_first = ['../post_analysis/sensitivity_analysis/Farm_1/non_grazing/Index_first_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_1/grazing/Index_first_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_8/non_grazing/Index_first_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_8/grazing/Index_first_OF.csv']

file_sensitivity_Total = ['../post_analysis/sensitivity_analysis/Farm_1/non_grazing/Index_Total_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_1/grazing/Index_Total_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_8/non_grazing/Index_Total_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_8/grazing/Index_Total_OF.csv']

# sobol
df_f1_first_ng = gather_data_index(file_sensitivity_first[0], method='SOBOL')
df_f1_total_ng = gather_data_index(file_sensitivity_Total[0], method='SOBOL')

df_f1_first_g = gather_data_index(file_sensitivity_first[1], method='SOBOL')
df_f1_total_g = gather_data_index(file_sensitivity_Total[1], method='SOBOL')

df_f8_first_ng = gather_data_index(file_sensitivity_first[2], method='SOBOL')
df_f8_total_ng = gather_data_index(file_sensitivity_Total[2], method='SOBOL')

df_f8_first_g = gather_data_index(file_sensitivity_first[3], method='SOBOL')
df_f8_total_g = gather_data_index(file_sensitivity_Total[3], method='SOBOL')

df_f1_ng = pd.concat([df_f1_first_ng, df_f1_total_ng], axis=0)
df_f1_g = pd.concat([df_f1_first_g, df_f1_total_g], axis=0)
df_f8_ng = pd.concat([df_f8_first_ng, df_f8_total_ng], axis=0)
df_f8_g = pd.concat([df_f8_first_g, df_f8_total_g], axis=0)

df_f1_first_ng.insert(df_f1_first_ng.shape[1], 'Operation', 'Without grazing')
df_f1_first_g.insert(df_f1_first_g.shape[1], 'Operation', 'With grazing')
df_f1_first = pd.concat([df_f1_first_ng, df_f1_first_g], axis=0)

df_f1_total_ng.insert(df_f1_total_ng.shape[1], 'Operation', 'Without grazing')
df_f1_total_g.insert(df_f1_total_g.shape[1], 'Operation', 'With grazing')
df_f1_total = pd.concat([df_f1_total_ng, df_f1_total_g], axis=0)

df_f8_first_ng.insert(df_f8_first_ng.shape[1], 'Operation', 'Without grazing')
df_f8_first_g.insert(df_f8_first_g.shape[1], 'Operation', 'With grazing')
df_f8_first = pd.concat([df_f8_first_ng, df_f8_first_g], axis=0)

df_f8_total_ng.insert(df_f8_total_ng.shape[1], 'Operation', 'Without grazing')
df_f8_total_g.insert(df_f8_total_g.shape[1], 'Operation', 'With grazing')
df_f8_total = pd.concat([df_f8_total_ng, df_f8_total_g], axis=0)


fig, axes = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True, sharex=True, sharey=True)

axes[0, 0] = sns.barplot(data=df_f1_first, x='Sensitivity Index', y='Parameters', hue='Operation', ax=axes[0, 0],
                         palette=['red', 'blue'])
axes[0, 0].set_title('WRE1: Grassland', fontsize=16)
axes[0, 0].set_xlabel('SOBOL (First)', fontsize=14)
axes[0, 0].set_ylabel('Parameters', fontsize=14)
axes[0, 0].grid(True)
axes[0, 0].tick_params(axis='both', which='major', labelsize=12)
axes[0, 0].legend_.remove()

axes[1, 0] = sns.barplot(data=df_f1_total, x='Sensitivity Index', y='Parameters', hue='Operation', ax=axes[1, 0],
                         palette=['red', 'blue'])
axes[1, 0].set_xlabel('SOBOL (Total)', fontsize=14)
axes[1, 0].set_ylabel('Parameters', fontsize=14)
axes[1, 0].grid(True)
axes[1, 0].tick_params(axis='both', which='major', labelsize=12)
axes[1, 0].legend_.remove()

axes[0, 1] = sns.barplot(data=df_f8_first, x='Sensitivity Index', y='Parameters', hue='Operation', ax=axes[0, 1],
                         palette=['red', 'blue'])
axes[0, 1].set_xlabel('SOBOL (First)', fontsize=14)
axes[0, 1].set_ylabel('', fontsize=14)
axes[0, 1].set_title('WRE8: Cropland', fontsize=16)
axes[0, 1].grid(True)
axes[0, 1].tick_params(axis='both', which='major', labelsize=12)
axes[0, 1].legend_.remove()

axes[1, 1] = sns.barplot(data=df_f8_total, x='Sensitivity Index', y='Parameters', hue='Operation', ax=axes[1, 1],
                         palette=['red', 'blue'])
axes[1, 1].set_xlabel('SOBOL (Total)', fontsize=14)
axes[1, 1].set_ylabel('', fontsize=14)
axes[1, 1].grid(True)
axes[1, 1].tick_params(axis='both', which='major', labelsize=12)
axes[1, 1].legend_.remove()

axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_sobol.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_sobol.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_sobol.jpeg'), dpi=600, bbox_inches="tight")
plt.show()
