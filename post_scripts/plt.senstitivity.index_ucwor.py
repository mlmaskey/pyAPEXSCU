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
                          '../post_analysis/sensitivity_analysis/Farm_1/grazing/Index_first_OF.csv']

file_sensitivity_Total = ['../post_analysis/sensitivity_analysis/Farm_1/non_grazing/Index_Total_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_1/grazing/Index_Total_OF.csv']

# sobol
df_f1_first_ng_SOBOL = gather_data_index(file_sensitivity_first[0], method='SOBOL')
df_f1_total_ng_SOBOL = gather_data_index(file_sensitivity_Total[0], method='SOBOL')

df_f1_first_g_SOBOL = gather_data_index(file_sensitivity_first[1], method='SOBOL')
df_f1_total_g_SOBOL = gather_data_index(file_sensitivity_Total[1], method='SOBOL')

df_f1_first_ng_FAST = gather_data_index(file_sensitivity_first[0], method='FAST')
df_f1_total_ng_FAST = gather_data_index(file_sensitivity_Total[0], method='FAST')

df_f1_first_g_FAST = gather_data_index(file_sensitivity_first[1], method='FAST')
df_f1_total_g_FAST = gather_data_index(file_sensitivity_Total[1], method='FAST')

df_f1_first_ng_SRC = gather_data_index(file_sensitivity_first[0], method='SRC')
df_f1_total_ng_SRC = gather_data_index(file_sensitivity_Total[0], method='SRC')

df_f1_first_g_SRC = gather_data_index(file_sensitivity_first[1], method='SRC')
df_f1_total_g_SRC = gather_data_index(file_sensitivity_Total[1], method='SRC')

df_f1_ng_SOBOL = pd.concat([df_f1_first_ng_SOBOL, df_f1_total_ng_SOBOL], axis=0)
df_f1_g_SOBOL = pd.concat([df_f1_first_g_SOBOL, df_f1_total_g_SOBOL], axis=0)
df_f1_ng_FAST = pd.concat([df_f1_first_ng_FAST, df_f1_total_ng_FAST], axis=0)
df_f1_g_FAST = pd.concat([df_f1_first_g_FAST, df_f1_total_g_FAST], axis=0)
df_f1_ng_SRC = pd.concat([df_f1_first_ng_SRC, df_f1_total_ng_SRC], axis=0)
df_f1_g_SRC = pd.concat([df_f1_first_g_SRC, df_f1_total_g_SRC], axis=0)

df_f1_first_ng_SOBOL.insert(df_f1_first_ng_SOBOL.shape[1], 'Operation', 'Without grazing')
df_f1_first_g_SOBOL.insert(df_f1_first_g_SOBOL.shape[1], 'Operation', 'With grazing')
df_f1_first_SOBOL = pd.concat([df_f1_first_ng_SOBOL, df_f1_first_g_SOBOL], axis=0)

df_f1_total_ng_SOBOL.insert(df_f1_total_ng_SOBOL.shape[1], 'Operation', 'Without grazing')
df_f1_total_g_SOBOL.insert(df_f1_total_g_SOBOL.shape[1], 'Operation', 'With grazing')
df_f1_total_SOBOL = pd.concat([df_f1_total_ng_SOBOL, df_f1_total_g_SOBOL], axis=0)

df_f1_first_ng_FAST.insert(df_f1_first_ng_FAST.shape[1], 'Operation', 'Without grazing')
df_f1_first_g_FAST.insert(df_f1_first_g_FAST.shape[1], 'Operation', 'With grazing')
df_f1_first_FAST = pd.concat([df_f1_first_ng_FAST, df_f1_first_g_FAST], axis=0)

df_f1_total_ng_FAST.insert(df_f1_total_ng_FAST.shape[1], 'Operation', 'Without grazing')
df_f1_total_g_FAST.insert(df_f1_total_g_FAST.shape[1], 'Operation', 'With grazing')
df_f1_total_FAST = pd.concat([df_f1_total_ng_FAST, df_f1_total_g_FAST], axis=0)

df_f1_first_ng_SRC.insert(df_f1_first_ng_SRC.shape[1], 'Operation', 'Without grazing')
df_f1_first_g_SRC.insert(df_f1_first_g_SRC.shape[1], 'Operation', 'With grazing')
df_f1_first_SRC = pd.concat([df_f1_first_ng_SRC, df_f1_first_g_SRC], axis=0)

df_f1_total_ng_SRC.insert(df_f1_total_ng_SRC.shape[1], 'Operation', 'Without grazing')
df_f1_total_g_SRC.insert(df_f1_total_g_SRC.shape[1], 'Operation', 'With grazing')
df_f1_total_SRC = pd.concat([df_f1_total_ng_SRC, df_f1_total_g_SRC], axis=0)


fig, axes = plt.subplots(2, 3, figsize=(12, 10), tight_layout=True, sharex=False, sharey=True)
axes[0, 0] = sns.barplot(data=df_f1_first_SOBOL, x='Sensitivity Index', y='Parameters', hue='Operation', ax=axes[0, 0],
                         palette=['red', 'blue'])
axes[0, 0].set_title('SOBOL', fontsize=16)
axes[0, 0].set_xlabel('', fontsize=14)
axes[0, 0].set_xlim(-0.05, 0.6)
axes[0, 0].set_ylabel('First order', fontsize=14)
axes[0, 0].grid(True)
axes[0, 0].tick_params(axis='both', which='major', labelsize=12)
axes[0, 0].legend_.remove()

axes[1, 0] = sns.barplot(data=df_f1_total_SOBOL, x='Sensitivity Index', y='Parameters', hue='Operation', ax=axes[1, 0],
                         palette=['red', 'blue'])
axes[1, 0].set_xlabel('Indices', fontsize=14)
axes[1, 0].set_ylabel('Total effect', fontsize=14)
axes[1, 0].set_xlim(-0.05, 0.6)
axes[1, 0].grid(True)
axes[1, 0].tick_params(axis='both', which='major', labelsize=12)
axes[1, 0].legend_.remove()

axes[0, 1] = sns.barplot(data=df_f1_first_FAST, x='Sensitivity Index', y='Parameters', hue='Operation', ax=axes[0, 1],
                         palette=['red', 'blue'])
axes[0, 1].set_ylabel('', fontsize=14)
axes[0, 1].set_xlabel('', fontsize=14)
axes[0, 1].set_title('FAST', fontsize=16)
axes[0, 1].set_xlim(1E-4, 1)
axes[0, 1].grid(True)
axes[0, 1].tick_params(axis='both', which='major', labelsize=12)
axes[0, 1].legend_.remove()
axes[0, 1].set_xscale('log')

axes[1, 1] = sns.barplot(data=df_f1_total_FAST, x='Sensitivity Index', y='Parameters', hue='Operation', ax=axes[1, 1],
                         palette=['red', 'blue'])
axes[1, 1].set_ylabel('', fontsize=14)
axes[1, 1].set_xlabel('Indices', fontsize=14)
axes[1, 1].set_xlim(1E-4, 1)
axes[1, 1].grid(True)
axes[1, 1].tick_params(axis='both', which='major', labelsize=12)
axes[1, 1].legend_.remove()
axes[1, 1].set_xscale('log')

axes[0, 2] = sns.barplot(data=df_f1_first_SRC, x='Sensitivity Index', y='Parameters', hue='Operation', ax=axes[0, 2],
                         palette=['red', 'blue'])
axes[0, 2].set_title('SRC', fontsize=16)
axes[0, 2].set_ylabel('', fontsize=14)
axes[0, 2].set_xlabel('', fontsize=14)
axes[0, 2].grid(True)
axes[0, 2].tick_params(axis='both', which='major', labelsize=12)
axes[0, 2].legend_.remove()

axes[1, 2] = sns.barplot(data=df_f1_total_SRC, x='Sensitivity Index', y='Parameters', hue='Operation', ax=axes[1, 2],
                         palette=['red', 'blue'])
axes[1, 2] .set_ylabel('', fontsize=14)
axes[1, 2].set_xlabel('Coefficients', fontsize=14)
axes[1, 2] .grid(True)
axes[1, 2] .tick_params(axis='both', which='major', labelsize=12)
axes[1, 2] .legend_.remove()

axes[1, 2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)

plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'UCWOR_Sensitivity_barplot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'UCWOR_Sensitivity_barplot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'UCWOR_Sensitivity_barplot.jpeg'), dpi=600, bbox_inches="tight")
plt.show()
