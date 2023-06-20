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


def barplot_indices(data, ax, x_label=None, y_label=None, title_st=None, is_x_scale_log=False):
    sns.barplot(data=data, x='Sensitivity Index', y='Parameters', hue='Order', ax=ax, ci=None)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title_st, fontsize=18)
    if is_x_scale_log:
        ax.set_xscale('log')
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend_.remove()
    return ax


file_sensitivity_first = ['../post_analysis/sensitivity_analysis/Farm_1/non_grazing/Index_first_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_1/grazing/Index_first_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_8/non_grazing/Index_first_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_8/grazing/Index_first_OF.csv']

file_sensitivity_Total = ['../post_analysis/sensitivity_analysis/Farm_1/non_grazing/Index_Total_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_1/grazing/Index_Total_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_8/non_grazing/Index_Total_OF.csv',
                          '../post_analysis/sensitivity_analysis/Farm_8/grazing/Index_Total_OF.csv']

# sobol
df_sobol_f1_first_ng = gather_data_index(file_sensitivity_first[0], method='SOBOL')
df_sobol_f1_total_ng = gather_data_index(file_sensitivity_Total[0], method='SOBOL')

df_sobol_f1_first_g = gather_data_index(file_sensitivity_first[1], method='SOBOL')
df_sobol_f1_total_g = gather_data_index(file_sensitivity_Total[1], method='SOBOL')

df_sobol_f8_first_ng = gather_data_index(file_sensitivity_first[2], method='SOBOL')
df_sobol_f8_total_ng = gather_data_index(file_sensitivity_Total[2], method='SOBOL')

df_sobol_f8_first_g = gather_data_index(file_sensitivity_first[3], method='SOBOL')
df_sobol_f8_total_g = gather_data_index(file_sensitivity_Total[3], method='SOBOL')

df_sobol_f1_ng = pd.concat([df_sobol_f1_first_ng, df_sobol_f1_total_ng], axis=0)
df_sobol_f1_g = pd.concat([df_sobol_f1_first_g, df_sobol_f1_total_g], axis=0)
df_sobol_f8_ng = pd.concat([df_sobol_f8_first_ng, df_sobol_f8_total_ng], axis=0)
df_sobol_f8_g = pd.concat([df_sobol_f8_first_g, df_sobol_f8_total_g], axis=0)


fig, axes = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True, sharex=True, sharey=True)
axes[0, 0] = barplot_indices(data=df_sobol_f1_ng, x_label=None, y_label='Parameters', ax=axes[0, 0],
                             title_st='WRE1 without grazing')
axes[0, 1] = barplot_indices(data=df_sobol_f1_g, x_label=None, y_label=None, ax=axes[0, 1],
                             title_st='WRE1 with grazing')
axes[1, 0] = barplot_indices(data=df_sobol_f8_ng, x_label='Sensitivity Index', y_label='Parameters', ax=axes[1, 0],
                             title_st='WRE8 with grazing')
axes[1, 1] = barplot_indices(data=df_sobol_f8_g, x_label='Sensitivity Index', y_label=None, ax=axes[1, 1],
                             title_st='WRE8 with grazing')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_sobol.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_sobol.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_sobol.jpeg'), dpi=600, bbox_inches="tight")
plt.show()

# FAST
df_fast_f1_first_ng = gather_data_index(file_sensitivity_first[0], method='FAST')
df_fast_f1_total_ng = gather_data_index(file_sensitivity_Total[0], method='FAST')
df_fast_f1_first_g = gather_data_index(file_sensitivity_first[1], method='FAST')
df_fast_f1_total_g = gather_data_index(file_sensitivity_Total[1], method='FAST')

df_fast_f8_first_ng = gather_data_index(file_sensitivity_first[2], method='FAST')
df_fast_f8_total_ng = gather_data_index(file_sensitivity_Total[2], method='FAST')
df_fast_f8_first_g = gather_data_index(file_sensitivity_first[3], method='FAST')
df_fast_f8_total_g = gather_data_index(file_sensitivity_Total[3], method='FAST')

df_fast_f1_ng = pd.concat([df_fast_f1_first_ng, df_fast_f1_total_ng], axis=0)
df_fast_f1_g = pd.concat([df_fast_f1_first_g, df_fast_f1_total_g], axis=0)
df_fast_f8_ng = pd.concat([df_fast_f8_first_ng, df_fast_f8_total_ng], axis=0)
df_fast_f8_g = pd.concat([df_fast_f8_first_g, df_fast_f8_total_g], axis=0)

fig, axes = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True, sharex=True, sharey=True)
axes[0, 0] = barplot_indices(data=df_fast_f1_ng, x_label=None, y_label='Parameters', ax=axes[0, 0],
                             title_st='WRE1 without grazing', is_x_scale_log=True)
axes[0, 1] = barplot_indices(data=df_fast_f1_g, x_label=None, y_label=None, ax=axes[0, 1],
                             title_st='WRE1 with grazing', is_x_scale_log=True)
axes[1, 0] = barplot_indices(data=df_fast_f8_ng, x_label='Sensitivity Index', y_label='Parameters', ax=axes[1, 0],
                             title_st='WRE8 with grazing', is_x_scale_log=True)
axes[1, 1] = barplot_indices(data=df_fast_f8_g, x_label='Sensitivity Index', y_label=None, ax=axes[1, 1],
                             title_st='WRE8 with grazing', is_x_scale_log=True)
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_FAST.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_FAST.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_FAST.jpeg'), dpi=600, bbox_inches="tight")
plt.show()

# src

df_src_f1_first_ng = gather_data_index(file_sensitivity_first[0], method='SRC')
df_src_f1_total_ng = gather_data_index(file_sensitivity_Total[0], method='SRC')
df_src_f1_first_g = gather_data_index(file_sensitivity_first[1], method='SRC')
df_src_f1_total_g = gather_data_index(file_sensitivity_Total[1], method='SRC')

df_src_f8_first_ng = gather_data_index(file_sensitivity_first[2], method='SRC')
df_src_f8_total_ng = gather_data_index(file_sensitivity_Total[2], method='SRC')
df_src_f8_first_g = gather_data_index(file_sensitivity_first[3], method='SRC')
df_src_f8_total_g = gather_data_index(file_sensitivity_Total[3], method='SRC')

df_src_f1_ng = pd.concat([df_src_f1_first_ng, df_src_f1_total_ng], axis=0)
df_src_f1_g = pd.concat([df_src_f1_first_g, df_src_f1_total_g], axis=0)
df_src_f8_ng = pd.concat([df_src_f8_first_ng, df_src_f8_total_ng], axis=0)
df_src_f8_g = pd.concat([df_src_f8_first_g, df_src_f8_total_g], axis=0)

fig, axes = plt.subplots(2, 2, figsize=(10, 10), tight_layout=True, sharex=True, sharey=True)
axes[0, 0] = barplot_indices(data=df_src_f1_ng, x_label=None, y_label='Parameters', ax=axes[0, 0],
                             title_st='WRE1 without grazing', is_x_scale_log=False)
axes[0, 1] = barplot_indices(data=df_src_f1_g, x_label=None, y_label=None, ax=axes[0, 1],
                             title_st='WRE1 with grazing', is_x_scale_log=False)
axes[1, 0] = barplot_indices(data=df_src_f8_ng, x_label='Sensitivity Index', y_label='Parameters', ax=axes[1, 0],
                             title_st='WRE8 with grazing', is_x_scale_log=False)
axes[1, 1] = barplot_indices(data=df_src_f8_g, x_label='Sensitivity Index', y_label=None, ax=axes[1, 1],
                             title_st='WRE8 with grazing', is_x_scale_log=False)
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_SRC.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_SRC.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_barplot_SRC.jpeg'), dpi=600, bbox_inches="tight")
plt.show()