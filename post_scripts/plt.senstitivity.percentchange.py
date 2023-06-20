# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:19:00 2023

@author: Mahesh.Maskey
"""

import os
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from utility import get_plot_heatmap

warnings.filterwarnings('ignore')
print('\014')
graph_dir = '..\post_analysis\Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)
dir_data = r'C:/Users/Mahesh.Maskey/Documents/Project/OklahomaWRE/post_analysis/sensitivity_analysis'
file_lists4farm = [r'C:/Users/Mahesh.Maskey/Documents/Project/OklahomaWRE/post_analysis/sensitivity_analysis/Farm_1'
                   r'/non_grazing/Heatmap-data_OF.csv',
                   r'C:/Users/Mahesh.Maskey/Documents/Project/OklahomaWRE/post_analysis/sensitivity_analysis/Farm_1'
                   r'/grazing/Heatmap-data_OF.csv',
                   r'C:/Users/Mahesh.Maskey/Documents/Project/OklahomaWRE/post_analysis/sensitivity_analysis/Farm_8'
                   r'/non_grazing/Heatmap-data_OF.csv',
                   r'C:/Users/Mahesh.Maskey/Documents/Project/OklahomaWRE/post_analysis/sensitivity_analysis/Farm_8'
                   r'/grazing/Heatmap-data_OF.csv']

fig, axes = plt.subplots(2, 2, figsize=(15, 15), tight_layout=True, sharex=True, sharey=True)
axes[0, 0] = get_plot_heatmap(filename=file_lists4farm[0], ax=axes[0, 0], custom_delta=-0.5, col_name='Percent_change',
                              cmap='BuPu', x_label=None, y_label=''r'$\Delta{\theta}$, %',
                              title_str='WRE1 without grazing', c_bar_title=''r'$\Delta{OF}$, %')

axes[1, 0] = get_plot_heatmap(filename=file_lists4farm[1], ax=axes[1, 0], custom_delta=-0.5, col_name='Percent_change',
                              cmap='BuPu', x_label='Parameters', y_label=''r'$\Delta{\theta}$, %',
                              title_str='WRE1 with grazing', c_bar_title=''r'$\Delta{OF}$, %')


axes[0, 1] = get_plot_heatmap(filename=file_lists4farm[2], ax=axes[0, 1], custom_delta=-0.5, col_name='Percent_change',
                              cmap='BuPu', x_label=None, y_label=None, title_str='WRE8 without grazing',
                              c_bar_title=''r'$\Delta{OF}$, %')

axes[1, 1] = get_plot_heatmap(filename=file_lists4farm[3], ax=axes[1, 1], custom_delta=-0.5, col_name='Percent_change',
                              cmap='BuPu', x_label='Parameters', y_label=None, title_str='WRE8 with grazing',
                              c_bar_title=''r'$\Delta{OF}$, %')

plt.savefig(os.path.join(graph_dir, 'Sensitivity_parameter_objective_function.png'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_parameter_objective_function.pdf'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_parameter_objective_function.jpeg'),
            dpi=600, bbox_inches="tight")
plt.show()

# Individual plots
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax = get_plot_heatmap(filename=file_lists4farm[0], ax=ax, custom_delta=-0.5, col_name='Percent_change', cmap='BuPu',
                      x_label='Parameters', y_label=''r'$\Delta{\theta}$, %', title_str='WRE1 without grazing',
                      c_bar_title=''r'$\Delta{OF}$, %')
plt.savefig(os.path.join(graph_dir, 'Farm_1_n_Sensitivity_parameter.png'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Farm_1_n_Sensitivity_parameter.pdf'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Farm_1_n_Sensitivity_parameter.jpeg'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax = get_plot_heatmap(filename=file_lists4farm[1], ax=ax, custom_delta=-0.5, col_name='Percent_change', cmap='BuPu',
                      x_label='Parameters',  y_label=''r'$\Delta{\theta}$, %', title_str='WRE1 with grazing',
                      c_bar_title=''r'$\Delta{OF}$, %')
plt.savefig(os.path.join(graph_dir, 'Farm_1_g_Sensitivity_parameter.png'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Farm_1_g_Sensitivity_parameter.pdf'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Farm_1_g_Sensitivity_parameter.jpeg'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax = get_plot_heatmap(filename=file_lists4farm[2], ax=ax, custom_delta=-0.5, col_name='Percent_change', cmap='BuPu',
                      x_label='Parameters',  y_label=''r'$\Delta{\theta}$, %', title_str='WRE8 without grazing',
                      c_bar_title=''r'$\Delta{OF}$, %')
plt.savefig(os.path.join(graph_dir, 'Farm_8_n_Sensitivity_parameter.png'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Farm_8_n_Sensitivity_parameter.pdf'), dpi=600,
            bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Farm_8_n_Sensitivity_parameter.jpeg'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax = get_plot_heatmap(filename=file_lists4farm[3], ax=ax, custom_delta=-0.5, col_name='Percent_change', cmap='BuPu',
                      x_label='Parameters', y_label=''r'$\Delta{\theta}$, %', title_str='WRE8 with grazing',
                      c_bar_title=''r'$\Delta{OF}$, %')
plt.savefig(os.path.join(graph_dir, 'Farm_8_g_Sensitivity_parameter.png'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Farm_8_g_Sensitivity_parameter.pdf'), dpi=600,
            bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Farm_8_g_Sensitivity_parameter.jpeg'),
            dpi=600, bbox_inches="tight")
plt.show()







