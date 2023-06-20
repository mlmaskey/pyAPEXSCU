from typing import List

import pandas as pd
import os
import warnings
from utility import get_calibrated_result
from utilis_parallel_coordinates import df_parallel_coordinates
warnings.filterwarnings('ignore')

list_stats: List[str] = ['OF', 'NSE', 'COD', 'PBIAS']

# choose attributes below
# ['PRCP', 'WYLD', 'YSD', 'RUS2', 'ET', 'PET', 'DPRK', 'BIOM', 'STD', 'STDL', 'STL', 'YLDF', 'YLDG', 'TN', 'TP',
#  'LAI', 'WS', 'TS', 'NS', 'PS', 'Mode']

graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)

_Farm_1_non_OF = get_calibrated_result(site='Farm_1', scenario='non_grazing', obs_attribute='runoff', metric='OF')
_Farm_1_non_PBIAS = get_calibrated_result(site='Farm_1', scenario='non_grazing', obs_attribute='runoff', metric='PBIAS')
_Farm_1_non_COD = get_calibrated_result(site='Farm_1', scenario='non_grazing', obs_attribute='runoff', metric='COD')
_Farm_1_non_NSE = get_calibrated_result(site='Farm_1', scenario='non_grazing', obs_attribute='runoff', metric='NSE')
df_Farm_1_non = pd.concat([_Farm_1_non_OF[0], _Farm_1_non_NSE[0], _Farm_1_non_COD[0], _Farm_1_non_PBIAS[0]], axis=0)
df_Farm_1_non.reset_index(inplace=True)
df_Farm_1_non.drop('index', axis=1, inplace=True)
df_Farm_1_non.insert(df_Farm_1_non.shape[1], 'PEM', list_stats)
plt = df_parallel_coordinates(df_Farm_1_non, cols=['WYLD', 'YSD', 'ET', 'PET', 'DPRK', 'BIOM', 'YLDF', 'TN', 'TP'],
                              colors=['blue', 'green', 'red', 'black'], labels=list_stats, fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_n_basic_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_n_basic_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_n_basic_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")

_Farm_1_grazing_OF = get_calibrated_result(site='Farm_1', scenario='grazing', obs_attribute='runoff', metric='OF')
_Farm_1_grazing_PBIAS = get_calibrated_result(site='Farm_1', scenario='grazing', obs_attribute='runoff', metric='PBIAS')
_Farm_1_grazing_COD = get_calibrated_result(site='Farm_1', scenario='grazing', obs_attribute='runoff', metric='COD')
_Farm_1_grazing_NSE = get_calibrated_result(site='Farm_1', scenario='grazing', obs_attribute='runoff', metric='NSE')
df_Farm_1_graze = pd.concat([_Farm_1_grazing_OF[0], _Farm_1_grazing_NSE[0], _Farm_1_grazing_COD[0],
                             _Farm_1_grazing_PBIAS[0]], axis=0)
df_Farm_1_graze.reset_index(inplace=True)
df_Farm_1_graze.drop('index', axis=1, inplace=True)
df_Farm_1_graze.insert(df_Farm_1_graze.shape[1], 'PEM', list_stats)
plt = df_parallel_coordinates(df_Farm_1_graze, cols=['WYLD', 'YSD', 'ET', 'PET', 'DPRK', 'BIOM', 'YLDF', 'TN', 'TP'],
                              colors=['blue', 'green', 'red', 'black'], labels=list_stats, fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_g_basic_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_g_basic_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_1_g_basic_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")

_Farm_8_non_OF = get_calibrated_result(site='Farm_8', scenario='non_grazing', obs_attribute='runoff', metric='OF')
_Farm_8_non_PBIAS = get_calibrated_result(site='Farm_8', scenario='non_grazing', obs_attribute='runoff', metric='PBIAS')
_Farm_8_non_COD = get_calibrated_result(site='Farm_8', scenario='non_grazing', obs_attribute='runoff', metric='COD')
_Farm_8_non_NSE = get_calibrated_result(site='Farm_8', scenario='non_grazing', obs_attribute='runoff', metric='NSE')
df_Farm_8_non = pd.concat([_Farm_8_non_OF[0], _Farm_8_non_NSE[0], _Farm_8_non_COD[0], _Farm_8_non_PBIAS[0]], axis=0)
df_Farm_8_non.reset_index(inplace=True)
df_Farm_8_non.drop('index', axis=1, inplace=True)
df_Farm_8_non.insert(df_Farm_8_non.shape[1], 'PEM', list_stats)
plt = df_parallel_coordinates(df_Farm_8_non, cols=['WYLD', 'YSD', 'ET', 'PET', 'DPRK', 'BIOM', 'YLDG', 'TN', 'TP'],
                              colors=['blue', 'green', 'red', 'black'], labels=list_stats, fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_n_basic_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_n_basic_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_n_basic_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")

_Farm_8_grazing_OF = get_calibrated_result(site='Farm_8', scenario='grazing', obs_attribute='runoff', metric='OF')
_Farm_8_grazing_PBIAS = get_calibrated_result(site='Farm_8', scenario='grazing', obs_attribute='runoff', metric='PBIAS')
_Farm_8_grazing_COD = get_calibrated_result(site='Farm_8', scenario='grazing', obs_attribute='runoff', metric='COD')
_Farm_8_grazing_NSE = get_calibrated_result(site='Farm_8', scenario='grazing', obs_attribute='runoff', metric='NSE')
df_Farm_8_graze = pd.concat([_Farm_8_grazing_OF[0], _Farm_8_grazing_NSE[0], _Farm_8_grazing_COD[0],
                             _Farm_8_grazing_PBIAS[0]],    axis=0)
df_Farm_8_graze.reset_index(inplace=True)
df_Farm_8_graze.drop('index', axis=1, inplace=True)
df_Farm_8_graze.insert(df_Farm_8_graze.shape[1], 'PEM', list_stats)
plt = df_parallel_coordinates(df_Farm_8_graze, cols=['WYLD', 'YSD', 'ET', 'PET', 'DPRK', 'BIOM', 'YLDG', 'TN', 'TP'],
                              colors=['blue', 'green', 'red', 'black'], labels=list_stats, fig_size=(12, 6))
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_g_basic_parallel_plot.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_g_basic_parallel_plot.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Calibration_Farm_8_g_basic_parallel_plot.jpeg'), dpi=600, bbox_inches="tight")
