import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from utility import read_sensitive_params, normalize_dataframe, calculate_euclidean_distance
from utility import get_params
from utility import discretize_std
from utility import read_stats
from configobj import ConfigObj
import warnings

warnings.filterwarnings('ignore')
print('\014')


# noinspection PyShadowingNames
def extract_uncertainty_results(site, scenario):
    if scenario == 'non_grazing':
        folder_name = 'pyAPEX_n'
    elif scenario == 'grazing':
        folder_name = 'pyAPEX_g'
    else:
        print('Not applicable')
    config_file = f'../{site}/{folder_name}/pyAPEX/runtime.ini'
    sensitive_param_file = f'../{site}/{folder_name}/pyAPEX/Utility/sensitive.PAR'
    file_param = f'../{site}/{folder_name}/pyAPEX/OutputUncertainty/APEXPARM.csv'
    file_metric = f'../{site}/{folder_name}/pyAPEX/OutputUncertainty/Statistics_runoff.csv'
    config = ConfigObj(config_file)
    id_sensitive_param = read_sensitive_params(sensitive_param_file)
    df_params, p_best, _ = get_params(file_param, id_sensitive_param, is_all=False)
    df_metrics, df_best_metrics = read_stats(file_metric)
    df_best_metrics = pd.DataFrame(df_best_metrics).T
    df_best_metrics.index = df_best_metrics.RunId.astype(int)
    df_best_metrics.insert(1, 'PARAM', np.nan)
    disc_range = discretize_std(config)
    param_list = df_params.columns[1:]
    list_params = ['all'] + param_list.tolist()
    list_stats = ['RunId', 'PARAM', 'CODC', 'RMSEC', 'NRMSEC', 'NSEC', 'PBIASC', 'OF2C', 'CODV', 'RMSEV', 'NRMSEV',
                  'NSEV', 'PBIASV', 'OF2V']
    df_best_metrics.columns = list_stats
    df_best_metrics = df_best_metrics.drop('RunId', axis=1)
    df_best_metrics.insert(df_best_metrics.shape[1], 'Parameters', 'best')
    param_names = []
    for i in range(len(id_sensitive_param)):
        ids = id_sensitive_param[i] - 70
        param_names.append(f'PARAM [{ids}]')
    col_params = ['RunId'] + param_names
    df_params.columns = col_params
    df_results, df_correlations, df_correlation_matrix = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for i in range(len(param_names)):
        param = param_names[i]
        df_param = df_params[['RunId', param]]
        df_metric = df_metrics.iloc[:, 1:]
        df_final = pd.concat([df_param, df_metric], axis=1)
        df_final.columns = list_stats
        df_final = df_final.drop('RunId', axis=1)
        df_corr = df_final.corr(method='pearson')
        df_corr_ = pd.DataFrame(df_corr.iloc[:, 0])
        df_corr_.columns = [param]
        df_correlation_matrix = pd.concat([df_correlation_matrix, df_corr_], axis=1)
        df_final.insert(df_final.shape[1], 'Parameters', param)
        df_corr.insert(df_corr.shape[1], 'Parameters', param)
        df_results = pd.concat([df_results, df_final], axis=0)
        df_correlations = pd.concat([df_correlations, df_corr], axis=0)
    df_results = pd.concat([df_best_metrics, df_results], axis=0)
    df_results.insert(df_results.shape[1], 'Site', site)
    df_results.insert(df_results.shape[1], 'Operation', scenario)
    df_correlations.insert(df_correlations.shape[1], 'Site', site)
    df_correlations.insert(df_correlations.shape[1], 'Operation', scenario)
    df_correlation_matrix.insert(df_correlation_matrix.shape[1], 'Site', site)
    df_correlation_matrix.insert(df_correlation_matrix.shape[1], 'Operation', scenario)
    print(f'Analysis done for {site} under {scenario} operation')
    return df_results, df_correlations, df_correlation_matrix


df_result_WRE1n, df_correlation_WRE1n, df_correlation_matrix_WRE1n = extract_uncertainty_results(site='Farm_1',
                                                                                                 scenario='non_grazing')
df_result_WRE1g, df_correlation_WRE1g, df_correlation_matrix_WRE1g = extract_uncertainty_results(site='Farm_1',
                                                                                                 scenario='grazing')
df_result_WRE8n, df_correlation_WRE8n, df_correlation_matrix_WRE8n = extract_uncertainty_results(site='Farm_8',
                                                                                                 scenario='non_grazing')
df_result_WRE8g, df_correlation_WRE8g, df_correlation_matrix_WRE8g = extract_uncertainty_results(site='Farm_8',
                                                                                                 scenario='grazing')
df_result = pd.concat([df_result_WRE1n, df_result_WRE1g, df_result_WRE8n, df_result_WRE8g], axis=0)
df_correlation = pd.concat([df_correlation_WRE1n, df_correlation_WRE1g, df_correlation_WRE8n, df_correlation_WRE8g],
                           axis=0)
df_correlation_matrix = pd.concat(
    [df_correlation_matrix_WRE1n, df_correlation_matrix_WRE1g, df_correlation_matrix_WRE8n,
     df_correlation_matrix_WRE8g],
    axis=0)

save_data_dir = '..\post_analysis\Results'
if not os.path.isdir(save_data_dir):
    os.makedirs(save_data_dir)
graph_dir = '..\post_analysis\Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)

df_result.to_csv(os.path.join(save_data_dir, 'Uncertainty_Parameter_statistics.csv'), index=False)
df_correlation.to_csv(os.path.join(save_data_dir, 'Uncertainty_Parameter_correlations.csv'))
df_correlation_matrix.to_csv(os.path.join(save_data_dir, 'Uncertainty_Parameter_correlationmatrix.csv'))

param_names = df_correlation_matrix.columns[:-2]
list_new = ['OF2C', 'NSEC', 'CODC', 'PBIASC', 'OF2V', 'NSEV', 'CODV', 'PBIASV']
list_new_name = ['OFC', 'NSEC', 'CODC', 'PBIASC', 'OFV', 'NSEV', 'CODV', 'PBIASV']
# Correlation statistics
df_cor1n, df_cor1g = df_correlation_matrix_WRE1n[param_names], df_correlation_matrix_WRE1g[param_names]
df_cor8n, df_cor8g = df_correlation_matrix_WRE8n[param_names], df_correlation_matrix_WRE8g[param_names]
df_cor1n, df_cor1g = df_cor1n.T, df_cor1g.T
df_cor1n, df_cor1g = df_cor1n[list_new], df_cor1g[list_new]
df_cor8n, df_cor8g = df_cor8n.T, df_cor8g.T
df_cor8n, df_cor8g = df_cor8n[list_new], df_cor8g[list_new]
df_cor1n.columns, df_cor1g.columns = list_new_name, list_new_name
df_cor8n.columns, df_cor8g.columns = list_new_name, list_new_name
fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=True, sharey=True, tight_layout=True)
sns.heatmap(df_cor1n, cmap='Spectral', vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'}, ax=axes[0, 0])
axes[0, 0].set_title('WRE1: Grassland', fontsize=20)
axes[0, 0].set_ylabel('Parameters', fontsize=18)
axes[0, 0].tick_params(axis='both', which='major', labelsize=14)
sns.heatmap(df_cor1g, cmap='Spectral', vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'}, ax=axes[1, 0])
axes[1, 0].set_ylabel('Parameters', fontsize=18)
axes[1, 0].set_xlabel('Metrics', fontsize=18)
axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
sns.heatmap(df_cor8n, cmap='Spectral', vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'}, ax=axes[0, 1])
axes[0, 1].set_title('WRE8: Cropland', fontsize=20)
axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
sns.heatmap(df_cor8g, cmap='Spectral', vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'}, ax=axes[1, 1])
axes[1, 1].set_xlabel('Metrics', fontsize=18)
axes[1, 1].tick_params(axis='both', which='major', labelsize=14)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Uncertainty_heat_map_metrics.png'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_heat_map_metrics.pdf'), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Uncertainty_heat_map_metrics.jpeg'), dpi=600, bbox_inches="tight")
plt.show()

