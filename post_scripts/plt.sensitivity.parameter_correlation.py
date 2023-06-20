import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from utility import read_sensitive_params, normalize_dataframe, calculate_euclidean_distance
from utility import get_params
from utility import read_stats
from utility import discretize_ids
from configobj import ConfigObj
import warnings

warnings.filterwarnings('ignore')
print('\014')


def extract_sensitivity_results(site, scenario):
    if scenario == 'non_grazing':
        folder_name = 'pyAPEX_n'
    elif scenario == 'grazing':
        folder_name = 'pyAPEX_g'
    else:
        print('Not applicable')
    config_file = f'../{site}/{folder_name}/pyAPEX/runtime.ini'
    sensitive_param_file = f'../{site}/{folder_name}/pyAPEX/Utility/sensitive.PAR'
    file_param = f'../{site}/{folder_name}/pyAPEX/OutputSensitivity/APEXPARM.csv'
    file_metric = f'../{site}/{folder_name}/pyAPEX/OutputSensitivity/Statistics_runoff.csv'
    config = ConfigObj(config_file)
    id_sensitive_param = read_sensitive_params(sensitive_param_file)
    df_params, p_best, _ = get_params(file_param, id_sensitive_param, is_all=False)
    df_metrics, df_best_metrics = read_stats(file_metric)
    df_best_metrics = pd.DataFrame(df_best_metrics).T
    df_best_metrics.index = df_best_metrics.RunId.astype(int)
    df_best_metrics.insert(1, 'PARAM', np.nan)
    n_param = len(id_sensitive_param)
    disc_range, id_starts, id_ends = discretize_ids(config, n_param)
    param_list = df_params.columns[1:]
    list_params = ['all'] + param_list.tolist()
    list_stats = ['RunId', 'PARAM', 'CODC', 'RMSEC', 'NRMSEC', 'NSEC', 'PBIASC', 'OF2C', 'CODV', 'RMSEV', 'NRMSEV',
                  'NSEV', 'PBIASV', 'OF2V']
    df_best_metrics.columns = list_stats
    df_best_metrics = df_best_metrics.drop('RunId', axis=1)
    df_best_metrics.insert(df_best_metrics.shape[1], 'Parameters', 'best')
    df_results, df_correlations = pd.DataFrame(), pd.DataFrame()
    for i in range(len(list_params)):
        param = list_params[i]
        if param == 'all':
            df_param = df_params.iloc[:id_ends[i], :]
            df_metric = df_metrics.iloc[:id_ends[i], :]
            df_param_normalized = normalize_dataframe(df_param, params=df_param.columns[1:])
            norms = calculate_euclidean_distance(df_param_normalized.iloc[:, 1:])
            df_param_all = pd.DataFrame({'RunId': df_param_normalized.RunId, 'PARAM': norms})
            df_final = pd.concat([df_param_all, df_metric.iloc[:, 1:]], axis=1)
            df_final.columns = list_stats
            df_final = df_final.drop('RunId', axis=1)
            df_corr = df_final.corr(method='pearson')
        else:
            df_param = df_params.iloc[id_starts[i] - 1:id_ends[i], :]
            df_param = df_param[['RunId', param]]
            df_metric = df_metrics.iloc[id_starts[i] - 1:id_ends[i], 1:]
            df_final = pd.concat([df_param, df_metric], axis=1)
            df_final.columns = list_stats
            df_final = df_final.drop('RunId', axis=1)
            df_corr = df_final.corr(method='pearson')
        df_final.insert(df_final.shape[1], 'Parameters', param)
        df_corr.insert(df_corr.shape[1], 'Parameters', param)
        df_results = pd.concat([df_results, df_final], axis=0)
        df_correlations = pd.concat([df_correlations, df_corr], axis=0)
    df_results = pd.concat([df_best_metrics, df_results], axis=0)
    df_results.insert(df_results.shape[1], 'Site', site)
    df_results.insert(df_results.shape[1], 'Operation', scenario)
    df_correlations.insert(df_correlations.shape[1], 'Site', site)
    df_correlations.insert(df_correlations.shape[1], 'Operation', scenario)
    print(f'Analysis done for {site} under {scenario} operation')
    return df_results, df_correlations


df_result_WRE1n, df_correlation_WRE1n = extract_sensitivity_results(site='Farm_1', scenario='non_grazing')
df_result_WRE1g, df_correlation_WRE1g = extract_sensitivity_results(site='Farm_1', scenario='grazing')
df_result_WRE8n, df_correlation_WRE8n = extract_sensitivity_results(site='Farm_8', scenario='non_grazing')
df_result_WRE8g, df_correlation_WRE8g = extract_sensitivity_results(site='Farm_8', scenario='grazing')
df_result = pd.concat([df_result_WRE1n, df_result_WRE1g, df_result_WRE8n, df_result_WRE8g], axis=0)
df_correlation = pd.concat([df_correlation_WRE1n, df_correlation_WRE1g, df_correlation_WRE8n, df_correlation_WRE8g],
                           axis=0)

save_data_dir = '..\post_analysis\Results'
if not os.path.isdir(save_data_dir):
    os.makedirs(save_data_dir)
graph_dir = '..\post_analysis\Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)

df_result.to_csv(os.path.join(save_data_dir, 'Parameter_statistics.csv'), index=False)
df_correlation.to_csv(os.path.join(save_data_dir, 'Parameter_correlations.csv'))

file_sensitive_param = f'../Farm_1/pyAPEX_n/pyAPEX/Utility/sensitive.PAR'
id_sensitive = read_sensitive_params(file_sensitive_param)
param_names = []
for i in range(len(id_sensitive)):
    ids = id_sensitive[i] - 70
    param_names.append(f'PARAM [{ids}]')
param_names = ['ALL'] + param_names
df_result_new = df_result[df_result.Parameters != 'best']
df_result_new = df_result_new[['CODC', 'NSEC', 'PBIASC', 'OF2C', 'CODV', 'NSEV', 'PBIASV', 'OF2V', 'Parameters', 'Site',
                               'Operation']]
params = df_result_new.Parameters.unique()

for i in range(len(param_names)):
    df_result_new.Parameters[df_result_new.Parameters == params[i]] = param_names[i]
df_result_new.Operation[df_result_new.Operation == 'non_grazing'] = 'Without grazing'
df_result_new.Operation[df_result_new.Operation == 'grazing'] = 'With grazing'

df_result_new1 = df_result_new[df_result_new.Site == 'Farm_1']
df_result_new8 = df_result_new[df_result_new.Site == 'Farm_8']

best_results = df_result[df_result.Parameters == 'best']
best_results.Operation[best_results.Operation == 'non_grazing'] = 'Without grazing'
best_results.Operation[best_results.Operation == 'grazing'] = 'With grazing'

best_results1 = best_results[best_results.Site == 'Farm_1']
best_results8 = best_results[best_results.Site == 'Farm_8']

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
sns.boxplot(data=df_result_new1, x='Parameters', y='OF2C', hue='Operation', fliersize=False, ax=axes[0])
axes[0].axhline(best_results1.OF2C.values[0], ls='--', color='black', label='Without grazing')
axes[0].axhline(best_results1.OF2C.values[1], ls='--', color='blue', label='With grazing')
axes[0].set_xlabel('Parameters', fontsize=14)
axes[0].set_ylabel('Objective function value', fontsize=14)
axes[0].set_title('WRE1: grassland', fontsize=16)
axes[0].tick_params(axis='both', which='major', labelsize=12)
axes[0].tick_params(axis='x', labelrotation=45)
axes[0].legend_.remove()

sns.boxplot(data=df_result_new8, x='Parameters', y='OF2C', hue='Operation', fliersize=False, ax=axes[1])
axes[1].axhline(best_results8.OF2C.values[0], ls='--', color='black', label='Without grazing')
axes[1].axhline(best_results8.OF2C.values[1], ls='--', color='blue', label='With grazing')
axes[1].tick_params(axis='x', labelrotation=45)
axes[1].set_xlabel('Parameters', fontsize=14)
axes[1].set_ylabel('Objective function value', fontsize=16)
axes[1].set_title('WRE8: cropland', fontsize=14)
axes[1].tick_params(axis='both', which='major', labelsize=12)
axes[1].legend_.remove()
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Sensitivity_Box_plot_OF.png'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_Box_plot_OF.pdf'), dpi=600,
            bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_Box_plot_OF.jpeg'),
            dpi=600, bbox_inches="tight")
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
sns.boxplot(data=df_result_new1, x='Parameters', y='NSEC', hue='Operation', fliersize=False, ax=axes[0])
axes[0].axhline(best_results1.NSEC.values[0], ls='--', color='black', label='Without grazing')
axes[0].axhline(best_results1.NSEC.values[1], ls='--', color='blue', label='With grazing')
axes[0].set_xlabel('Parameters', fontsize=14)
axes[0].set_ylabel('NSE', fontsize=14)
axes[0].set_title('WRE1: grassland', fontsize=16)
axes[0].tick_params(axis='both', which='major', labelsize=12)
axes[0].tick_params(axis='x', labelrotation=45)
axes[0].legend_.remove()

sns.boxplot(data=df_result_new8, x='Parameters', y='NSEC', hue='Operation', fliersize=False, ax=axes[1])
axes[1].axhline(best_results8.NSEC.values[0], ls='--', color='black', label='Without grazing')
axes[1].axhline(best_results8.NSEC.values[1], ls='--', color='blue', label='With grazing')
axes[1].tick_params(axis='x', labelrotation=45)
axes[1].set_xlabel('Parameters', fontsize=14)
axes[1].set_ylabel('NSE', fontsize=14)
axes[1].set_title('WRE8: cropland', fontsize=16)
axes[1].tick_params(axis='both', which='major', labelsize=12)
axes[1].legend_.remove()
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.30), ncol=6)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Sensitivity_Box_plot_NSE.png'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_Box_plot_NSE.pdf'), dpi=600,
            bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_Box_plot_NSE.jpeg'),
            dpi=600, bbox_inches="tight")
plt.show()

# Correlation statistics
df_cor_WRE1n = df_correlation_WRE1n[['PARAM', 'Parameters', 'Site', 'Operation']]
vec_corr = df_cor_WRE1n.PARAM.values
mat_corr1n = vec_corr.reshape(21, 13)
df_cor1n = pd.DataFrame(mat_corr1n, index=param_names, columns=['PARAM', 'CODC', 'RMSEC', 'NRMSEC', 'NSEC', 'PBIASC',
                                                                'OF2C', 'CODV', 'RMSEV', 'NRMSEV', 'NSEV', 'PBIASV',
                                                                'OF2V'])
df_cor1n = df_cor1n[['CODC', 'NSEC', 'PBIASC', 'OF2C', 'CODV', 'NSEV', 'PBIASV', 'OF2V']]

df_cor_WRE1g = df_correlation_WRE1g[['PARAM', 'Parameters', 'Site', 'Operation']]
vec_corr = df_cor_WRE1g.PARAM.values
mat_corr1g = vec_corr.reshape(21, 13)
df_cor1g = pd.DataFrame(mat_corr1g, index=param_names, columns=['PARAM', 'CODC', 'RMSEC', 'NRMSEC', 'NSEC', 'PBIASC',
                                                                'OF2C', 'CODV', 'RMSEV', 'NRMSEV', 'NSEV', 'PBIASV',
                                                                'OF2V'])
df_cor1g = df_cor1g[['CODC', 'NSEC', 'PBIASC', 'OF2C', 'CODV', 'NSEV', 'PBIASV', 'OF2V']]

df_cor_WRE8n = df_correlation_WRE8n[['PARAM', 'Parameters', 'Site', 'Operation']]
vec_corr = df_cor_WRE8n.PARAM.values
mat_corr8n = vec_corr.reshape(21, 13)
df_cor8n = pd.DataFrame(mat_corr8n, index=param_names, columns=['PARAM', 'CODC', 'RMSEC', 'NRMSEC', 'NSEC', 'PBIASC',
                                                                'OF2C', 'CODV', 'RMSEV', 'NRMSEV', 'NSEV', 'PBIASV',
                                                                'OF2V'])
df_cor8n = df_cor8n[['CODC', 'NSEC', 'PBIASC', 'OF2C', 'CODV', 'NSEV', 'PBIASV', 'OF2V']]

df_cor_WRE8g = df_correlation_WRE8g[['PARAM', 'Parameters', 'Site', 'Operation']]
vec_corr = df_cor_WRE8g.PARAM.values
mat_corr8g = vec_corr.reshape(21, 13)
df_cor8g = pd.DataFrame(mat_corr8g, index=param_names, columns=['PARAM', 'CODC', 'RMSEC', 'NRMSEC', 'NSEC', 'PBIASC',
                                                                'OF2C', 'CODV', 'RMSEV', 'NRMSEV', 'NSEV', 'PBIASV',
                                                                'OF2V'])
df_cor8g = df_cor8g[['CODC', 'NSEC', 'PBIASC', 'OF2C', 'CODV', 'NSEV', 'PBIASV', 'OF2V']]

fig, axes = plt.subplots(2, 2, figsize=(16, 16), sharex=True, sharey=True)
sns.heatmap(df_cor1n, cmap='Spectral', vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'}, ax=axes[0, 0])
axes[0, 0].set_title('WRE1 (Without grazing)', fontsize=16)
axes[0, 0].set_ylabel('Parameters', fontsize=14)
axes[0, 0].tick_params(axis='both', which='major', labelsize=12)
sns.heatmap(df_cor1g, cmap='Spectral', vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'}, ax=axes[1, 0])
axes[1, 0].set_title('WRE1 (With grazing)', fontsize=16)
axes[1, 0].set_ylabel('Parameters', fontsize=14)
axes[1, 0].set_xlabel('Metrics', fontsize=14)
axes[1, 0].tick_params(axis='both', which='major', labelsize=12)
sns.heatmap(df_cor8n, cmap='Spectral', vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'}, ax=axes[0, 1])
axes[0, 1].set_title('WRE8 (Without grazing)', fontsize=16)
axes[0, 1].tick_params(axis='both', which='major', labelsize=12)
sns.heatmap(df_cor8g, cmap='Spectral', vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'}, ax=axes[1, 1])
axes[1, 1].set_title('WRE8 (With grazing)', fontsize=16)
axes[1, 1].set_xlabel('Metrics', fontsize=14)
axes[1, 1].tick_params(axis='both', which='major', labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'Sensitivity_heat_map_metrics.png'),
            dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_heat_map_metrics.pdf'), dpi=600,
            bbox_inches="tight")
plt.savefig(os.path.join(graph_dir, 'Sensitivity_heat_map_metrics.jpeg'),
            dpi=600, bbox_inches="tight")
plt.show()
