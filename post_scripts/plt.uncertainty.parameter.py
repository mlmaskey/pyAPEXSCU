import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from utility import get_range, find_id
import os
from configobj import ConfigObj

warnings.filterwarnings('ignore')
print('\014')
graph_dir = '../post_analysis/Figures'
if not os.path.isdir(graph_dir):
    os.makedirs(graph_dir)


def read_sensitive_params(src_dir):
    with open(f'{src_dir}/Utility/sensitive.PAR') as f:
        line = f.read()
    f.close()
    l = line.split(',')
    id_sensitive = [int(item) for item in l]
    for i in range(len(id_sensitive)):
        id_sensitive[i] = id_sensitive[i] + 69
    return id_sensitive


def get_param_list(site, scenario):
    if scenario == 'non_grazing':
        sub_folder = 'pyAPEX_n'
    else:
        sub_folder = 'pyAPEX_g'
    src_dir = f'../{site}/{sub_folder}/pyAPEX'
    file_limits = f'{src_dir}/Utility/APEX_parameter_limits.csv'
    df_param_limit = pd.read_csv(file_limits, index_col=0, encoding="ISO-8859-1")
    sensitive_param_id = read_sensitive_params(src_dir)
    df_param_sensitive = df_param_limit.iloc[:, sensitive_param_id]
    param_list = df_param_sensitive.iloc[0, :].to_list()
    param_new_list = []
    for i in range(len(param_list)):
        param_ = param_list[i]
        list_str = param_.split('_')
        param_new = f'{list_str[0].upper()} [{list_str[1]}]'
        param_new_list.append(param_new)
    return param_list, param_new_list, sensitive_param_id


def get_param_stats(site, scenario):
    if scenario == 'non_grazing':
        sub_folder = 'pyAPEX_n'
    else:
        sub_folder = 'pyAPEX_g'
    main_dir = f'../{site}/{sub_folder}/pyAPEX'
    file_param = f'{main_dir}/OutputUncertainty/APEXPARM.csv'
    file_stat = f'{main_dir}/OutputUncertainty/Statistics_runoff.csv'
    df_param = pd.read_csv(file_param, index_col=0)
    # df_param.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
    # df_param.RunId = df_param.RunId.astype(int)
    best_param = df_param.iloc[0, :]
    df_param = df_param.iloc[1:, :]
    df_stats = pd.read_csv(file_stat, index_col=0)
    # df_stats.rename(columns={'Unnamed: 0': 'RunId'}, inplace=True)
    # df_stats.RunId = df_stats.RunId.astype(int)
    best_stats = df_stats.iloc[0, :]
    df_stats = df_stats.iloc[1:, :]

    return df_param, best_param, df_stats, best_stats


def compile_data(site, scenario, str_th):
    param_list_sensitive, param_new_list, sensitive_param_id = get_param_list(site, scenario)
    data_param, df_best_param, data_stats, df_best_stats = get_param_stats(site, scenario)
    df_params = data_param[param_list_sensitive]
    df_best_param = df_best_param[param_list_sensitive]
    df_params.columns = param_new_list
    df_best_param.index = param_new_list
    uncertainty_range = get_range(site, scenario, task="Uncertainty")
    id_lower, id_upper = find_id(uncertainty_range, -str_th), find_id(uncertainty_range, str_th)
    df_params_range = df_params.loc[id_lower:id_upper, :]
    df_stats_range = data_stats.loc[id_lower:id_upper, :]
    return df_params_range, df_stats_range, df_best_param, df_best_stats


def plot_matrix(site, pem, str_th, save_dir, is_individual):
    _, param_list, _ = get_param_list(site, scenario='non_grazing')
    df_params_1_n, df_stats_1_n, param_1_n, stats_1_n = compile_data(site, scenario='non_grazing', str_th=str_th)
    df_params_1_g, df_stats_1_g, param_1_g, stats_1_g = compile_data(site, scenario='grazing', str_th=str_th)

    id_vec = np.arange(0, len(param_list))
    id_matrix = id_vec.reshape(5, 4)
    fig, axes = plt.subplots(5, 4, figsize=(15, 10), sharex=False, sharey=True, tight_layout=True)
    for i in range(5):
        for j in range(4):
            param_id = id_matrix[i, j]
            param = param_list[param_id]
            df_plot_data_n = pd.DataFrame({'X': df_params_1_n[param].values, 'Y': df_stats_1_n[pem].values},
                                          index=df_params_1_n.index)
            df_plot_data_g = pd.DataFrame({'X': df_params_1_g[param].values, 'Y': df_stats_1_g[pem].values},
                                          index=df_params_1_g.index)
            ax = axes[i, j]
            ax.scatter(df_plot_data_n.X.values, df_plot_data_n.Y.values, c='r', marker='.', s=2,
                       label='Without grazing')
            ax.scatter(param_1_n[param_id], stats_1_n[pem], c='k', marker='o', s=50)
            ax.scatter(df_plot_data_g.X.values, df_plot_data_g.Y.values, c='b', marker='.', s=2, label='With grazing')
            ax.scatter(param_1_g[param_id], stats_1_g[pem], c='k', marker='+', s=50)
            ax.set_xlabel(param, fontsize=14)
            ax.set_ylabel('')
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{site}_Parameters_{pem}.png'), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f'{site}_Parameters_{pem}.jpeg'), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, f'{site}_Parameters_{pem}.pdf'), dpi=600, bbox_inches="tight")

    if is_individual:
        save_dir_i = f'{save_dir}/{site}'
        if not os.path.isdir(save_dir_i):
            os.makedirs(save_dir_i)
        # plotting individual
        for i in range(len(id_vec)):
            param_id = id_vec[i]
            param = param_list[param_id]
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            df_plot_data_n = pd.DataFrame({'X': df_params_1_n[param].values, 'Y': df_stats_1_n[pem].values},
                                          index=df_params_1_n.index)
            df_plot_data_g = pd.DataFrame({'X': df_params_1_g[param].values, 'Y': df_stats_1_g[pem].values},
                                          index=df_params_1_g.index)
            ax.scatter(df_plot_data_n.X.values, df_plot_data_n.Y.values, c='r', marker='.', s=2, label='Without grazing')
            ax.scatter(param_1_n[param_id], stats_1_n[pem], c='k', marker='o', s=50)
            ax.scatter(df_plot_data_g.X.values, df_plot_data_g.Y.values, c='b', marker='.', s=2, label='With grazing')
            ax.scatter(param_1_g[param_id], stats_1_g[pem], c='k', marker='+', s=50)
            ax.set_xlabel(param, fontsize=14)
            ax.set_ylabel('Objective function', fontsize=14)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(True)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            plt.savefig(os.path.join(save_dir_i, f'{site}_{param}_{pem}.png'), dpi=600, bbox_inches="tight")
            plt.savefig(os.path.join(save_dir_i, f'{site}_{param}_{pem}.pdf'), dpi=600, bbox_inches="tight")
            plt.savefig(os.path.join(save_dir_i, f'{site}_{param}_{pem}.jpeg'), dpi=600, bbox_inches="tight")


plot_matrix(site='Farm_1', pem='OF2DC', str_th=1.0, save_dir=graph_dir, is_individual=True)
plot_matrix(site='Farm_8', pem='OF2DC', str_th=1.0, save_dir=graph_dir, is_individual=True)