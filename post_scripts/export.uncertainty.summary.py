import pandas as pd
from utility import get_calibrated_result
from utility import summarize_uncertainty_data


def make_data_4_pc(site, scenario, obs_attribute, metric):
    data_calibrated, df_c, df_s = get_calibrated_result(site, scenario, obs_attribute, metric)
    data_calibrated.insert(data_calibrated.shape[1], 'Mode', 'Calibrated')
    sd_data, all_data, selected_data = summarize_uncertainty_data(site, scenario, obs_attribute)
    data_calibrated = data_calibrated[selected_data.columns]
    plot_data = pd.concat([selected_data, data_calibrated], axis=0)
    plot_data.to_csv(f'../post_analysis/Results/Uncertainty_{site}_{scenario}_{metric}.csv')
    return plot_data, (sd_data, all_data, selected_data), (data_calibrated, df_c, df_s)


set_farm_1_n = make_data_4_pc(site='Farm_1', scenario='non_grazing', obs_attribute='runoff', metric='OF')
set_farm_1_g = make_data_4_pc(site='Farm_1', scenario='grazing', obs_attribute='runoff', metric='OF')
set_farm_8_n = make_data_4_pc(site='Farm_8', scenario='non_grazing', obs_attribute='runoff', metric='OF')
set_farm_8_g = make_data_4_pc(site='Farm_8', scenario='grazing', obs_attribute='runoff', metric='OF')
