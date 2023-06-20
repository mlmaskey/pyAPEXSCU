import pandas as pd
from utility import get_calibrated_result
metric = 'PBIAS'

data_cal_WRE1_n, _, _ = get_calibrated_result(site='Farm_1',scenario='non_grazing', obs_attribute='runoff',
                                              metric=metric)
data_cal_WRE1_g, _, _ = get_calibrated_result(site='Farm_1',scenario='grazing', obs_attribute='runoff',
                                              metric=metric)
data_cal_WRE8_n, _, _ = get_calibrated_result(site='Farm_8',scenario='non_grazing', obs_attribute='runoff',
                                              metric=metric)
data_cal_WRE8_g, _, _ = get_calibrated_result(site='Farm_8',scenario='grazing', obs_attribute='runoff', metric=metric)

data_summary = pd.concat([data_cal_WRE1_n, data_cal_WRE1_g, data_cal_WRE8_n, data_cal_WRE8_g])
data_summary.to_csv(f'../post_analysis/Results/Calibration_all_{metric}.csv')
