from utility import summarize_uncertainty_attributes

df_summary1n = summarize_uncertainty_attributes(site='Farm_1', scenario='non_grazing', obs_attribute='runoff')
df_summary1g = summarize_uncertainty_attributes(site='Farm_1', scenario='grazing', obs_attribute='runoff')
df_summary8n = summarize_uncertainty_attributes(site='Farm_8', scenario='non_grazing', obs_attribute='runoff')
df_summary8g = summarize_uncertainty_attributes(site='Farm_8', scenario='grazing', obs_attribute='runoff')