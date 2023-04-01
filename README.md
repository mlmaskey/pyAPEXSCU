# Python Code for Calibrating, Senstiivy Analysis, and Uncertainty Analysis of Agriculture Policy Environmental Extender (APEX)
-- py: Python Code
-- APEX: Agriculture Policy Environmental Extender (APEX)
-- S: Senstiivy Analysis
-- C: Calibration
-- U: Uncertainty Analysis

# Basic Linux Code for Parallel Computation
* To check the full path of folder you are in, use `pwd.`
* To list the numbers of jobs ran in the `SCINet (ATLAS or CERES)`: `squeue -u user_first_name.user_last_name.`
* To count the number of file by specifif extenstion: `find folder_path/ -name file_name_with_extension -printf '.' | wc -m~.`  For instance,  `find ./Output_*/ -name "daily_outlet_*.csv" -printf '.' | wc -m~` calculates the number of fliles with unique name containing `daily_outlet_` with extension `csv` inside the folders having prefix `Output_.` For this, one should above the folder contatining subfolders `Output_*.`
