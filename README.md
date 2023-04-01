# Python Code for Calibrating, Senstiivy Analysis, and Uncertainty Analysis of Agriculture Policy Environmental Extender (APEX)
-- py: Python Code
-- APEX: Agriculture Policy Environmental Extender (APEX)
-- S: Senstiivy Analysis
-- C: Calibration
-- U: Uncertainty Analysis

# Basic Linux Commands for Parallel Computation
* List the contents in the folder: `ls` if you are in; `ls [folder_path]` if you are outside the folder.
* Check the full path of folder you are in, use `pwd.`
* Read the content of a file: `cat [file_name]`
* Edit the content of a file: `nano [file_name]`
* List the numbers of jobs ran in the `SCINet (ATLAS or CERES)`: `squeue -u user_first_name.user_last_name.`
* Kill jobs recently ran in the `SCINet (ATLAS or CERES)`: `scancel -u user_first_name.user_last_name.`
* Count the number of file by specifif extenstion: `find [folder_path/] -name [file_name_with_extension] -printf '.' | wc -m~.`  For instance,  `find ./Output_*/ -name "daily_outlet_*.csv" -printf '.' | wc -m~` calculates the number of fliles with unique name containing `daily_outlet_` with extension `csv` inside the folders having prefix `Output_.` For this, one should above the folder contatining subfolders `Output_*.`
