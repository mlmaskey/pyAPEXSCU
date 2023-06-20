# pyAPEXSCU
## Python Code for Calibrating, Senstiivy Analysis, and Uncertainty Analysis of Agriculture Policy Environmental Extender (APEX)
-- py: Python Code
-- APEX: Agriculture Policy Environmental Extender (APEX)
-- S: Senstiivy Analysis
-- C: Calibration
-- U: Uncertainty Analysis

**Authors**: *Mahesh Lal Maskey*, *Amanda M. Nelson*, *Haitao Huang*, and *Briain Stucky*

**Contributors**: *Daniel N. Moriasi*, and *Brian Northup*

## Introduction
## File Structure
* Main folder: **`pyAPEX`**
    * Preogram folder: `Program`
      * Default input files made for APEX program `APEXgraze.exe,`  containing `*.DAT`, `*.SIT`, `*.SUB`, `*.mgt`, `*.sol`, `*.WND`, `*.WPM`, `*.DLY, etc.
      * Calibration data file: `calibration_data.csv.`  
      * APEX editor spreadsheet: `xlsAPPXgraze??.xlsm.`        
    * Utility folder: `Utility` contining utility files used in the pre and post analysis of parameters, statsistics
 * Main files:
    * pyAPEXSCU.py: Main script built for calibration, senstivity and  uncertainty analysis
    * pyAPEXin.py: Contains class `inAPEX` that stores the `APEX` parameters from `Program` folder
    * pyCALAPEX.py: Contains class colled `calAPEX` devoted to calibration after running the program certain iterations specified in`runtime.ini` 
    * pySAAPEX.py: Contains class called `senanaAPEX` devoted to sensitivity analysis 
    * pyUAAPEX.py: Contains class called `unaAPEX` devoted to uncertainty analysis 
    * runtime.ini: Sets the runtime parameters like number of iterations, output location, range of parameter space.
    * calibration.py: Calls `Python` script `pyCALAPEX.py` for calibration after completing the iteration, including filtering parameter sets based on the `MORIASI CRITERIA` and finds best set
    * sensitivity_analysis.py: Calls `Python` script `pySAAPEX.py` for sensitivity analysis 
    * task_worker.py: Batch script used for jobs
    * uncertainty_analysis.py: Calls `Python` script `pyUAAPEX.py` for uncertainty analysis
## Utilities
   * The folder `post_scripts` should be independent of the `pyAPEX` folder and includes scripts for post processing, including making graphs, summarizing tables and so on.
   * If there is single project you cam merge these scripts with main folder `pyAPEX.` Otherwise, it is suggested to put in outside the project folder.
## File Organization
   * Create a project folder **APEX_project**
   * Create sub folders for different sites inside the prohect folders, e.g., `SITE1`, `SITE2` ... 
   * Create sub folders for each scenarios inside `SITE#` folder, e.g., 'pyAPEX_scn1`, 'pyAPEX_scn2`, ...
   * Copy main folder **`pyAPEX`** into each scenario folder simulatenously
   * Copy `post_scripts` folder into the main project folder
## Steps for (un) parallel computation
### Calibration
### Senstivity analysis
### Uncertainty analysis
### Usage of batch script
### Basic Linux Commands for Parallel Computation
* List the contents in the folder: `ls` if you are in; `ls [folder_path]` if you are outside the folder.
* Check the full path of folder you are in, use `pwd.`
* Read the content of a file: `cat [file_name]`
* Edit the content of a file: `nano [file_name]`
* List the numbers of jobs ran in the `SCINet (ATLAS or CERES)`: `squeue -u user_first_name.user_last_name.`
* Kill jobs recently ran in the `SCINet (ATLAS or CERES)`: `scancel -u user_first_name.user_last_name.`
* Count the number of file by specifif extenstion: `find [folder_path/] -name [file_name_with_extension] -printf '.' | wc -m~.`  For instance,  `find ./Output_*/ -name "daily_outlet_*.csv" -printf '.' | wc -m~` calculates the number of fliles with unique name containing `daily_outlet_` with extension `csv` inside the folders having prefix `Output_.` For this, one should above the folder contatining subfolders `Output_*.`
