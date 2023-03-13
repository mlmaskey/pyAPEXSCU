#!/bin/bash

#
# Launches parallel calibration runs of the APEX model.
#
# Author: Brian Stucky (USDA)
#
# commands before run this script：
#
# module load miniconda singularity
# source activate
# conda activate /project/swmru_apex/py_env
#
# the following line is optional. only for error 'wine: could not load kernel32.dll, status c0000135'
# export WINEPREFIX=/project/swmru_apex
#

#SBATCH --account=swmru_apex
#SBATCH --job-name="apex_sim"
#SBATCH --partition atlas
#SBATCH --array=1-500
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --time 10:00:00
##SBATCH --mail-user=emailAddress
##SBATCH --mail-type=BEGIN,END,FAIL

# The total number of calibration simulations to run.
NSIMS=200000

# The start simulation id
START_SIM_ID=1 

# The location for writing simulation outputs.
OUTPUT_PATH=/90daydata/swmru_apex/OklahomaWRE/Farm_1/pyAPEX_n/Output

# The loction of wine.sif
WINE_FILE=/project/swmru_apex/wine.sif

module load miniconda singularity
source activate
conda activate /project/swmru_apex/py_env

python pyAPEX/task_worker.py \
    --ntasks $SLURM_ARRAY_TASK_COUNT \
    --nsims $NSIMS \
    --taskidmin $SLURM_ARRAY_TASK_MIN \
    --taskid $SLURM_ARRAY_TASK_ID \
    --simidstart $START_SIM_ID \
    --outputdir $OUTPUT_PATH \
    --winepath $WINE_FILE


