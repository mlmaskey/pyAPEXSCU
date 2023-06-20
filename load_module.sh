#!/bin/bash

#SBATCH --account=swmru_apex
#SBATCH --job-name="python_module"
#SBATCH --partition atlas
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=1
#SBATCH --time 20:00:00
##SBATCH --mail-user=emailAddress
##SBATCH --mail-type=BEGIN,END,FAIL

module load miniconda singularity
source activate
conda activate /project/swmru_apex/py_env

python calibration.py
