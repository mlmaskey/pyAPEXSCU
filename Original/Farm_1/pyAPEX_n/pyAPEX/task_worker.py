# -*- coding: utf-8 -*-
"""
Manages APEX model runs in the context of a Slurm task that is running
concurrently with (potentially many) other model run tasks.  Expects to receive
an ID range that uniquely identifies these model runs from the model runs in
all other Slurm tasks.

@author: Brian Stucky (USDA), Mahesh.Maskey
"""
import os
import os.path
from pathlib import Path
import shutil
import argparse
from pyAPEXin import inAPEX
# from pyAPEXSCU import simAPEX
from pyAPEXSCU import simAPEX
from configobj import ConfigObj
import numpy as np

print('\014')
parser = argparse.ArgumentParser()
parser.add_argument(
    '--ntasks', type=int, required=True,
    help='The total number of worker tasks in the modeling run.'
)
parser.add_argument(
    '--nsims', type=int, required=True,
    help='The total number of simulations to run across all worker tasks.'
)
parser.add_argument(
    '--taskidmin', type=int, required=True, help='The Start ID of all worker tasks.'
)
parser.add_argument(
    '--taskid', type=int, required=True, help='The ID of this worker task.'
)
parser.add_argument(
    '--simidstart', type=int, required=True,
    help='The start simulation id to run.'
)
parser.add_argument(
    '--outputdir', type=str, required=True,
    help='The model run and output location.'
)
parser.add_argument(
    '--winepath', type=str, required=True,
    help='The location of an executable Wine container image.'    
)
parser.add_argument(
    '--id_mode', type=int, required=True,
    help='Type of simulation.'    
)

args = parser.parse_args()

# Get the location of the simulation software and output dir.
src_dir = Path(os.path.dirname(os.path.realpath(__file__)))
outputdir = Path(args.outputdir)
    
# Calculate the simulation ID range for this worker task.
if args.nsims < args.ntasks:
    raise Exception(
        f'Incompatible simulation and task counts: {args.nsims}, {args.ntasks}.'
    )

if args.ntasks > 999:
    raise Exception(
        f'Task count too large.'
    )
#Select the model mode 0: calibration, 1: sensitivity, 2: uncertainty
#id_mode = 2
# evenly distribute the sim ids to each task
sim_per_task = np.ceil(args.nsims / args.ntasks).astype(int)
id_mode = args.id_mode
z = np.zeros((sim_per_task, args.ntasks))  # zero pads
sim_ids = np.arange(args.simidstart, args.simidstart + args.nsims)
sim_ids.resize(z.shape, refcheck=False)
# sim_ids = np.zeros(z.shape)

id  = args.simidstart
for c in range(args.ntasks):
    for r in range(sim_per_task):
        if sim_ids[r, c] > 0:
            sim_ids[r, c] = id
            id += 1

sim_id_min = sim_ids[0, args.taskid - args.taskidmin] - 1
sim_id_max = sim_ids[:, args.taskid - args.taskidmin].max().astype(int)

# Load and adjust the configuration settings.
config = ConfigObj(str(src_dir / 'runtime.ini'))
model_mode = ['calibration', 'sensitivity', 'uncertainty']
config['n_start'] = sim_id_min
config['n_simulation'] = sim_id_max

# Make a unique copy of the simulation code for this task, if needed.
if id_mode == 0:
    task_progdir = outputdir / f'Program_{args.taskid:03}'
    if not(task_progdir.is_dir()): 
        shutil.copytree(src_dir / 'Program', task_progdir)
else:
    task_progdir = src_dir/'Program'  

# Run the simulations.
# isall_try = True
isall_try = False
curr_directory = os.getcwd()
os.chdir(task_progdir)
input_APEX = inAPEX(config, task_progdir)
sim_APEX = simAPEX(
    config, src_dir, args.winepath, input_APEX, model_mode[id_mode],
    scale='daily', isall=isall_try
)
os.chdir(curr_directory)
