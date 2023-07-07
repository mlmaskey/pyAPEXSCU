from pyAPEX import inAPEX
from pyAPEX import simAPEX
from pathlib import Path
import argparse
import shutil
import os.path
import warnings
# python run_APEX.py --type=Scenario --case_id=1 --winepath='' --scen_name=BASE
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument(
    '--type', type=str, required=True,
    help='Type of analysis, e.g., Scenario.'
)
parser.add_argument(
    '--case_id', type=int, required=True,
    help='Case for calibrated parameters or base case. 1: Objective function, '
         '2:Nash-Sutcliffe Efficiency, 3:Coefficient of Determination, 4:Percent Bias'
)
parser.add_argument(
    '--winepath', type=str, required=True,
    help='The location of an executable Wine container image.'
)
parser.add_argument(
    '--scen_name', type=str, required=True,
    help='Scenario name, like climate model.'
)


args = parser.parse_args()
output_dir = f'../{args.type}'
task_name = args.scen_name

task_prog_dir = os.path.join(output_dir, f'Program_{task_name}')
src_dir = Path(os.path.dirname(os.path.realpath(__file__)))
curr_directory = os.getcwd()

if not os.path.isdir(task_prog_dir):
    shutil.copytree(src_dir / 'Program', task_prog_dir)
else:
    print(f'deleting folder {task_prog_dir}')
    files_in_dir = os.listdir(task_prog_dir)
    for file in files_in_dir:  # loop to delete each file in folder
        os.remove(f'{task_prog_dir}/{file}')  # delete file
    os.rmdir(task_prog_dir)
    shutil.copytree(src_dir / 'Program', task_prog_dir)
os.chdir(task_prog_dir)
input_APEX = inAPEX(scenario=task_name)
sim_APEX = simAPEX(src_dir, winepath=args.winepath, in_obj=input_APEX, attribute='runoff',
                   is_pesticide=True, scenario=task_name, id_case=args.case_id,
                   warm_years=4, calib_years=11)
os.chdir(curr_directory)
