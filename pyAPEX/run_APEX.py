from pyAPEX import inAPEX
from pyAPEX import simAPEX
from pathlib import Path
import shutil
import os.path
import warnings
warnings.filterwarnings('ignore')
output_dir = '../Scenario'
task_name = 'test'
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
input_APEX = inAPEX()
sim_APEX = simAPEX(src_dir, winepath=None, in_obj=input_APEX, attribute='runoff', is_pesticide=True, scenario=task_name,
                   warm_years=4, calib_years=11)
os.chdir(curr_directory)
