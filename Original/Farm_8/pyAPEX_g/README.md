# apex_simulations

This project has a pre-built conda environment at /project/swmru_apex/py_env/
with the following packages:

* python=3.8.13
* numpy=1.18.1
* pandas=1.0.1
* configobj=5.0.6
* fortranformat=1.2.2

## Running the simulations
1. Load all required packages. This step only needs to be run once after login.
    ```
    module load miniconda singularity
    source activate
    conda activate /project/swmru_apex/py_env
    ```
1. Run the simulations
    ```
    cd /project/swmru_apex/apex_simulations
    sbatch launch_calibration_sims.sh
    ```
1. If you get the error "wine: could not load kernel32.dll, status c0000135", run the following command, then repeat the previous step.
    ```
    export WINEPREFIX=/project/swmru_apex
    ```

## Outputs
Directories "Program_X", the copies of 'Program' will be created in /project/swmru_apex/test_outputs
X is the paralleled task id.



## Modify the running parameters in **launch_calibration_sims.sh**


1. change the task arrays
    ```
    #SBATCH --array=1-5
    ```

2. change the running time. (format of time: d-hh:mm:ss)
    ```
    #SBATCH --time 10:00:00
    ```


3. the total number of calibration simulations to run.
    ```
    NSIMS=17
    ```


4. The start simulation id
    ```
    START_SIM_ID=1
    ```


5. The location for writing simulation outputs.
    ```
    OUTPUT_PATH=/project/swmru_apex/test_outputs
    ```

