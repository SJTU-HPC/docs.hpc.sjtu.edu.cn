# CPMD 

[CPMD](http://www.cpmd.org/) is a plane wave/pseudopotential DFT code for ab initio molecular dynamics simulations.

## How to access CPMD 
NERSC uses [modules](https://docs.nersc.gov/environment/#nersc-modules-environment) to manage access to software. 
To use the default version of CPMD, type,

```shell
cori$ module load cpmd 
```
## How to run CPMD 

###Running interactively
To run CPMD interactively, you need to request a batch session using the "salloc" command, e.g., 
the following command requests one Cori Haswell node for 1 hour, 

```shell
cori$ salloc -N 1 -q interactive -C haswell -t 1:00:00
```
When the batch session returns with a shell prompt, execute the following commands to run CPMD,
```shell
cori$ module load cpmd
cori$ srun -n 64 cpmd.x test.in [PP-path] > test.out 
```
You need to replace [PP-path] with the path to the pseudo potential files for your job.
 
###Running batch jobs

Here is an example run script. 

!!! example "Cori Haswell"
    ```shell
    --8<-- "docs/applications/cpmd/cori-hsw.sh"
    ```
This script requests two Cori Haswell node for four hours to run CPMD. You need to submit the batch script using the sbatch command, assuming your job script is "run.slurm", 

```shell
cori$ sbatch run.slurm
```

## Documentation

[CPMD Online Manual](http://cpmd.org/documentation) 

