# AMBER 

[Amber](http://ambermd.org/) (Assisted Model Building with Energy Refinement) is the collective name for a suite of programs designed to carry out molecular mechanical force field simulations, particularly on biomolecules.  See [Amber force fields](http://ambermd.org/#ff),  AMBER consists of about 50 programs.  Two major ones are:

* sander: Simulated annealing with NMR-derived energy restraints
* pmemd: This is an extensively-modified version of sander, optimized for periodic, PME simulations, and for GB simulations. It is faster than sander and scales better on parallel machines.


## How to access AMBER 
NERSC uses [modules](https://www.nersc.gov/users/software/user-environment/modules/) to manage access to software.
To use the default version of AMBER, type:
```shell
% module load amber
```
To see where the AMBER executables reside (the bin directory) and what environment variables it defines, type
```shell
% module show amber
```
E.g., on Cori,

```shell
% module show amber
-------------------------------------------------------------------
/usr/common/usg/Modules/modulefiles/amber/14:

module-whatis AMBER is a collection molecular dynamics simulation programs
setenv AMBERHOME /usr/common/usg/amber/14
setenv AMBER_DAT /usr/common/usg/amber/14/dat
setenv OMP_NUM_THREADS 1
prepend-path PYTHONPATH /usr/common/usg/amber/14/bin
prepend-path PATH /usr/common/usg/amber/14/bin
-------------------------------------------------------------------
```
To see the available executables, type
```shell
% ls -l /usr/common/usg/amber/14/bin
```
You should choose an appropriate binary to run your jobs. The sander, sander.LES, sander.PIMD are the serial binaries, their parallel binaries are sander.MPI, sander.LES.mpi, sander.PIMD.MPI, respectively.

## How to run AMBER 
 
There are two ways of running AMBER: submitting a batch job, or running interactively in an interactive batch session. Here is a sample batch script to run AMBER on Cori:

!!! example "Cori Haswell"
    ```shell
    --8<-- "docs/applications/amber/cori-hsw.sh"
    ```
Then submit the job script using sbatch command, e.g., assume the job script name is test_amber.slurm,
```shell
% sbatch test_amber.slurm
```
To request an interactive batch session, issue a command such as this one (e.g., requesting two nodes on Cori):
```shell
% salloc -N 2 -q interactive -C haswell -t 30:00 
```
when a new batch session prompts, type the following commands:
```shell
% module load abmer

% #on Cori,
% srun -n 64 sander.MPI -i mytest.in -o mytest.out ... (more sander command line options)
```

## Documentation
[Amber Home Page](http://ambermd.org/)

