# MOLPRO 

[MOLPRO](https://www.molpro.net/) 
is a complete system of ab initio programs for molecular electronic structure calculations, written and maintained by H.-J. Werner and P. J. Knowles, with contributions from several other authors. As distinct from other commonly used quantum chemistry packages, the emphasis is on highly accurate computations, with extensive treatment of the electron correlation problem through the multiconfiguration-reference CI, coupled cluster, and associated methods. Using recently developed integral-direct local electron correlation methods, which significantly reduce the increase of the computational cost with molecular size, accurate ab initio calculations can be performed for much larger molecules than with most other programs.

The heart of the program consists of the multiconfiguration SCF, multireference CI, and coupled-cluster routines, and these are accompanied by a full set of supporting features.

## Accessing MOLPRO

NERSC uses [modules](https://www.nersc.gov/users/software/user-environment/modules/) to manage access to software. To use the default version of MOLPRO, type:
```shell
% module load molpro
```
To see all the available versions, use:
```shell
% module avail molpro
```
To see where the MOLPRO executables reside (the bin directory) and what environment variables the modulefile defines, use:
```shell
% module show molpro
```
## Running MOLPRO  
You must use the batch system to run MOLPRO on the compute nodes. You can do this interactively or you can use a script. Examples of both are below.

To run a parallel job interactively use the "salloc" command to request an interactive batch session. 
Here is an example, requesting 1 Haswell node for 30 minutes to run jobs interactively

```shell
% salloc -N 1 -q interactive -t 1:00:00 -C haswell
```
When this command is successful a new batch session will start in the window where you typed the command. Then, issue commands similar to the following:

```shell
%  module load molpro 
%  molpro -n 32 your_molpro_inputfile_name
```
Note that there are 32 cores (or 64 logical cores with Hyperthreading) per Haswell node on Cori. 
You can run up to 32 way (or 64 way with Hyperthreading) parallel molpro jobs on a single node. 
Note that when the time limit (one hour) is reached the job will end, and the session will exit.

To run a batch job on Cori, use a job script similar to this one:
Put those commands or similar ones in a file, say, run.slurm and then use the sbatch command to submit the job:

!!! example "Cori Haswell"
    ```shell
    --8<-- "docs/applications/molpro/cori-hsw.sh"
    ```
```shell
% sbatch run.slurm
```
If your job requires large memory, meaning more than available on per core memory, 4.0 GB, on Cori, you can run with a reduced number of cores per node:

In this example, the job will run with only 8 cores on the node (out of 32 cores available on a Cori node), each task will then able to use up to 4 times as much as memory (4x4.0GB=16GB on Cori). Note that your repo will still be charged for the full node (all 32 cores on the node) although you use only 8 out of 32 available cores. 

!!! example "Cori Haswell"
    ```shell
    --8<-- "docs/applications/molpro/cori-hsw-less-cores.sh"
    ```
If you run small parallel jobs using less than 32 cores available, you can use the shared partition, for which jobs are charged for the number of cores actually used instead of the full nodes (all 32 cores). The shared partition allows a  much higher submit limit than the regular partition. Here is a sample job script,

!!! example "Cori Haswell"
    ```shell
    --8<-- "docs/applications/molpro/cori-hsw-shared.sh"
    ```
You can run short jobs interactively using the shared partition as well. Note that the shared partition has a longer wall limit. For example, the following command request 8 cores under the shared partition for 1 hour:
```shell
% salloc -n 8 -q shared -t 1:00:00
```
When a batch shell prompts, do:
```shell
%  module load molpro 
%  molpro -n 8 your_molpro_inputfile_name
```
##Restart Capabilities

By default, the job is run so that all MOLPRO files are generated in $TMPDIR. This is fine if the calculation finishes in one job, but does not provide for restarts. This section describes techniques which can be used to restart calculations.

MOLPRO has three main files which contain information which can be used for a restart: file 1 is the main file, holding basis set, geometry, and the one and two electron integrals; file 2 is the dump file and used to store the wavefunction information, i.e. orbitals, CI coefficients, and density matrices; file 3 is an auxiliary file which can be used in addition to file 2 for restart purposes. File 1 is usually too large to be saved in permanent storage

By putting the following lines in the input file, the wavefunction file (file number 2) can be saved as file "h2.wfu",and the auxiliary file (file number 3) saved as "h2.aux". By default, the files are saved to the subdirectory "wfu" of your home directory if the job runs out of time.

```shell
***,H2
file,2,h2.wfu,new
file,3,h2.aux,new
basis=vdz;
geometry={angstrom;h1;h2,h1,.74}
optscf;
```
The directory where the files are saved may be changed using the "-W" command line option.

These files enable some restarts to be performed, as they provide snapshots of the calculation as each module finishes. Unfortunately, restarting an incomplete SCF or CI calculation is not possible. To use the files in a restart, remove the "new" qualifier from the "file" command:

```shell
***,H2
file,2,h2.wfu
file,3,h2.aux
basis=vdz;
geometry={angstrom;h1;h2,h1,.74}
optscf;
```
##Documentation

[MOLPRO User's manual](https://www.molpro.net/info/release/doc/manual/index.html?portal=user&choice=User%27s+manual)

