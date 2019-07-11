# Q-Chem

[Q-Chem](http://www.q-chem.com/) is a comprehensive ab initio quantum chemistry package for accurate predictions of molecular structures, 
reactivities, and vibrational, electronic and NMR spectra. The new release of Q-Chem 5 represents the state-of-the-art of methodology from 
the highest performance DFT/HF calculations to high level post-HF correlation methods:

* Fully integrated graphic interface including molecular builder, input generator, contextual help and visualization toolkit 
(See amazing image below generated with IQmol; multiple copies available free of charge);
* Dispersion-corrected and double hybrid DFT functionals;
* Faster algorithms for DFT, HF, and coupled-cluster calculations;
* Structures and vibrations of excited states with TD-DFT;
* Methods for mapping complicated potential energy surfaces;
* Efficient valence space models for strong correlation;
* More choices for excited states, solvation, and charge-transfer;
* Effective Fragment Potential and QM/MM for large systems;
* For a complete list of new features, click [here](http://www.q-chem.com/qchem-website/whatsNew5.html). 

## How to access Q-Chem
NERSC uses [modules](https://www.nersc.gov/users/software/user-environment/modules/) to manage access to software. 
To see the available Q-Chem modules, type *"module avail qchem"* command. To access a specific qchem module, 
type *"module load &lt;qchem modulefile&gt;"*, e.g., "module load qchem/5.2". 
In general we recommend users to use the default module, which can be accessed with the following command,

```shell
% module load qchem
```
## How to run Q-Chem

###Running interactively
To run Q-Chem interactively, you need to request a batch session using the "salloc" command, e.g., 
the following command requests one Cori Haswell node for one hour, 

```shell
% salloc -N 1 -q interactive -C haswell -t 1:00:00
```
When the batch session returns with a shell prompt, execute the following commands to run Q-Chem 
```shell
% module load qchem
% qchem -slurm -nt 32 <Q-Chem input file>
```
The above qchem command will run the code with 32 OpenMP threads (-nt 32) on a Cori Haswell node.
 
!!! Note
	1. You should not run Q-Chem jobs on the login nodes, which are shared by many users. The interactive QOS is for users to run jobs interactively.  
	2. Due to the system overhead the memory available to user applications is lower than the physical memory (128 GB for Haswell, and 96 GB for KNL nodes) available on the nodes. Use no more than 118 GB (Haswell) and 87 GB (KNL) in your Q-Chem input files if you specify the total memory for your jobs.   

###Running batch jobs

Here are a few example run scripts. You need to submit the batch script using the sbatch command, assuming your job script is "run.surm",
 
```shell
% sbatch run.slurm
```

!!! example "Cori Haswell"
    ```shell
    --8<-- "docs/applications/qchem/cori-hsw-omp.sh"
    ```
This script requests to run qchem on one Cori Haswell node with 32 OpenMP threads (-nt 32) per task.

The Q-Chem modules available on Cori were built for Haswell, however, the Haswell binaries runs on KNL nodes. You may want to run Q-Chem on Cori KNL nodes for a better job throughput. Here is a sample job script for **Cori KNL**, 

!!! example "Cori KNL"
    ```shell
    --8<-- "docs/applications/qchem/cori-knl-omp.sh"
    ```
If you run single node Q-Chem jobs and do not need all the cores and memory available on the node, you can run your jobs under the shared QOS, which then you will be charged less. For more information about using the shared QOS, see our [Running Jobs](https://docs.nersc.gov/jobs/examples/#shared) page. Here is a sample job script to run Q-Chem with the shared QOS:

!!! example "Cori Haswell: using shared QOS"
    ```shell
    --8<-- "docs/applications/qchem/cori-hsw-shared-omp.sh"
    ```
The above script requests two cores (-n 2) (four CPUs in total) for one hour.

The distributed memory parallelism (MPI) was enabled in the default Q-Chem module, so you can run Q-Chem across multiple nodes. However,
Only basic DFT and TD-DFT features in Q-Chem are capable of utilizing MPI parallelism, 
please consult the Q-Chem manual to ensure effective utilization of computational resources.  
Here is a sample job script to run Q-Chem across multiple nodes. 

!!! example "Cori Haswell: running with multiple nodes"
    ```shell
    --8<-- "docs/applications/qchem/cori-hsw-mpi.sh"
    ```
The job script requests two Haswell nodes, and run qchem with two MPI tasks (-np 2) each with 32 OpenMP threads (-nt 32). 

## Documentation

Q-Chem 5.2 manual [pdf](http://manual.q-chem.com/5.2/qchem_manual_5.2.pdf) [html](http://manual.q-chem.com/5.2/) 

