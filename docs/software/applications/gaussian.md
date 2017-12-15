# Description

Gaussian 09 is a connected series of programs for performing semi-empirical, density functional theory and ab initio molecular orbital calculations. Starting from the fundamental laws of quantum mechanics, Gaussian 09 predicts the energies, molecular structures, vibrational frequencies and molecular properties of molecules and reactions in a wide variety of chemical environments. Gaussian 09’s models can be applied to both stable species and compounds which are difficult or impossible to observe experimentally (e.g., short-lived intermediates and transition structures).

# How to Access Gaussian 09

NERSC uses modules to manage access to software. To use the default version of Gaussian 09, type

```FORTRAN
% module load g09
```

# Access Restrictions

Gaussian is available to the general user community at NERSC subject to the License To Use Agreement between the U.C. Regents, Lawrence Berkeley National Lab and Gaussian Inc. This agreement restricts use of the Gaussian software in that NERSC may only "provide to third parties who are not directly or indirectly engaged in competition with Gaussian access to the binary code of the Software."

You must certify that this condition is met by using the g09_register command. A NERSC consulting trouble ticket will be generated and you will be provided access in a timely manner.  The procedure for doing this is as follows:

1. module load g09
2. g09_register
3. follow instructions
4. wait for confirmation email from NERSC consultants

Note: If you already registered for Guassian 03 at NERSC then you don't have to register for Guassian 09 again. Gaussian 09 and Gaussian 03 share the same license agreement.


# Using Gaussian on Cori Phase 1

G09 can be used on Cori Phase 1 by utilizing Cray Compatibility Mode and Shifter.

Use the following syntax in your batch script:

#!/bin/bash -l
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:30:00
cd $SLURM_SUBMIT_DIR

module load g09

g09launch < test.com > test.out

Each Cori Phase 1 node has 32 cores. In general, choose nproclinda to the number of nodes you want to run on, and nprocshared to 32. Please run all your calculations out of a sub-folder under your $SCRATCH directory. Please do not run calculations in your home directory area.

Gaussian 16 is now available on Cori as well. It should be used as follows:

#!/bin/bash -l
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:30:00
cd $SLURM_SUBMIT_DIR

module load g16

g16launch < test.com > test.out

 

# Using Gaussian 09 on Edison

A similar batch script to Cori is required. However, pay attention to the "--ccm" line and the need to load the shifter module.

#!/bin/bash -l
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -t 00:30:00
#SBATCH --ccm
cd $SLURM_SUBMIT_DIR

module load g09 shifter

g09launch < test.com > test.out

 

# Notes on Memory and Storage:

Some jobs, especially MP2, may consume large memory and disk storage resources. Instead of running these kinds of jobs in distributed memory Linda-parallel mode it might be better to use a shared-memory parallel approach. For larger systems Gaussian09 also allows a mixed-mode approach using shared-memory-parallelism within nodes and Linda only between nodes.

Using shared memory parallel execution can save a lot of disk space usage (roughly eight times), since tasks on the same node will share a copy of the scratch file whereas each Linda-parallel task will create its own copy of the scratch data file. The savings of up to a factor of eight can be quite significant because the minimum disk required for MP2 frequencies is a multiple of N^4 (where N is the number of basis functions).

For a one-node job (eight cores) use, for example, something like:

%mem=16gb
%nprocshared=8
and for multiple nodes job (for example, two nodes), use something like:

%mem=16gb
%NProcShared=8
%NProcLinda=2
The parameter NProcLinda should equal the number of nodes used for your job. The total number of the processors used to run the g09 job is NProcLinda X NProcShared.

For very large jobs, you might consider setting two Gaussian09 parameters, %Mem and %MaxDisk, that affect the amount of memory and disk, respectively, in order to produce good general performance. For the types of calculations that obey %MaxDisk, the disk usage will be kept below this value. See the Gaussian Efficiency Considerations web page for details. There are some examples in the directory $g09root/g09/tests/com.

When using multiple processors with shared memory, a good estimate of the memory required is the amount of memory required for a single processor job times the number of cores used per node. In other words, %Mem represents the total memory requested for each node. For distributed memory calculations using Linda, the amount of memory specified in %Mem should be equal to or greater than the value for a single processor job.

When setting %mem, remember that some memory will be used by the operating system. Also Gaussian needs some memory on top of what you reserve for data with %mem. So for example on Edison, of 64GB memory on the node, about 61GB is available to jobs. Gaussian will use a few GB more, so if you set %mem much higher than about 55GB, it may fail due to being unable to allocate memory.

Special Memory Notes for Carver:

We have set memory limits (soft limit 2.5Gb, hard 20Gb) on Carver compute nodes to protect the nodes from crashes due to using too much memory.  As a result, g09 jobs that request more than 2.5Gb memory through the %mem key word will fail. The error message you will see is "galloc: could not allocate memory."

The workaround, if you use bash or if your gaussian job script uses the bash shell, is to add the following in your batch script (submit file):

ulimit -v 20971520
If you use a csh job script, put the following in your batch script (submit file):

limit vmemoryuse unlimited
Make sure to put the above IF blocks after the NERSC_HOST is set in your dot files.

# Documentation

Gaussian 09 Online Manual

