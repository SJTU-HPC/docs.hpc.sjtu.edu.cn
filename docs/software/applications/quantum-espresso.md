#Quantum ESPRESSO/PWSCF

##Description
[Quantum ESPRESSO](http://www.quantum-espresso.org) is an integrated suite of computer codes for electronic structure calculations and materials modeling at the nanoscale. It builds on the electronic structure codes PWscf, PHONON, CP90, FPMD, and Wannier.  It is based on density-functional theory, plane waves, and pseudopotentials (both norm-conserving and ultrasoft).

##Using Quantum ESPRESSO at NERSC

NERSC uses [modules]() to manage access to software. To use the default version of Espresso, type:

```Shell
module load espresso/<version>
```

where the available versions can be found by using `module avail espresso`.

Recently, we have optimized the hybrid DFT calculations in Quantum ESPRESSO (pw.x) for hybrid OpenMP+MPI applications. These changes are described in our [Quantum ESPRESSO case study](../../performance/case-studies/quantum-espresso/quantum-espresso.md) and available in the module version 6.1 we provide. In order to maximize the pw.x performance, you should use the following example script:

```Shell
#!/bin/bash
#SBATCH -p regular
#SBATCH -C haswell
#SBATCH -N 2
#SBATCH -t 02:00:00
#SBATCH -J my_job

export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load espresso/6.1
srun -n 8 -c 16 --cpu_bind=cores pw.x -nbgrp 8 -input test.in
```

Here we are running 8 MPI ranks on 2 nodes, thus 4 MPI ranks per node and therefore 2 MPI ranks per socket. Since the total number of logical cores per node is 64, we request 64/4=16 cores in the -c statement. This will make all cores available to the program. Since QE does not benefit much from hyperthreading, we set the number of OpenMP threads to 8 and declare spread thread binding so that we have 1 thread running on each physical processor.

For Xeon Phi 7250 (Knight's Landing), the same script should look like

```Shell
#!/bin/bash
#SBATCH -p regular
#SBATCH -C knl,quad,cache
#SBATCH -N 2
#SBATCH -t 02:00:00
#SBATCH -J my_job

export OMP_NUM_THREADS=16
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load espresso/6.1
srun -n 8 -c 68 --cpu_bind=cores pw.x -nbgrp 8 -input test.in
```

Note that there are in total 272 logical cores, so running 4 ranks per node results in 272/4=68 available cores per rank and only a quarter can be used reasonably. To make the number of used cores even, i.e. making sure that all tiles are either empty or packed, we are using only 64/4=16 threads per rank. Nevertheless, to allow for proper thread binding, `-c` is set to 68. 

Concerning band-group parallelization, it is recommended to run one band group per MPI rank. However, please keep in mind that you cannot use more band-group than there are bands in your system, so adjust the number accordingly if you run into issues. Also note that the new implementation is much more efficient, so you might be able to use much fewer nodes and still get the solution within the same wallclock time.

For all routines except pw.x, please run QE in full MPI mode as there is currently no efficient OpenMP implementation available. Therefore, please use the following script (for Haswell, e.g. running `ph.x`):

```Shell
#!/bin/bash
#SBATCH -p regular
#SBATCH -C haswell
#SBATCH -N 2
#SBATCH -t 02:00:00
#SBATCH -J my_job

export OMP_NUM_THREADS=1

module load espresso/6.1
srun -n 64 -c 2 --cpu_bind=cores ph.x -input test.in
```

Pay close attention to the fact that we explicitly set `OMP_NUM_THREADS=1` when running in pure MPI mode. This is optimal when intending to run with only MPI tasks.

independently of what application you want to run and how, submit the job script using the sbatch command, e.g., assuming the job script name is `test_espresso.pbs`:

```Shell
sbatch test_espresso.sl
```

##Compilation Instructions
Some advanced users may be interested in tweaking the Quantum ESPRESSO build parameters and building QE themselves in their own directory. In order to aid in this process, and to provide a greater degree of transparency, the build instructions for the QE module are listed below. The following procedure was used to build Quantum ESPRESSO versions >5.4 on Cori. In the root QE do:

```Shell
./configure
cp /usr/common/software/espresso/<version>/<arch>/<comp>/make.inc .
make <application-name, e.g. pw>
```

whre `<version>` specifies the version, `<arch>` the architecture (usually `hsw` or `knl` for Haswell and KNL respectively) and `<comp>` the compiler (usually `gnu` or `intel`). Note that not all versions are available for all architectures or compilers.