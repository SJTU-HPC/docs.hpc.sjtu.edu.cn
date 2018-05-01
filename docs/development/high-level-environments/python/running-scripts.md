
# Running Scripts

Run serial Python scripts on a login node, or on a compute node in an interactive session (started via salloc) or batch job (submitted via sbatch) as you normally would in any Unix-like environment.
On login nodes, please be mindful of resource consumption since those nodes are shared by many users at the same time.

Parallel Python scripts launched in an interactive (salloc) session or batch job (sbatch), such as those using MPI via the mpi4py module, must use srun to launch:

    srun -n 64 python ./hello-world.py

## Parallelism in Python

Many scientists have come to appreciate Python for developing scientific computing applications.
Creating such applications that scale in modern high-performance computing environments can be a challenge.
There are a number of approaches to parallel processing in Python.
Here we describe approaches that we know work for users at NERSC.
For advice on scaling up Python applications, see this page.

### MPI for Python (mpi4py)

MPI standard bindings to the Python programming language.
Documentation on mpi4py is available here and a useful collection of example scripts can be found here.

Here is an example of how to use mpi4py on Cori using Anaconda Python 3.6.
Consider this minimal example program:

    #!/usr/bin/env python
    from mpi4py import MPI
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
    print(mpi_rank, mpi_size)

This program will initialize MPI, find each MPI task's rank in the global communicator, find the total number of ranks in the global communicator, print out these two results, and exit.
Finalizing MPI with mpi4py is not necessary; it happens automatically when the program exits.

Suppose we put this program into a file called "mympi.py."
To run it on the Haswell nodes on Cori, we could create the following batch script in the same directory as our Python script, that we call "myjob.sh:"

    #!/bin/bash
    #SBATCH --constraint=haswell
    #SBATCH --nodes=3
    #SBATCH --time=5
    
    module load python/2.7-anaconda
    srun -n 96 -c 2 python mympi.py

More detailed documentation about how to run batch jobs on Cori is available here.
We also provide a job script generator form at MyNERSC that you may find useful.

To run "mympi.py" in batch on Cori, we submit the batch script from the command line using sbatch, and wait for it to run:

    % sbatch myjob.sh
    Submitted batch job 987654321

After the job finishes, the output will be found in the file "slurm-987654321.out:"

    % cat slurm-987654321.out
    ...
    91 96
    44 96
    31 96
    ...
    0 96
    ...

### Python Multiprocessing

Python's standard library provides a multiprocessing package that supports spawning of processes.
This can be used to achieve some level of parallelism within a single compute node.
It cannot be used to achieve parallelism across compute nodes.
For that, users are referred to the discussion on mpi4py below.

If you are using the multiprocessing module, it is advised that you tell srun to use all the threads available on the node with the "-c" argument.
For example, on Cori use:

    srun -n 1 -c 64 python script-using-multiprocessing.py

NOTE: Python multiprocessing achieves process-level parallelism through fork().
By default you can only expect multiprocessing to do a "pretty good" job of load-balancing tasks.
For more fine-grained control of parallelism within a node, consider parallelism via Cython or writing C/C++/Fortran extensions that take advantage of OpenMP or threads.

FURTHER NOTE: Staff at various other centers go so far as to recommend strongly against using multiprocessing at all in an HPC context because of issues with affinity of forked processes; Python multiprocessing's shared memory model interacting poorly with many MPI implementations, threaded libraries, and libraries using shared memory; and debuggers and performance tools have trouble following forked processes.
We suppose that it can have limited application in specific cases, provided users are informed of the issues.

#### Multiprocessing Interaction with OpenMP

If your multiprocessing code makes calls to a threaded library like numpy with threaded MKL support then you need to consider oversubscription of threads.
While process affinity can be controlled to some degrees in certain contexts (e.g. Python distributions that implement os.sched_{get,set}affinity) it is generally easier to reduce the number of threads used by each process.
Actually it is most advisable to set it to a single thread.
In particular for OpenMP:

    export OMP_NUM_THREADS=1

Furthermore, use of Python multiprocessing on KNL you are advised to specify

    export KMP_AFFINITY=disabled

as explained here. (TBD)

#### Issues Combining Multiprocessing and MPI

Users have been able to combine Python multiprocessing and mpi4py to achieve hybrid parallelism on NERSC systems, but not without issues.
If you decide to try to combine mpi4py and Python multiprocessing, be advised that on the NERSC Cray systems (Cray MPICH) one must set the following environment variable:

    export MPICH_GNI_FORK_MODE=FULLCOPY

See the "mpi_intro" man-page for details.
Again we advise that combining Python multiprocessing and mpi4py qualifies as a "hack" that may work for developers in the short term.
Users are strongly encouraged to consider alternatives.
