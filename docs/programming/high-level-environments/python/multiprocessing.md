
#Python Multiprocessing

Python's standard library provides a multiprocessing package that supports
spawning of processes. This can be used to achieve some level of parallelism
within a single compute node. It cannot be used to achieve parallelism across
compute nodes. For that, users are referred to the discussion on mpi4py
[here](mpi4py.md).

If you are using the multiprocessing module, it is advised that you tell srun
to use all the threads available on the node with the "-c" argument.  For
example, on Cori use:

    srun -n 1 -c 64 python script-using-multiprocessing.py

!!! tip "Multiprocessing is a *good* solution, but not the *best* solution"
    Python multiprocessing achieves process-level parallelism through fork().  By
    default you can only expect multiprocessing to do a "pretty good" job of
    load-balancing tasks. For more fine-grained control of parallelism within a
    node, consider parallelism via Cython or writing C/C++/Fortran extensions that
    take advantage of OpenMP or threads.

!!! warning "Consider carefully whether multiprocessing is a good fit for your HPC application"
    Staff at various other centers go so far as to recommend strongly against using
    multiprocessing at all in an HPC context because of issues with affinity of
    forked processes; Python multiprocessing's shared memory model interacting
    poorly with many MPI implementations, threaded libraries, and libraries using
    shared memory; and debuggers and performance tools have trouble following
    forked processes. We suppose that it can have limited application in specific
    cases, provided users are informed of the issues.

#### Multiprocessing Interaction with OpenMP

If your multiprocessing code makes calls to a threaded library like numpy with
threaded MKL support then you need to consider oversubscription of threads.
While process affinity can be controlled to some degrees in certain contexts
(e.g. Python distributions that implement os.sched_{get,set}affinity) it is
generally easier to reduce the number of threads used by each process.
Actually it is most advisable to set it to a single thread.  In particular for
OpenMP:

    export OMP_NUM_THREADS=1

Furthermore, use of Python multiprocessing on KNL you are advised to specify:

    export KMP_AFFINITY=disabled


#### Issues Combining Multiprocessing and MPI

Users have been able to combine Python multiprocessing and mpi4py to achieve
hybrid parallelism on NERSC systems, but not without issues.  If you decide to
try to combine mpi4py and Python multiprocessing, be advised that on the NERSC
Cray systems (Cray MPICH) one must set the following environment variable:

    export MPICH_GNI_FORK_MODE=FULLCOPY

See the "mpi_intro" man-page for details.  Again we advise that combining
Python multiprocessing and mpi4py qualifies as a "hack" that may work for
developers in the short term.  Users are strongly encouraged to consider
alternatives.
