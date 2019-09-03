# Using MPI in Python (mpi4py) 

## Intro to mpi4py and an example on Cori

mpi4py provides MPI standard bindings to the Python programming language.
Documentation on mpi4py is available [here](https://mpi4py.readthedocs.io/en/stable/).

Here is an example of how to use mpi4py on Cori using Anaconda Python 3.7.
Consider this minimal example program:

    #!/usr/bin/env python
    from mpi4py import MPI
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
    print(mpi_rank, mpi_size)

This program will initialize MPI, find each MPI task's rank in the global
communicator, find the total number of ranks in the global communicator, print
out these two results, and exit.  Finalizing MPI with mpi4py is not necessary;
it happens automatically when the program exits.

Suppose we put this program into a file called "mympi.py." To run it on the
Haswell nodes on Cori, we could create the following batch script in the same
directory as our Python script, that we call "myjob.sh:"

    #!/bin/bash
    #SBATCH --constraint=haswell
    #SBATCH --nodes=3
    #SBATCH --time=5

    module load python/3.7-anaconda-2019.07
    srun -n 96 -c 2 python mympi.py

More detailed documentation about how to run batch jobs on Cori is available
[here](../../../jobs/index.md). We also provide a [job script
generator](https://my.nersc.gov/script_generator.php) at MyNERSC that you may
find useful.

To run "mympi.py" in batch on Cori, we submit the batch script from the command
line using sbatch, and wait for it to run:

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

## `mpi4py` in your custom conda environment

If you would like to use mpi4py in a custom conda environment, you will need
to install and build it inside your environment.

!!! Warning "Do NOT conda/pip install mpi4py"
    You can install mpi4py using these tools without any warnings,
    but your mpi4py programs just won't work. To use Cori's
    MPICH MPI, you'll need to build it yourself using the [Cray compiler
    wrappers](../../../../programming/compilers/wrappers) that link in Cray
    MPICH libraries.

You can build `mpi4py` and install it into a conda environment on Cori
using a recipe like the following:

    wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.0.tar.gz
    tar zxvf mpi4py-3.0.0.tar.gz
    cd mpi4py-3.0.0
    module swap PrgEnv-intel PrgEnv-gnu
    module unload craype-hugepages2M
    python setup.py build --mpicc="$(which cc) -shared"
    python setup.py install

The MPI-enabled Python interpreter is not required (see [this
page](https://mpi4py.readthedocs.io/en/stable/appendix.html#mpi-enabled-python-interpreter)
in the mpi4py documentation). To install it however, use these additional
steps:

    python setup.py build_exe --mpicc="$(which cc) -dynamic"
    python setup.py install_exe

!!! Warning "MPI_COMM_WORLD size is 1 ?!?!"
    If you try to use mpi4py and you observe something like an apparent
    `MPI_COMM_WORLD` size of 1 and all processes report that they are rank
    0, check to see if you have installed mpi4py from Anaconda with the
    Conda tool (which will not work on our systems). If you have,
    scroll back up and see the directions
    about how to build mpi4py correctly in your conda environment.

Ok so now you have mpi4py built and ready to use. Make sure you grab
a compute node either via the interactive queue or with sbatch. MPI
is disabled on our login nodes to prevent users from running their
expensive computations there. If you try to use MPI on a login node
you'll see this warning:

!!! Warning "MPI doesn't work on NERSC login nodes"
    Initializing MPI on a login node will not work at NERSC.  This is what you will
    see if you try to do it:
    ```
    nersc$ module load python
    nersc$ python -c 'from mpi4py import MPI'
    [Fri Aug  9 09:26:55 2019] [unknown] Fatal error in PMPI_Init_thread: Other MPI error, error stack:
    MPIR_Init_thread(537):
    MPID_Init(246).......: channel initialization failed
    MPID_Init(647).......:  PMI2 init failed: 1
    Aborted
    ```

If you see this kind of output from a batch job or in an interactive allocation
then it means something different. It likely means that `MPI_Init()` exceeded
a timeout, perhaps due to I/O issues. This is more likely to occur when the
file system you are importing packages from isn't optimized for serving up code
to the compute nodes. Increasing the timeout is a temporary fix:
```
    export PMI_MMAP_SYNC_WAIT_TIME=300
```
but it just gives your job more time to start up.  What you want is for your
job to start up more quickly.  See the documentation on
[`/global/common/software`](../../../../filesystems/global-common)
or better yet,
[Shifter.](../../shifter/overview.md)

!!! tip "About Huge Memory Pages (As of 2019-08-02)"
    Note also that we recommend you unload craype-hugepages2M before
    compiling.  There's an issue with how Python and huge memory pages
    can work together, but Cray is working on a solution.  When that fix
    is in place we'll reconsider the guidance here, but for now
    compiling mpi4py without huge memory pages seems the easiest path
    forward for users.
