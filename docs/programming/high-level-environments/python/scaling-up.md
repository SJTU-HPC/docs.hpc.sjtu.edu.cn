# Scaling Up

Many scientists appreciate Python's power for prototyping and developing
scientific computing and data-intensive applications. However, creating
parallel Python applications that scale well in modern high-performance
computing environments can be challenging.

## Process Parallelism in Python

Multiprocessing is a Python API that provides the ability for Python
interpreters to run in parallel and even share memory (with caveats) on a
**single** node. For more information please see our
[multiprocessing](multiprocessing.md) page.

The mpi4py library provides bindings for using MPI in Python. It can be used on a
single node and all the way up to thousands of nodes. For more information
about using mpi4py at NERSC please see our [mpi4py](mpi4py.md) page. A word of
caution: using mpi4py on many nodes (~100+) can be very slow to start up. For larger mpi4py
jobs we **strongly** recommend Shifter (see below) to help improve startup time.

Dask is a task framework that allows Python to flexibly scale from small to large
systems. You can read more about it on our Dask page [here](dask.md).

Combining mpi4py and the Python subprocess module may be a suitable way to
manage embarrassingly parallel execution. Typically a user writes a
Python+mpi4py driver code that runs another program through subprocess.  This
approach has been observed to work with the Cray MPICH used to build mpi4py if
the external worker code is a compiled executable (C, C++, Fortran) compiled
without the Cray wrappers (cc, CC, ftn).  That is, via gcc, g++ or gfortran.

## Importing Python Packages at Scale

When a Python application imports a module, the interpreter executes system
calls to locate the module on the file system and expose its contents to the
application. The number of system calls per "import" statement may be very
large because Python searches for each import target across many paths.  For
instance, `import scipy` on Cori presently opens or attempts to open about 4000
files.

When a large number of Python tasks are running simultaneously, especially if
they are launched with MPI, the result is many tasks trying to open the same
files at the same time, causing contention and degradation of performance.
Python applications running at the scale of a few hundred or a thousand tasks
may take an unacceptable amount of time simply starting up.

To overcome this problem, NERSC strongly advises users to build a Docker image
containing their Python stack and use Shifter to run it.  This is the best
approach to overcome the at-scale import bottleneck at NERSC.  Shifter and
alternatives are described below.

### Shifter (Best Choice)

[Shifter](../../shifter/overview.md) is a technology developed at NERSC to
provide scalable Linux container deployment in a high-performance computing
environment. The idea is to package up an application and its entire software
stack into a Linux container and distribute these containers to the compute
nodes. This localizes the modules and any associated shared object libraries
to the compute nodes, eliminating contention on the shared file system. Using
Shifter results in tremendous speed-ups for launching larger process-parallel
Python applications.

### /global/common/software

The /global/common file system is where NERSC staff install software.  This
file system is mounted read-only from compute nodes with client-side caching
enabled.  These features help mitigate the import bottleneck.  Performance can
be almost as good as Shifter, but Shifter is always the best choice.  Users can
also install software to this read-optimized file system.  Read more about how
to do that [here.](../../../../filesystems/global-common)

### $SCRATCH or /project

The $SCRATCH file system is optimized for large-scale data I/O. It generally
has better performance than, say, the /project file system but for importing
Python packages the performance is not as good as with /global/common. Both the
$SCRATCH filesystem and /project also experience variable load.  Users may
install Python packages to the $SCRATCH filesystem but must remember the purge
policy: we do not recommend doing this.  While /project has no purge
policy, performance during Python import at scale is worse than on $SCRATCH so
NERSC also does not recommend installing Python stacks to /project.

### Other Approaches

There are a few other interventions that we are aware of that can help users
scale their Python applications at NERSC.  One is to bundle up Python and the
dependency stack and broadcast it to the compute nodes where it is placed into
/dev/shm.  This has been described
[here](https://github.com/rainwoodman/python-mpi-bcast).
