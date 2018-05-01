
# Scaling Up

Many scientists appreciate Python's power for prototyping and developing scientific computing and data-intensive applications.
  However, creating parallel Python applications that scale well in modern high-performance computing environments can be challenging for a variety of reasons.  

Approaches to parallel processing in Python at NERSC are described on this page.
 Here we outline various approaches to scaling parallel Python applications at NERSC so that users may select the solution that best suits their performance needs, development practices, and preferences.

## Importing Python Packages at Scale

When a Python application imports a module, the interpreter executes system calls to locate the module on the file system and expose its contents to the application.
 The number of system calls per "import" statement may be very large because Python searches for each import target across many paths.
 For instance, "import scipy" on Cori presently opens or attempts to open about 4000 files.

When a large number of Python tasks are running simultaneously, especially if they are launched with MPI, the result is many tasks trying to open the same files at the same time, causing contention and degradation of performance.
 Python applications running at the scale of a few hundred or a thousand tasks may take an unacceptable amount of time simply starting up.

Overcoming this problem is an area of active study and experimentation at NERSC and elsewhere.
  At the present we can make the following recommendations and offer various approaches to scaling Python applications at NERSC.

### Using /global/common/edison and /global/common/cori

The /global/common file system is where NERSC staff install software.
In August of 2015 this file system was configured so that it is mounted read-only from compute nodes, and client-side caching has been enabled.
This means that importing Python modules installed on the /global/common filesystem should now scale much better than before.
Users who have Python packages they would like installed on /global/common (either under the NERSC Python modules or Anaconda) should contact NERSC staff.

### Using $SCRATCH or /project File Systems

The $SCRATCH file systems on the Cray systems are optimized for I/O.
They generally have better performance than, say, the /project file system but for importing Python packages the performance is not as good as with /global/common.
The $SCRATCH filesystem also experiences variable load that depends on user activity.
Users may install Python packages to the $SCRATCH filesystem but must remember the purge policy.
Users actively developing larger Python packages may wish to use the $SCRATCH file system for this purpose, since access to /global/common requires staff intervention that would interrupt an efficient development cycle.

### Shifter

Shifter is a technology developed at NERSC to provide scalable Linux container deployment in a high-performance computing environment.
The idea is to package up an application and its entire software stack into a Linux container and distribute these containers to the compute nodes.
This localizes the module files and associated shared object libraries to the compute nodes, eliminating contention on the shared file system.
Using Shifter results in tremendous speed-ups for launching Python applications.

### Other Approaches

There are a few other interventions that we are aware of that can help users scale their Python applications at NERSC.
One is to bundle up Python and the dependency stack and broadcast it to the compute nodes where it is placed into /dev/shm.
This has been implemented in the python-mpi-bcast library, developed at BIDS by Yu Feng.
Detailed instructions for using python-mpi-bcast at NERSC are available here. 

## Combining MPI4PY and Subprocess

Combining mpi4py and the Python subprocess module may be a suitable way to manage embarrassingly parallel execution. Typically a user writes a Python+mpi4py driver code that runs another program through subprocess.  This may not be a particularly stable execution model, but has been observed to work with the Cray MPICH used to build mpi4py if the external worker code is a compiled executable (C, C++, Fortran) compiled without the Cray wrappers (cc, CC, ftn).  That is, via gcc, g++ or gfortran.

## H5py

Basic usage and best practice of serial/parallel H5py. 
