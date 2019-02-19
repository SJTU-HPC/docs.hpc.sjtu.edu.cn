# Open MPI

The [Open MPI Project](https://www.open-mpi.org/) is an open source [Message Passing Interface](https://www.mpi-forum.org/) implementation that is developed and maintained by a consortium of academic, research, and industry partners. Open MPI is therefore able to combine the expertise, technologies, and resources from all across the High Performance Computing community in order to build the best MPI library available. Open MPI offers advantages for system and software vendors, application developers and computer science researchers.

This wiki is primarily intended for NERSC users who wish to use Open MPI on Cori, however these instructions are sufficiently general that they should be largely applicable to other Cray XC systems running SLURM.

## Using Open MPI at NERSC

### Compiling

Load the Open MPI module to pick up the packages compiler wrappers, mpirun launch command, and other utilities:

```shell
nersc$ module use /global/common/software/m3169/modulefiles
nersc$ module load openmpi
```

Open MPI has been compiled using the PrgEnv-gnu/6.0.4 and PrgEnv-intel/6.0.4 Cray programming environments.  The module file
will detect which compiler environment you have loaded and load the appropriately built Open MPI package.

The simplest way to compile your application when using Open MPI is via the compiler wrappers, e.g.

```shell
nersc$ mpicc -o my_c_exec my_c_prog.c
nersc$ mpif90 -o my_f90_exec my_f90_prog.f90
```

You pass extra compiler options to the back end compiler just as you would if using the compiler (not the cray wrappers) directly.  Note by default the Open MPI compiler wrappers will build dynamic executables.

### Job Launch

There are two ways to launch jobs compiled against Open MPI on cori.  You can either use the Open MPI supplied ```mpirun``` job launcher, or SLURM's srun job launcher, e.g.

```shell
nersc$ salloc -N 5 --ntasks-per-node=32 -C haswell
nersc$ srun -n 160 ./my_c_exec
```
or

```shell
nersc$ salloc -N 5 --ntasks-per-node=32 -C haswell
nersc$ mpirun -np 160 ./my_c_exec
```

If you wish to use srun, you should use the same srun options as if your application was compiled and linked against the vendor's MPI implementation.

If you wish to use the Open MPI ```mpirun``` job launcher,  the same options you use on other HPC clusters with Haswell or KNL nodes can be used.  See the mpirun man page for more details about command line options.  ```mpirun --help``` may also be used to get more information about mpirun command line options.

Note if you wish to use MPI dynamic process functionality such as ```MPI_Spawn```, you must use ```mpirun``` to launch the application.

The Open MPI package has been built so that it can be used on both the Haswell and KNL partitions.

### Using Java Applications with Open MPI

The Open MPI supports a Java interface.  Note this interface has not been standardized by the MPI Forum.  Information on how use Open MPI's Java interface is available on the [Open MPI Java FAQ](https://www.open-mpi.org/faq/?category=java).  Note you will need to load the java module to use Open MPI's Java interface.

### Using Open MPI with OFI libfabric

The Open MPI installed on cori is configured to optionally use the OFI libfabric transport.  By default Open MPI's native uGNI interface to the Aries network is used.  If you would like to work with the OFI libfabric transport instead, the following environment variable needs to be set:

```shell
export OMPI_MCA_pml=cm
```

Note that the OFI libfabric GNI provider targets functionality over performance.
