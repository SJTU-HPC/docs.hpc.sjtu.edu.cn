# Best practices for jobs

## Time Limits

Due to backfill scheduling short and variable length jobs generally
start quickly resulting in much better job throughput.

```bash
#SBATCH --time-min=<lower_bound>
#SBATCH --time=<upper_bound>
```

## Long running jobs

Simulations which must run for a long period of time achieve the best
throughput when composed of main small jobs utilizing
checkpoint/restart chained together.

* [Example: job chaining](./examples/index.md#dependencies)

## I/O performance

Edison and Cori each have dedicated large, local, parallel scratch
file systems.  The scratch file systems are intended for temporary
uses such as storage of checkpoints or application input and
output. Data and I/O intensive applications should use the local
scratch (or Burst Buffer) filesystems.

These systems should be referenced with the environment variable
`$SCRATCH`.

!!! tip
	On Cori the [Burst Buffer](#) offers the best I/O performance.

!!! warn
	Scratch filesystems are not backed up and old files are
	subject to purging.

## File Sytem Licenses

A batch job will not start if the specified file system is unavailable
due to maintenance or an outage or if a performance issue with
filesystem is detected.

```bash
#SBATCH --license=SCRATCH,project
```

### Cori

* `cscratch1` (or `SCRATCH`)
* `project`
* `projecta`
* `projectb`
* `dna`
* `seqfs`
* `cvmfs`

### Edison

* `scratch1` (or `SCRATCH`)
* `scratch2` (or `SCRATCH`)
* `scratch3`
* `cscratch1`
* `project`
* `projecta`
* `projectb`
* `dna`
* `seqfs`
* `cvmfs`

## Running Large Jobs (over 1500 MPI tasks)

Large jobs may take a longer to start up, especially on KNL nodes. The
srun option --bcast=<destination_path> is recommended for large jobs
requesting over 1500 MPI tasks. By default Slurm loads the executable
to the allocated compute nodes from the current working directory,
this may take long time when the file system (where the executable
resides) is slow. With the --bcast=/tmp/myjob, the executable will be
copied to the /tmp/myjob directory. Since /tmp is part of the memory
on the compute nodes, it can speed up the job startup time.

```bash
% sbcast --compress=lz4 ./mycode.exe /tmp/mycode.exe     # here -C is to compress first
% srun <srun options> /tmp/mycode.exe
# or in the case of when numactl is not needed: % srun --bcast=/tmp/mycode.exe --compress=lz4 <srun options> ./mycode.exe
```

## Network locality

For jobs which are sensitive to interconnect (MPI) performance and
utilize less than ~300 nodes it is possible to request that all nodes
are in a single Aries dragonfly group.

Slurm has a concept of "switches" which on Cori and Edison are
configured to map to Aries electrical groups. Since this places an
additional constraint on the scheduler a maximum time to wait for the
requested topology can be specified.

!!! example
	Wait up to 60 minutes
	```bash
	sbatch --switches=1@60 job.sh
	```

!!! info "Additional details and information"
	* [Cray XC Series Network (pdf)](https://www.cray.com/sites/default/files/resources/CrayXCNetwork.pdf)

## Core Specialization

Core specialization is a feature designed to isolate system overhead
(system interrupts, etc.) to designated cores on a compute node. It is
generally helpful for running on KNL, especially if the application
does not plan to use all physical cores on a 68-core compute node. Set
aside 2 or 4 cores for core specialization is recommended.

The srun flag for core specialization is "-S" or "--core-spec".  It
only works in a batch script with sbatch.  It can not be requested as
a flag with salloc for interactive batch, since salloc is already a
wrapper script for srun.

* [Example](examples/index.md#core-specialization)

## Process placement

Several mechanisms exsist to control process placement on NERSC's Cray
systems. Application performance can depend strongly on placement
depending on the communication pattern and other computational
characteristics.

Examples are run on Cori.

### Default

```
user@nid01041:~> srun -n 8 -c 2 check-mpi.intel.cori|sort -nk 4
Hello from rank 0, on nid01041. (core affinity = 0-63)
Hello from rank 1, on nid01041. (core affinity = 0-63)
Hello from rank 2, on nid01111. (core affinity = 0-63)
Hello from rank 3, on nid01111. (core affinity = 0-63)
Hello from rank 4, on nid01118. (core affinity = 0-63)
Hello from rank 5, on nid01118. (core affinity = 0-63)
Hello from rank 6, on nid01282. (core affinity = 0-63)
Hello from rank 7, on nid01282. (core affinity = 0-63)
```

### `MPICH_RANK_REORDER_METHOD`

The `MPICH_RANK_REORDER_METHOD` environment variable is used to
specify other types of MPI task placement. For example, setting it to
0 results in a round-robin placement:

```
user@nid01041:~> MPICH_RANK_REORDER_METHOD=0 srun -n 8 -c 2 check-mpi.intel.cori|sort -nk 4
Hello from rank 0, on nid01041. (core affinity = 0-63)
Hello from rank 1, on nid01111. (core affinity = 0-63)
Hello from rank 2, on nid01118. (core affinity = 0-63)
Hello from rank 3, on nid01282. (core affinity = 0-63)
Hello from rank 4, on nid01041. (core affinity = 0-63)
Hello from rank 5, on nid01111. (core affinity = 0-63)
Hello from rank 6, on nid01118. (core affinity = 0-63)
Hello from rank 7, on nid01282. (core affinity = 0-63)
```

There are other modes available with the `MPICH_RANK_REORDER_METHOD`
environment variable, including one which lets the user provide a file
called `MPICH_RANK_ORDER` which contains a list of each task's
placement on each node. These options are described in detail in the
`intro_mpi` man page on Cori and Edison.

#### `grid_order`

For MPI applications which perform a large amount of nearest-neighbor
communication, e.g., stencil-based applications on structured grids,
Cray provides a tool in the `perftools-base` module called
`grid_order` which can generate a `MPICH_RANK_ORDER` file automatically
by taking as parameters the dimensions of the grid, core count,
etc. For example, to place MPI tasks in row-major order on a Cartesian
grid of size $(4, 4, 4)$, using 32 tasks per node on Cori:

```
cori$ module load perftools-base
cori$ grid_order -R -c 32 -g 4,4,4
# grid_order -R -Z -c 32 -g 4,4,4
# Region 3: 0,0,1 (0..63)
0,1,2,3,16,17,18,19,32,33,34,35,48,49,50,51,4,5,6,7,20,21,22,23,36,37,38,39,52,53,54,55
8,9,10,11,24,25,26,27,40,41,42,43,56,57,58,59,12,13,14,15,28,29,30,31,44,45,46,47,60,61,62,63
```

One can then save this output to a file called `MPICH_RANK_ORDER` and
then set `MPICH_RANK_REORDER_METHOD=3` before running the job, which
tells Cray MPI to read the `MPICH_RANK_ORDER` file to set the MPI task
placement. For more information, please see the man page `man
grid_order` (available when the `perftools-base` module is loaded) on
Cori and Edison.

## Serial jobs

Users requiring large numbers of serial jobs have several options at
NERSC.

* [shared qos](/jobs/examples/index.md#shared)
* [job arrays](/jobs/examples/index.md#job-arrays)
* [workflow tools](/jobs/workflow-tools.md)
