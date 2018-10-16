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

Large jobs may take a longer to start up, especially on KNL nodes. The srun option --bcast=<destination_path> is recommended for large jobs requesting over 1500 MPI tasks. By default SLURM loads the executable to the allocated compute nodes from the current working directory, this may take long time when the file system (where the executable resides) is slow. With the --bcast=/tmp/myjob, the executable will be copied to the /tmp/myjob directory. Since /tmp is part of the memory on the compute nodes, it can speed up the job startup time.

```bash
% sbcast --compress=lz4 ./mycode.exe /tmp/mycode.exe     # here -C is to compress first
% srun <srun options> /tmp/mycode.exe
# or in the case of when numactl is not needed: % srun --bcast=/tmp/mycode.exe --compress=lz4 <srun options> ./mycode.exe
```

## Network locality

For jobs which are sensitive to interconnect (MPI) performance and
utilize less than ~300 nodes it is possible to request that all nodes
are in a single Aries dragonfly group.

* [Example](examples/index.md#network-topology)

## Core Specialization

Core specialization is a feature designed to isolate system overhead (system interrupts, etc.) to designated cores on a compute node. It is generally helpful for running on KNL, especially if the application does not plan to use all physical cores on a 68-core compute node. Set aside 2 or 4 cores for core specialization is recommended.

The srun flag for core specialization is "-S" or "--core-spec".  It only works in a batch script with sbatch.  It can not be requested as a flag with salloc for interactive batch, since salloc is already a wrapper script for srun.

* [Example](examples/index.md#core-specialization)
