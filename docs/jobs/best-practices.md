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

* [Example: job chaining](./examples/index.html#dependencies)

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

## Network locality

For jobs which are sensitive to interconnect (MPI) performance and
utilize less than ~300 nodes it is possible to request that all nodes
are in a single Aries dragonfly group.

* [Example](examples/index.html#network-topology)
