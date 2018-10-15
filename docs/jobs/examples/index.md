# Examples

## Basic MPI

One MPI processes per physical core.

??? example "Edison"
	```bash
	--8<-- "docs/jobs/examples/basic-mpi/edison/basic-mpi.sh"
	```

??? example "Cori Haswell"
	```bash
	--8<-- "docs/jobs/examples/basic-mpi/cori-haswell/basic-mpi.sh"
	```

??? example "Cori KNL"
	```bash
	--8<-- "docs/jobs/examples/basic-mpi/cori-knl/basic-mpi.sh"
	```

## Hybrid MPI+OpenMP jobs

One MPI process per socket and 1 OpenMP thread per
physical core

!!! warning
	In Slurm each hyper thread is considered a "cpu" so the
	`--cpus-per-task` option must be adjusted accordingly. Generally
	best performance is obtained with 1 OpenMP thread per physical
	core.

??? example "Edison"
	```bash
	--8<-- "docs/jobs/examples/hybrid-mpi-openmp/edison/hybrid-mpi-openmp.sh"
	```

??? example "Cori Haswell"
	```bash
	--8<-- "docs/jobs/examples/hybrid-mpi-openmp/cori-haswell/hybrid-mpi-openmp.sh"
	```

??? example "Cori KNL"
	```bash
	--8<-- "docs/jobs/examples/hybrid-mpi-openmp/cori-knl/hybrid-mpi-openmp.sh"
	```

## Interactive

Interactive jobs are launched with the `salloc` command.

!!! tip
	Cori has dedicated nodes for interactive work.

??? example "Edison"
	```bash
	edison$ salloc --qos=debug --time=30 --nodes=2
	```

??? example "Cori Haswell"
	```bash
	cori$ salloc --qos=interactive -C haswell --time=60 --nodes=2
	```

??? example "Cori KNL"
	```bash
	cori$ salloc --qos=interactive -C knl --time=60 --nodes=2
	```

!!! note
	[Additional details](../interactive.md) on Cori's interactive
	QOS

## Running Multiple Parallel Jobs Sequentially

Multiple sruns can be executed one after another in a single batch script. Be sure to specify the total walltime needed to run all jobs.

??? example "Cori Haswell"
	```bash
	--8<-- "docs/jobs/examples/multiple-parallel-jobs/cori-haswell/sequential-parallel-jobs.sh"
	```

## Running Multiple Parallel Jobs Simultaneously

Multiple sruns can be executed simultaneously in a single batch script. Be sure to specify the total number of nodes needed to run all jobs at the same time. By default, multiple concurrent srun executions cannot share compute nodes under Slurm in the non-shared QOSs.

In the following example, a total of 192 cores are required, which would hypothetically fit on 192 / 32 = 6 Haswell nodes. However, because sruns cannot share nodes by default, we instead have to dedicate 2 nodes to the first execution (44 cores), 4 to the second (108 cores), and again 2 to the third (40 cores).   For all three executables, the node is not fully packed, and number of MPI tasks per node is not a divisor of 64, so both -c and --cpu_bind flags are used in srun commands.

Notice the "&" at the end of each srun command.  Also the "wait" command at the end of the script is very important.  It makes sure the batch job won't exit before all the simultaneous sruns are completed.

??? example "Cori Haswell"
	```bash
	--8<-- "docs/jobs/examples/multiple-parallel-jobs/cori-haswell/simultaneous-parallel-jobs.sh"
	```

## Job Arrays

Job arrays offer a mechanism for submitting and managing collections
of similar jobs quickly and easily.

This example submits 3 jobs. Each job uses 1 node and has the same
time limit and QOS. The `SLURM_ARRAY_TASK_ID` environment variable is
set to the array index value.

!!! example "Cori KNL"
	```bash
	--8<-- "docs/jobs/examples/job-array/cori-knl/job-array.sh"
	```

!!! info "Additional examples and details"
	* [Slurm job array documentation](https://slurm.schedmd.com/job_array.html)
	* Manual pages via `man sbatch` on NERSC systems

## Dependencies

Job dependencies can be used to construct complex pipelines or chain
together long simulations requiring multiple steps.

!!! note
	The `--parsable` option to `sbatch` can simplify working with job
	dependencies.

!!! example
	```bash
	$ jobid=$(sbatch --parsable first_job.sh)
	$ sbatch --dependency=afterok:$jobid second_job.sh
	```

!!! example
	```bash
	$ jobid1=$(sbatch --parsable first_job.sh)
    $ jobid2=$(sbatch --parsable --dependency=afterok:$jobid1 second_job.sh)
	$ jobid3=$(sbatch --parsable --dependency=afterok:$jobid1 third_job.sh)
	$ sbatch --dependency=afterok:$jobid2,afterok:$jobid3 last_job.sh
	```

!!! info "Additional examples and details"
	* [Bash command substitution](https://www.gnu.org/software/bash/manual/bashref.html#Command-Substitution)
	* [sbatch documentation](https://slurm.schedmd.com/sbatch.html)
	* Manual pages via `man sbatch` on NERSC systems

## Shared

Unlike other QOS's in the shared QOS a single node can be shared by
multiple users or jobs. Jobs in the shared QOS are charged for each
*physical core* in allocated to the job.

The number of physical cores allocated to a job by Slurm is
controlled by three parameters:

 * `-n` (`--ntasks`)
 * `-c` (`--cpus-per-task`)
 * `--mem` - Total memory available to the job (`MemoryRequested`)

!!! note
	In Slurm a "cpu" corresponds to a *hyperthread*. So there are 2
	cpus per *physical core*.

The memory on a node is divided evenly among the "cpus" (or
hyperthreads):

| System | MemoryPerCpu (megabytes)     |
|--------|------------------------------|
| Edison | 1300                         |
| Cori   | 1952                         |

The number of physical cores used by a job is computed by

$$
\text{physical cores} =
\Bigl\lceil
\frac{1}{2}
\text{max} \left(
\Bigl\lceil
\frac{\mathrm{MemoryRequested}}{\mathrm{MemoryPerCpu}}
\Bigr\rceil,
\mathrm{ntasks} * \mathrm{CpusPerTask}
\right) \Bigr\rceil
$$

!!! example "Cori-Haswell"
	A four rank MPI job which utilizes 4 physical cores (and 8
	hyperthreads) of a Haswell node.

	```bash
	#!/bin/bash
	#SBATCH --qos=shared
	#SBATCH --time=5
	#SBATCH --ntasks=1
	#SBATCH --cpus-per-task=8
	```

## Variable-time jobs

The variable-time jobs are for the users who wish to get a better queue turnaround or need to run long running jobs, including jobs longer than 48 hours, the max wall-clock time allowed on Cori and Edison. The variable-time jobs are the jobs submitted with the minimum and maximum time limits. Jobs specifying the minimum time can start execution earlier than otherwise with a time limit anywhere between the minimum and maximum time limits. The pre-terminated jobs are automatically requeued to resume from where the previous executions left off, until the cumulative execution time reaches the requested maximum time limit or the job completes before the requested time limit. The variable-time jobs enable users to run jobs with any length, e.g., one week or even longer. Note that to use the variable-time jobs applications are required to be able to checkpoint/restart by themselves.

In the following example, the --comment option is used to request the user desired maximum time limit, which could be longer than maximum time limit allowed by the batch system. In addition to the time limit (--time option), the --time-min option is used to request the minimum amount of time the job should run. The variable ckpt_overhead is used to specify the amount of time (in seconds) needed for checkpointing. The --signal=B:USR1@<sig-time> option is used to send signal USR1 to the job within sig-time seconds of its end time to terminate the job after checkpointing. The sig-time should match the checkpoint overhead time, ckpt_overhead. The ata module defines the bash functions used in the scripti, and users may need to modify the scripts (get a copy) for their use.

??? example "Edison"
        ```bash
        --8<-- "docs/jobs/examples/variable-time-jobs/edison/variable-time-jobs.sh"
        ```

??? example "Cori Haswell"
        ```bash
        --8<-- "docs/jobs/examples/variable-time-jobs/cori-haswell/variable-time-jobs.sh"
        ```

??? example "Cori KNL"
        ```bash
        --8<-- "docs/jobs/examples/variable-time-jobs/cori-knl/variable-time-jobs.sh"
        ```


## Burst buffer

All examples for the burst buffer are shown with Cori Haswell
nodes. Options related to the burst buffer do not depend on Hawell or
KNL node choice.

!!! note
	The burst buffer is only available on Cori.

### Scratch

Use the burst buffer as a scratch space to store temporary data during
the execution of I/O intensive codes. In this mode all data from the
burst buffer allocation will be removed automatically at the end of
the job.

```bash
--8<-- "docs/jobs/examples/burstbuffer/scratch.sh"
```

### Stage in/out

Copy the named file or directory into the Burst Buffer, which can then
be accessed using `$DW_JOB_STRIPED`.

!!! note
	* Only files on the Cori `$SCRATCH` filesystem can be staged in
	* A full path to the file must be used
	* You must have permissions to access the file
	* The job start may be delayed until the transfer is complete
	* Stage out occurs *after* the job is completed so there is no
      charge

```bash
--8<-- "docs/jobs/examples/burstbuffer/stagein.sh"
```

```bash
--8<-- "docs/jobs/examples/burstbuffer/stageout.sh"
```

### Persistent Reservations

Persistent reservations are useful when multiple jobs need access to
the same files.

!!! warning
	* Reservations must be deleted when no longer in use.
	* There are no guarantees of data integrity over long periods of
	time.

!!! note
	Each persistent reservation must have a unique name.

#### Create

```bash
--8<-- "docs/jobs/examples/burstbuffer/create-persistent-reservation.sh"
```

#### Use

Take care if multiple jobs will be using the reservation to not
overwrite data.

```bash
--8<-- "docs/jobs/examples/burstbuffer/use-persistent-reservation.sh"
```

#### Destroy

Any data on the resevration at the time the script executes will be
removed.

```bash
--8<-- "docs/jobs/examples/burstbuffer/destroy-persistent-reservation.sh"
```

### Interactive

The burst buffer is available in interactive sessions. It is
recommended to use a configuration file for the burst buffer
directives:

```shell
cori$ cat bbf.conf
#DW jobdw capacity=10GB access_mode=striped type=scratch
#DW stage_in source=/global/cscratch1/sd/username/path/to/filename destination=$DW_JOB_STRIPED/filename type=file
```

```shell
cori$ salloc --qos=interactive -C haswell -t 00:30:00 --bbf=bbf.conf
```

## Network topology

Slurm has a concept of "switches" which on Cori and Edison are
configured to map to Aries electrical groups. Since this places an
additional constraint on the scheduler a maximum time to wait for the
requested topology can be specified.

!!! example
	Wait up to 60 minutes
	```bash
	$ sbatch --switches=1@60 job.sh
	```

!!! info "Additional details and information"
	* [Cray XC Series Network (pdf)](https://www.cray.com/sites/default/files/resources/CrayXCNetwork.pdf)
	* [sbatch documentation](https://slurm.schedmd.com/sbatch.html)
    * Manual pages via `man sbatch` on NERSC systems

