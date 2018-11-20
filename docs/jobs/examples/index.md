# Examples

## Basic MPI

One MPI process per physical core.

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

Multiple sruns can be executed one after another in a single batch
script. Be sure to specify the total walltime needed to run all jobs.

??? example "Cori Haswell"
	```bash
	--8<-- "docs/jobs/examples/multiple-parallel-jobs/cori-haswell/sequential-parallel-jobs.sh"
	```

## Running Multiple Parallel Jobs Simultaneously

Multiple sruns can be executed simultaneously in a single batch
script.

!!! tip
	Be sure to specify the total number of nodes needed to run all
	jobs at the same time.

!!! note
	By default, multiple concurrent srun executions cannot share
	compute nodes under Slurm in the non-shared QOSs.

In the following example, a total of 192 cores are required, which
would hypothetically fit on 192 / 32 = 6 Haswell nodes. However,
because sruns cannot share nodes by default, we instead have to
dedicate:

* 2 nodes to the first execution (44 cores)
* 4 to the second (108 cores)
* 2 to the third (40 cores)

For all three executables the node is not fully packed and number of
MPI tasks per node is not a divisor of 64, so both `-c` and `--cpu-bind`
flags are used in `srun` commands.

!!! note
	The "`&`" at the end of each `srun` command and the `wait`
	command at the end of the script are very important to ensure the
	jobs are run in parallel and the batch job will not exit before
	all the simultaneous sruns are completed.

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
	jobid=$(sbatch --parsable first_job.sh)
	sbatch --dependency=afterok:$jobid second_job.sh
	```

!!! example
	```bash
	jobid1=$(sbatch --parsable first_job.sh)
    jobid2=$(sbatch --parsable --dependency=afterok:$jobid1 second_job.sh)
	jobid3=$(sbatch --parsable --dependency=afterok:$jobid1 third_job.sh)
	sbatch --dependency=afterok:$jobid2,afterok:$jobid3 last_job.sh
	```

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

!!! example "Cori-Haswell MPI"
	A two rank MPI job which utilizes 2 physical cores (and 4
	hyperthreads) of a Haswell node.

	```bash
	#!/bin/bash
	#SBATCH --qos=shared
	#SBATCH --constraint=haswell
	#SBATCH --time=5
	#SBATCH --ntasks=2
	#SBATCH --cpus-per-task=2

	srun --cpu-bind=cores ./a.out
	```

??? example "Cori-Haswell MPI/OpenMP"
	A two rank MPI job which utilizes 4 physical cores (and 8
	hyperthreads) of a Haswell node.

	```bash
	#!/bin/bash
	#SBATCH --qos=shared
	#SBATCH --constraint=haswell
	#SBATCH --time=5
	#SBATCH --ntasks=2
	#SBATCH --cpus-per-task=4
	export OMP_NUM_THREADS=2
	srun --cpu-bind=cores ./a.out
	```

??? example "Cori-Haswell OpenMP"
	An OpenMP only code which utilizes 6 physical cores.

	```bash
	#!/bin/bash
	#SBATCH --qos=shared
	#SBATCH --constraint=haswell
	#SBATCH --time=5
	#SBATCH --ntasks=1
	#SBATCH --cpus-per-task=12
	export OMP_NUM_THREADS=6
	./my_openmp_code.exe
	```

??? example "Cori-Haswell serial"
	A serial job should start by requesting a single slot and
	increase the amount of memory required only as needed to
	maximize thoughput and minimize charge and wait time.

	```bash
	#!/bin/bash
	#SBATCH --qos=shared
	#SBATCH --constraint=haswell
	#SBATCH --time=5
	#SBATCH --ntasks=1
	#SBATCH --mem=1GB

	./serial.exe
	```

## Xfer queue

The intended use of the xfer queue is to transfer data between Cori or
Edison and HPSS. The xfer jobs run on one of the login nodes and are
free of charge. If you want to transfer data to the HPSS archive
system at the end of a regular job, you can submit an xfer job at the
end of your batch job script via `sbatch -M escori hsi put
<my_files>` (use esedison on Edison), so that you will not get
charged for the duration of the data transfer. The xfer jobs can be
monitored via `squeue -M escori`. The number of running jobs for each
user is limited to the number of concurrent HPSS sessions (15).

!!! warning
    Do not run computational jobs in the xfer queue.

??? example "Edison transfer job"
    ```bash
    #!/bin/bash -l
    #SBATCH -M esedison
    #SBATCH -q xfer
    #SBATCH -t 12:00:00
    #SBATCH -J my_transfer
    #SBATCH -L SCRATCH

    #Archive run01 to HPSS
    htar -cvf run01.tar run01
    ```

??? example "Cori transfer job"
    ```bash
    #!/bin/bash -l
    #SBATCH -M escori
    #SBATCH -q xfer
    #SBATCH -t 12:00:00
    #SBATCH -J my_transfer
    #SBATCH -L SCRATCH

    #Archive run01 to HPSS
    htar -cvf run01.tar run01
    ```

Xfer jobs specifying `-N nodes` will be rejected at submission
time. Also `-C haswell` is not needed since the job does not run on
compute nodes. By default, xfer jobs get ~2GB of memory allocated. If
you're archiving larger files, you'll need to request more memory. You
can do this by adding `#SBATCH --mem=XGB` to the above script (5 - 10
GB is a good starting point for large files).

To monitor your xfer jobs, please use the `squeue -M escori` command,
or `scontrol -M escori show job job_id`.

## Variable-time jobs

Variable-time jobs are for the users who wish to get a better
queue turnaround or need to run long running jobs, including jobs
longer than 48 hours, the max wall-clock time allowed on Cori and
Edison.

Variable-time jobs are submitted with minimum and maximum time limits.
Jobs specifying a minimum time can start execution earlier than
otherwise with a time limit anywhere between the minimum and maximum
time limits. Pre-terminated jobs are automatically requeued to
resume from where the previous executions left off, until the
cumulative execution time reaches the requested maximum time limit or
the job completes before the requested time limit. Variable-time
jobs enable users to run jobs with any length, e.g., one week or even
longer.

!!! note
	Variable-time jobs applications are required to be able to
	checkpoint/restart by themselves.

In the following example, the `--comment` option is used to request
the user desired maximum time limit, which could be longer than
maximum time limit allowed by the batch system. In addition to the
time limit (`--time` option), the `--time-min` option is used to
request the minimum amount of time the job should run. The variable
ckpt_overhead is used to specify the amount of time (in seconds)
needed for checkpointing. The `--signal=B:USR1@<sig-time>` option is
used to send signal `USR1` to the job within sig-time seconds of its
end time to terminate the job after checkpointing. The sig-time should
match the checkpoint overhead time, ckpt_overhead. The ata module
defines the bash functions used in the scripti, and users may need to
modify the scripts (get a copy) for their use.

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
nodes. Options related to the burst buffer do not depend on Haswell or
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
	sbatch --switches=1@60 job.sh
	```

!!! info "Additional details and information"
	* [Cray XC Series Network (pdf)](https://www.cray.com/sites/default/files/resources/CrayXCNetwork.pdf)

## Additional information

* [sbatch documentation](https://slurm.schedmd.com/sbatch.html)
* Manual pages (`man sbatch` on NERSC systems)
