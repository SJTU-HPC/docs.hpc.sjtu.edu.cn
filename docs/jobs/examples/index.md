# Example job scripts

For details of terminology used on this page please see
our
[jobs overview](../index.md). Correct
[affinity settings](../affinity/index.md) are essential for good
performance.

## Basic MPI batch script

One MPI process per physical core.

??? example "Cori Haswell"
	```slurm
	--8<-- "docs/jobs/examples/basic-mpi/cori-haswell/basic-mpi.sh"
	```

??? example "Cori KNL"
	```slurm
	--8<-- "docs/jobs/examples/basic-mpi/cori-knl/basic-mpi.sh"
	```

## Hybrid MPI+OpenMP jobs

!!! warning 
	In Slurm each hyper thread is considered a "cpu" so the
	`--cpus-per-task` option must be adjusted accordingly. Generally
	best performance is obtained with 1 OpenMP thread per physical
	core. [Additional details about affinity settings](../affinity/index.md).

### Example 1

One MPI process per socket and 1 OpenMP thread per
physical core

??? example "Cori Haswell"
	```slurm
	--8<-- "docs/jobs/examples/hybrid-mpi-openmp/cori-haswell/hybrid-mpi-openmp.sh"
	```

??? example "Cori KNL"
	```slurm
	--8<-- "docs/jobs/examples/hybrid-mpi-openmp/cori-knl/hybrid-mpi-openmp.sh"
	```

### Example 2

28 MPI processes with 8 OpenMP threads per process, each OpenMP thread
has 1 physical core

!!! note
	The addition of `--cpu-bind=cores` is useful for getting correct
	[affinity settings](../affinity/index.md).

??? example "Cori Haswell"
	```slurm
	--8<-- "docs/jobs/examples/hybrid-mpi-openmp/cori-haswell/example2.sh"
	```

??? example "Cori KNL"
	```slurm
	--8<-- "docs/jobs/examples/hybrid-mpi-openmp/cori-knl/example2.sh"
	```

## Interactive

Interactive jobs are launched with the `salloc` command.

!!! tip
	Cori has dedicated nodes for interactive work.

??? example "Cori Haswell"
	```slurm
	cori$ salloc --qos=interactive -C haswell --time=60 --nodes=2
	```

??? example "Cori KNL"
	```slurm
	cori$ salloc --qos=interactive -C knl --time=60 --nodes=2
	```

!!! note
	[Additional details](../interactive.md) on Cori's interactive
	QOS

## Multiple Parallel Jobs Sequentially

Multiple sruns can be executed one after another in a single batch
script. Be sure to specify the total walltime needed to run all jobs.

??? example "Cori Haswell"
	```slurm
	--8<-- "docs/jobs/examples/multiple-parallel-jobs/cori-haswell/sequential-parallel-jobs.sh"
	```

## Multiple Parallel Jobs Simultaneously

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
	```slurm
	--8<-- "docs/jobs/examples/multiple-parallel-jobs/cori-haswell/simultaneous-parallel-jobs.sh"
	```

## Job Arrays

Job arrays offer a mechanism for submitting and managing collections
of similar jobs quickly and easily.

This example submits 3 jobs. Each job uses 1 node and has the same
time limit and QOS. The `SLURM_ARRAY_TASK_ID` environment variable is
set to the array index value.

!!! example "Cori KNL"
	```slurm
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

	```slurm
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

	```slurm
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

	```slurm
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

	```slurm
	#!/bin/bash
	#SBATCH --qos=shared
	#SBATCH --constraint=haswell
	#SBATCH --time=5
	#SBATCH --ntasks=1
	#SBATCH --mem=1GB

	./serial.exe
	```

## Using Intel MPI 

Applications built with Intel MPI can be launched via srun in the SLURM batch script on Cori compute nodes. The module `impi` needs to be loaded, and the application should be built using the `mpiicc` (`for C Codes`) or `mpiifort` (`for Fortran codes`) or `mpiicpc` (`for C++ codes`) commands. Below is a sample compile and run script.  

??? example "Cori Haswell"
	```slurm
	--8<-- "docs/jobs/examples/intel-mpi/cori-haswell/intel-mpi.sh"
	```

## Using Open MPI

Applications built with Open MPI can be launched via srun or Open MPI's mpirun command.  The module `openmpi` needs to be loaded to build an application against Open MPI.  Typically one builds the application using the `mpicc` (`for C Codes`),  `mpifort` (`for Fortran codes`), or `mpiCC` (`for C++ codes`) commands.  Alternatively, Open MPI supports use of `pkg-config` to obtain the include and library paths.  For example, `pkg-config --cflags --libs ompi-c` returns the flags that must be passed to the backend `c` compiler (e.g. gcc, gfortran, icc, ifort)  to build against Open MPI.  Open MPI also supports Java MPI bindings.  Use `mpijavac` to compile Java codes that use the Java MPI bindings.  For Java MPI, it is highly recommended to launch jobs using Open MPI's mpirun command.  Note the Open MPI packages at NERSC do not support static linking.

See [Open MPI](../../programming/programming-models/mpi/openmpi.md) for more
information about using Open MPI on NERSC systems.

??? example "Cori Haswell Open MPI"
	```slurm
	--8<-- "docs/jobs/examples/basic-mpi/cori-haswell-open-mpi/basic-mpi.sh"
	```

??? example "Cori KNL Open MPI"
	```slurm
	--8<-- "docs/jobs/examples/basic-mpi/cori-knl-open-mpi/basic-mpi.sh"
	```

## Xfer queue

The intended use of the xfer queue is to transfer data between compute
systems and HPSS. The xfer jobs run on one of the login nodes and are
free of charge. If you want to transfer data to the HPSS archive
system at the end of a regular job, you can submit an xfer job at the
end of your batch job script via `module load esslurm; sbatch hsi put
<my_files>` (be sure to load the `esslurm` module first, or you'll end
up in the regular queue), so that you will not get charged for the
duration of the data transfer. The xfer jobs can be monitored via
`module load esslurm; squeue`. The number of running jobs for each
user is limited to the number of concurrent HPSS sessions (15).

!!! warning
    Do not run computational jobs in the xfer queue.

??? example "Xfer transfer job"
    ```slurm
    #!/bin/bash
    #SBATCH --qos=xfer
    #SBATCH --time=12:00:00
    #SBATCH --job-name=my_transfer
    #SBATCH --licenses=SCRATCH

    #Archive run01 to HPSS
    htar -cvf run01.tar run01

    #Submit job with
    #module load esslurm
    #sbatch <job_script>
    ```

Xfer jobs specifying `-N nodes` will be rejected at submission
time. When submitting an Xfer job from Cori, the `-C haswell` is not
needed since the job does not run on compute nodes. By default, xfer
jobs get 2GB of memory allocated. The memory footprint scales
somewhat with the size of the file, so if you're archiving larger
files, you'll need to request more memory. You can do this by adding
`#SBATCH --mem=XGB` to the above script (where X in the range of 5 -
10 GB is a good starting point for large files).

To monitor your xfer jobs please load the `esslurm` module, then you
can use Slurm commands like `squeue` or `scontrol` to access the xfer
queue on Cori.

## Variable-time jobs

Variable-time jobs are for users who wish to get a better queue
turnaround and/or need to run long running jobs, including jobs longer
than 48 hours, the maximum wall-clock time allowed on Cori.

Variable-time jobs are jobs submitted with a minimum time, `#SBATCH
--time-min`, in addition to the maximum time (`#SBATCH –time`). 
Here is an example job script for variable-time jobs:

!!! example "Sample job script with --time-min"

    ```slurm
    --8<-- "docs/jobs/examples/variable-time-jobs/cori-knl/flex-jobs.sh"
    ```
Jobs specifying a minimum time can start execution earlier than they
would otherwise with a time limit anywhere between the minimum and
maximum time requests. Pre-terminated jobs can be requeued (or resubmitted) by using
the `scontrol requeue` command (or sbatch) to resume from where the previous
executions left off, until the cumulative execution time reaches the
desired time limit or the job completes.  

!!! note
	To use variable-time jobs, applications are required to be
	able to checkpoint and restart by themselves.
	
### Using the flex QOS for charging discount for variable-time jobs on KNL

Variable-time jobs, specifying a shorter amount of time that a job should run, 
increase backfill opportunities, meaning
users will see a better queue turnaround. 
In addition, the process of job resubmitting can be automated, 
so users can run a long job in multiple shorter chunks with a single job script (see the automated job script sample below). 
However, variable-time jobs incur checkpoint/restart overheads from splitting a longer job into multiple shorter ones. 
To compensate for this overhead and to encourage users to use Cori KNL where more backfill opportunities are available, 
we have created a flex QOS on Cori KNL (#SBATCH -q flex) with a charging discount for variable-time jobs.
See the [Queues and Policy page for Cori KNL](http://docs.nersc.gov/jobs/policy) for more details on the flex QOS. 

!!! note
        * The flex QOS is free of charge currently. The discount rate is subject to change. 
        * Variable-time jobs work with any QOS on Cori, but the
          charging discount is available only with the flex QOS on
          Cori KNL.

### Annotated example - automated variable-time jobs

Here is a sample job script for variable-time jobs, which automates
the process of executing, pre-terminating, requeuing and restarting
the job repeatedly until it runs for the desired amount of time or the
job completes.

??? example "Cori Haswell"
    ```slurm
    --8<-- "docs/jobs/examples/variable-time-jobs/cori-haswell/variable-time-jobs.sh"
    ```

!!! example "Cori KNL"
    ```slurm
    --8<-- "docs/jobs/examples/variable-time-jobs/cori-knl/variable-time-jobs.sh"
    ```

In the above example, the `--comment` option is used to enter the
user’s desired maximum wall-clock time, which could be longer than the
maximum time limit allowed by the batch system (96 hours in this
example). In addition to the time limit (`--time`), the `--time-min`
option is used to specify the minimum amount of time the job should
run (2 hours).

The script `setup.sh` defines a few bash functions (e.g.,
`requeue_job`, `func_trap`) that are used to automate the process.
The `requeue_job func_trap USR1` command executes the `func_trap`
function, which contains a list of actions to checkpoint and requeue
the job upon trapping the `USR1` signal. Users may want to modify the
scripts (get a copy) as needed, although they should work for most
applications as they are now.

The job script works as follows:

1. User submits the above job script.
2. The batch system looks for a backfill opportunity for the job. If
   it can allocate the requested number of nodes for this job for any
   duration (e.g., 3 hours) between the specified minimum time (2
   hours) and the time limit (12 hours) before those nodes are used
   for other higher priority jobs, the job starts execution.
3. The job runs until it receives a signal USR1
   (`--signal=B:USR1@<sig_time`) 60 seconds (`sig_time=60` in this
   example) before it hits the allocated time limit (3 hours).
4. Upon receiving the signal, the job checkpoints and requeues itself
   with the remaining max time limit before it gets terminated. The
   variable `ckpt_overhead` is used to specify the amount of time (in
   seconds) needed for checkpointing and requeuing the job. It should
   match the sig_time in the `--signal` option.
5. Steps 2-4 repeat until the job runs for the desired amount of
   time (96 hours) or the job completes.

!!! note
	* If your application requires external triggers or commands to do
	  checkpointing, you need to provide the checkpoint commands using
	  the variable, `ckpt_command`. It could be a script containing
	  several commands to be executed within the specified checkpoint
	  overhead time (`ckpt_overhead`).
	* Additionally, if you need to change the job input files to
      resume the job, you can do so within `ckpt_command`.
	* If your application does checkpointing periodically, like most
      molecular dynamics codes do, you don’t need 
      `ckpt_command` (just leave it blank).
	* You can send the `USR1` signal outside the job script any time
      using the `scancel -b -s USR1 <jobid>` command to terminate the
      currently running job. The job still checkpoints and requeues
      itself before it gets terminated.
	* The `srun` command must execute in the background (notice the
      `&` at the end of the srun command line and the `wait` command
      at the end of the job script), so to catch the signal (`USR1`)
      on the wait command instead of `srun`, allow `srun` to run for a
      bit longer (up to `sig_time` seconds) to complete the
      checkpointing.

### VASP example

!!! example "VASP atomic relaxation jobs for Cori KNL"
    ```slurm
    --8<-- "docs/jobs/examples/variable-time-jobs/cori-knl/vasp-relaxation-job.sh"
    ```


## MPMD (Multiple Program Multiple Data) jobs

Run a job with different programs and different arguments for each
task.  To run MPMD jobs under Slurm use `--multi-prog
<config_file_name>`.

```slurm
srun -n 8 --multi-prog myrun.conf
```

### Configuration file format

 *  Task rank

    One or more task ranks to use this configuration. Multiple values
    may be comma separated. Ranges may be indicated with two numbers
    separated with a '-' with the smaller number first (e.g. "0-4" and
    not "4-0"). To indicate all tasks not otherwise specified, specify
    a rank of '*' as the last line of the file. If an attempt is made
    to initiate a task for which no executable program is defined, the
    following error message will be produced "No executable program
    specified for this task".

 *  Executable

    The name of the program to execute. May be fully qualified pathname
    if desired.

 *  Arguments

    Program arguments. The expression "%t" will be replaced with the
    task's number. The expression "%o" will be replaced with the
    task's offset within this range (e.g. a configured task rank value
    of "1-5" would have offset values of "0-4"). Single quotes may be
    used to avoid having the enclosed values interpreted. This field
    is optional. Any arguments for the program entered on the command
    line will be added to the arguments specified in the configuration
    file.

### Example

Sample job script for MPMD jobs. You need to create a configuration
file with format described above, and a batch script which passes this
configuration file via `--multi-prog` flag in the srun command.

!!! example "Cori-Haswell"
	```slurm
    --8<-- "docs/jobs/examples/mpmd/cori-haswell/mpmd"
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

```slurm
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

```slurm
--8<-- "docs/jobs/examples/burstbuffer/stagein.sh"
```

```slurm
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

```slurm
--8<-- "docs/jobs/examples/burstbuffer/create-persistent-reservation.sh"
```

#### Use

Take care if multiple jobs will be using the reservation to not
overwrite data.

```slurm
--8<-- "docs/jobs/examples/burstbuffer/use-persistent-reservation.sh"
```

#### Destroy

Any data on the resevration at the time the script executes will be
removed.

```slurm
--8<-- "docs/jobs/examples/burstbuffer/destroy-persistent-reservation.sh"
```

### Interactive

The burst buffer is available in interactive sessions. It is
recommended to use a configuration file for the burst buffer
directives:

```bash
cori$ cat bbf.conf
#DW jobdw capacity=10GB access_mode=striped type=scratch
#DW stage_in source=/global/cscratch1/sd/username/path/to/filename destination=$DW_JOB_STRIPED/filename type=file
```

```bash
cori$ salloc --qos=interactive -C haswell -t 00:30:00 --bbf=bbf.conf
```

## Large Memory

There are two nodes on Cori with 750 GB of memory that can be used for
jobs that require very high memory per node. There are only two nodes,
so this resource is limited and should *only* be used for jobs that
require high memory. In an effort to make these useful to more users
at once, these nodes can be shared among users. If you need to run
with multiple threads, you will need to request the whole node. To do
this, add `#SBATCH --exclusive` and add the `-c 32` flag to your
`srun` call.

!!! example "Cori Example"

	A sample bigmem job which needs only one core.

	```slurm
	#!/bin/bash
	#SBATCH --clusters=escori
	#SBATCH --qos=bigmem
	#SBATCH --nodes=1
	#SBATCH --time=01:00:00
	#SBATCH --job-name=my_big_job
	#SBATCH --licenses=SCRATCH
	#SBATCH --mem=250GB

	srun -n 1 ./my_big_executable
	```

## Realtime

The "realtime" QOS is used for running jobs with the need of getting
realtime turnaround time.

!!! note
	Use of this QOS requires special approval.

	["realtime" QOS Request Form](https://nersc.service-now.com/com.glideapp.servicecatalog_cat_item_view.do?v=1&sysparm_id=d4757aa66fc8d2008ca9d15eae3ee45b&sysparm_link_parent=e15706fc0a0a0aa7007fc21e1ab70c2f&sysparm_catalog=e0d08b13c3330100c8b837659bba8fb4&sysparm_catalog_view=catalog_default)

The realtime QOS is a user-selective shared QOS, meaning you can
request either exclusive node access (with the `#SBATCH --exclusive`
flag) or allow multiple applications to share a node (with the
`#SBATCH --share` flag).

!!! tip
	It is recommended to allow sharing the nodes so more jobs can
	be scheduled in the allocated nodes.  Sharing a node is the
	default setting, and using `#SBATCH --share` is optional.

!!! example
	Uses two full nodes
	```slurm
	#!/bin/bash
	#SBATCH --qos=realtime
	#SBATCH --constraint=haswell
	#SBATCH --nodes=2
	#SBATCH --ntasks-per-node=32
	#SBATCH --cpus-per-task=2
	#SBATCH --time=01:00:00
	#SBATCH --job-name=my_job
	#SBATCH --licenses=project
	#SBATCH --exclusive

	srun --cpu-bind=cores ./mycode.exe   # pure MPI, 64 MPI tasks
	```

If you are requesting only a portion of a single node, please add
`--gres=craynetwork:0` as follows to allow more jobs on the
node. Similar to using the "shared" QOS, you can request number of
slots on the node (total of 64 CPUs, or 64 slots) by specifying the
`-ntasks` and/or `--mem`. The rules are the same as the shared QOS.

!!! example
	Two MPI ranks running with 4 OpenMP threads each.  The job
	is using in total 8 physical cores (8 "cpus" or hyperthreads per
	"task") and 10GB of memory.

	```slurm
	#!/bin/bash
	#SBATCH --qos=realtime
    #SBATCH --constraint=haswell
    #SBATCH --nodes=1
    #SBATCH --ntasks=2
	#SBATCH --gres=craynetwork:0
    #SBATCH --cpus-per-task=8
	#SBATCH --mem=10GB
    #SBATCH --time=01:00:00
    #SBATCH --job-name=my_job2
    #SBATCH --licenses=project
    #SBATCH --shared

	export OMP_NUM_THREADS=4
	srun --cpu-bind=cores ./mycode.exe
	```

!!! example
	OpenMP only code running with 6 threads. Note that `srun`
	is not required in this case.

	```slurm
    #!/bin/bash
    #SBATCH --qos=realtime
    #SBATCH --constraint=haswell
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --gres=craynetwork:0
    #SBATCH --cpus-per-task=12
	#SBATCH --mem=16GB
    #SBATCH --time=01:00:00
    #SBATCH --job-name=my_job3
    #SBATCH --licenses=project,SCRATCH
    #SBATCH --shared

    export OMP_NUM_THREADS=6
    ./mycode.exe
    ```

## Multiple Parallel Jobs While Sharing Nodes

Under certain scenarios, you might want two or more independent
applications running simultaneously on each compute node allocated to
your job. For example, a pair of applications that interact in a
client-server fashion via some IPC mechanism on-node (e.g. shared
memory), but must be launched in distinct MPI communicators.

This latter constraint would mean that MPMD mode (see below) is not an
appropriate solution, since although MPMD can allow multiple
executables to share compute nodes, the executables will also share an
`MPI_COMM_WORLD` at launch.

Slurm can allow multiple executables launched with concurrent srun
calls to share compute nodes as long as the sum of the resources
assigned to each application does not exceed the node resources
requested for the job. Importantly, you cannot over-allocate the CPU,
memory, or "craynetwork" resource. While the former two are
self-explanatory, the latter refers to limitations imposed on the
number of applications per node that can simultaneously use the Aries
interconnect, which is currently limited to 4.

Here is a quick example of an sbatch script that uses two compute
nodes and runs two applications concurrently. One application uses 8
cores on each node, while the other uses 24 on each node. The number
of CPUs per node is again controlled with the "-n" and "-N" flags,
while the amount of memory per node with the "--mem" flag. To specify
the "craynetwork" resource, we use the "--gres" flag available in both
"sbatch" and "srun".

!!! example "Cori Haswell"
	```slurm
	--8<-- "docs/jobs/examples/multiple-parallel-share-nodes/cori-haswell/multiple-parallel-share-nodes.sh"
	```

This is example is quite similar to the mutliple srun jobs shown for
[running simultaneous parallel
jobs](#multiple-parallel-jobs-simultaneously), with the
following exceptions:

1. For our sbatch job, we have requested "--gres=craynetwork:2" which
will allow us to run up to two applications simultaneously per compute
node.

2. In our srun calls, we have explicitly defined the maximum amount of
memory available to each application per node with "--mem" (in this
example 50 and 60 GB, respectively) such that the sum is less than the
resource limit per node (roughly 122 GB).

3. In our srun calls, we have also explicitly used one of the two
requested craynetwork resources per call.

Using this combination of resource requests, we are able to run
multiple parallel applications per compute node.

One additional observation: when calling srun, it is permitted to
specify "--gres=craynetwork:0" which will not count against the
craynetwork resource. This is useful when, for example, launching a
bash script or other application that does not use the
interconnect. We don't currently anticipate this being a common use
case, but if your application(s) do employ this mode of operation it
would be appreciated if you let us know.

## Additional information

* [sbatch documentation](https://slurm.schedmd.com/sbatch.html)
* Manual pages (`man sbatch` on NERSC systems)
