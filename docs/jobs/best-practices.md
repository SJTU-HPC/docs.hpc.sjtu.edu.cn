## Data

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

## Run short jobs

Short jobs can usually jump the queue and start quickly, long jobs
will typically queue for several days. Running many short jobs is good
way to achieve good throughput.

## Long running jobs

!!! example
	Climate codes, molecular dynamics

If you have a simulation which must run for a certain number of
iterations and the run time of the code is greater than the maximum
run time NERSC allows then you can chain server jobs together with job
dependencies.

```
cori$ jobid=$(sbatch --time=24:00:00 --qos=regular --constraint=knl job.sh | awk '{print $NF}')
cori$ sbatch --time=24:00:00 --qos=regular --constraint=knl --depend=after:$jobid restart_and_continue_job.sh
```

!!! tip
	If you know the minimum amount of time needed for your job to
	make progress then better throughput can be achieved by
	specifying the `--time-min` option. This enables the job take
	advantage of backfill scheduling opportunities.
