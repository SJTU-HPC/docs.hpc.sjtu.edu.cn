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
	make progress then you can specify the `--time-min` option to
	allow the job take advantage of backfill scheduling. This can
	result in faster throughput.
