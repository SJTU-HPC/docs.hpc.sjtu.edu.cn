# Batch jobs

> :bulb: **Key points**
>
> * A batch job is just a script, decorated with #SBATCH directives that tell
>   Slurm what it needs to know to schedule and run this job
>
> * At NERSC, each job must specify:
>     * the number of nodes Slurm should allocate to this job
>     * how long it should allocate them for
>     * what type of nodes (usually knl, haswell or ivybridge)
>     * which filesystems it needs
>     * which QOS to submit to (determines priorites and cost)
>
> * Short jobs can usually jump the queue and start quickly, long jobs will
>   typically queue for several days


A basic NERSC job script looks like [this example](examples/first-job.sh):
```bash
{!jobs/examples/first-job.sh!}
```

Here's how you can submit it and check on its progress:
```console
nersc$ sbatch first-job.sh
Submitted batch job 864933
nersc$ sqs
JOBID   ST  USER   NAME         NODES REQUESTED USED  SUBMIT               PARTITION SCHEDULED_START      REASON
864933  PD  elvis  first-job.*  2     10:00     0:00  2018-01-06T14:14:23  regular   avail_in_~48.0_days  None
```

When you submit the job, Slurm responds with the job's ID, which will be used 
to identify this job in reports from Slurm.

The `PD` in the second field of the `sqs` output indicates that the job is 
"pending", that is, waiting in the queue. When it starts to run the status will
change to `R`, and once it completes it will no longer be shown by `sqs`

While you wait for first-job.sh to complete, here's a version of that same 
script with comments explaining what each part is for:

```bash
{!jobs/examples/first-job-annotated.sh!}
```

Once the job finishes, you will see a file with a name like `slurm-864934.out` 
in the directory you submitted from (the number is the job ID). This file
contains the stdout and stderr from the job: 

```console
nersc$ ls -l
total 4
-rw-rw---- 1 elvis elvis 285 Jan  5 15:52 first-job.sh
-rw-rw---- 1 elvis elvis 290 Jan  6 14:14 slurm-864933.out
nersc$ cat slurm-864933.out 
{!jobs/examples/first-job.sh.stdout!}
```

(**TODO** more text..)

## MPI jobs

**TODO** include source code and build instructions (prob via link to elsewhere in docs)
for building a simple MPI program - maybe with xthi info - and use that in these examples.
Also include links to notes about affinity, etc

## OpenMP jobs

blah blah

## Hybrid MPI+OpenMP jobs

blah blah

## Using a burst buffer for I/O-intensive jobs

blah blah

## Running containerized (Docker) applications with Shifter

blah blah

## MPMD and multi-program jobs 

multi-prog stuff

