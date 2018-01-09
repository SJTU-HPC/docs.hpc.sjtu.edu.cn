# Batch jobs

> :bulb: **Key points**
>
> * A batch job is just a script, decorated with #SBATCH directives that tell
>   Slurm what it needs to know to schedule and run this job
>     * directives can also be specified as command line options, and those
>       take precedence over in-script directives
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

## A typical batch script

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
change to `R`, and once it completes it will no longer be shown by `sqs`.

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

## Useful `sbatch` options

You can see the full list of directives from the command line with 
`man sbatch`. Each option can be specified either as a directive in the job
script:

```bash
#!/bin/bash -l
#SBATCH -N 2
```

Or as a command line option when submitting the script:

```console
nersc$ sbatch -N 2 ./first-job.sh
```

The command line and directive versions of an option are equivalent and 
interchangeable. If the same option is present both on the command line and as
a directive, the command line will be honored. If the same option or directive
is specified twice, the last value supplied will be used.

Also, many options have both a long form, eg `--nodes=2` and a short form, eg
`-N 2`. These are equivalent and interchangable.

Many options are common to both `sbatch` and `srun`, for example 
`sbatch -N 4 ./first-job.sh` allocates 4 nodes to `first-job.sh`, and 
`srun -N 4 uname -n` inside the job runs a copy of `uname -n` on each of 4 
nodes. If you don't specify an option in the `srun` command line, `srun` will
inherit the value of that option from `sbatch`.

In these cases the default behavior of `srun` is to assume the same 
options as were passed to `sbatch`. This is acheived via environment variables:
`sbatch` sets a number of environment variables with names like `SLURM_NNODES`
and srun checks the values of those variables. This has two important 
consequences:

1. Your job script can see the settings it was submitted with by checking
   these environment variables

2. You should not override these environment variables. Also be aware that
   if your job script does certain tricky things, such as using ssh to 
   launch a command on another node, the environment might not be 
   propagated and your job may not behave correctly

### Number of nodes: -N

The number of nodes to be allocated to the job


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

