## Jobs

A **job** is an allocation of resources such as compute nodes assigned
to a user for an ammount of time. Jobs can be interactive or batch
(e.g. a script) scheduled for later execution.

Once a job is assigned a set of nodes, the user is able to initiate
parallel work in the form of job steps (sets of tasks) in any
configuration within the allocation.

When you login to a NERSC cluster you land on a *login node*. Login
nodes are for editing, compiling, preparing jobs. They are not for
running jobs. From the login node you can interact with Slurm to
submit job scripts or start interactive jobs.

NERSC supports a diverse workload including high-throughput serial
tasks, full system capability simulations and complex workflows.

## Slurm

NERSC uses [Slurm](https://slurm.schedmd.com) for cluster/resource
management and job scheduling. Slurm is responsible for allocating
resources to users, providing a framework for starting, executing and
monitoring work on allocated resources and scheduling work for future
execution.

### Commands

#### sacct

`sacct` is used to report job or job step accounting information about
active or completed jobs.

#### sbatch

`sbatch` is used to submit a job script for later execution. The
script will typically contain one or more srun commands to launch
parallel tasks.

When you submit the job, Slurm responds with the job's ID, which will
be used to identify this job in reports from Slurm.

```
nersc$ sbatch first-job.sh
Submitted batch job 864933
```

#### srun

`srun` is used to submit a job for execution or initiate job steps in
real time. A job can contain multiple job steps executing sequentially
or in parallel on independent or shared resources within the job's
node allocation.

#### sqs

`sqs` is used to view job information for jobs managed by Slurm. This
is a custom script provided by NERSC which incorportates information
from serveral sources.

```
nersc$ sqs
JOBID   ST  USER   NAME         NODES REQUESTED USED  SUBMIT               PARTITION SCHEDULED_START      REASON
864933  PD  elvis  first-job.*  2     10:00     0:00  2018-01-06T14:14:23  regular   avail_in_~48.0_days  None
```

### Options

At a minimum a job script must include number of nodes, time, type of
nodes (constraint), and quality of service (QOS). If a script does not
specify any of these options then a default may be applied.

The full list of directives is documented in the man pages for the
`sbatch` command (see. `man sbatch`). Each option can be specified
either as a directive in the job script:

```bash
#!/bin/bash -l
#SBATCH -N 2
```

Or as a command line option when submitting the script:

```
nersc$ sbatch -N 2 ./first-job.sh
```

The command line and directive versions of an option are equivalent
and interchangeable. If the same option is present both on the command
line and as a directive, the command line will be honored. If the same
option or directive is specified twice, the last value supplied will
be used.

Also, many options have both a long form, eg `--nodes=2` and a short
form, eg `-N 2`. These are equivalent and interchangable.

Many options are common to both `sbatch` and `srun`, for example
`sbatch -N 4 ./first-job.sh` allocates 4 nodes to `first-job.sh`, and
`srun -N 4 uname -n` inside the job runs a copy of `uname -n` on each
of 4 nodes. If you don't specify an option in the `srun` command line,
`srun` will inherit the value of that option from `sbatch`.

In these cases the default behavior of `srun` is to assume the same
options as were passed to `sbatch`. This is acheived via environment
variables: `sbatch` sets a number of environment variables with names
like `SLURM_NNODES` and srun checks the values of those
variables. This has two important consequences:

1. Your job script can see the settings it was submitted with by
   checking these environment variables

2. You should not override these environment variables. Also be aware
   that if your job script does certain tricky things, such as using
   ssh to launch a command on another node, the environment might not
   be propagated and your job may not behave correctly

#### Defaults

| Option     | Cori       | Edison     |
|------------|------------|------------|
| nodes      | 1          | 1          |
| time       | 10minutes   | 10minutes   |
| qos        | debug      | debug      |
| constraint | haswell    | ivybridge  |
| account    | set in NIM | set in NIM |


#### Email notification

```bash
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=user@domain.com
```
