# Batch jobs

A **job** is an allocation of resources such as compute nodes assigned to a user for an ammount of time. A batch job is a job scheduled for later execution. A job script is just a script with additional directives which tell the workload manager (Slurm) what resources are requested, which policies to apply and how to execute work on those resources.

## Options

At a minimum a job script must include number of nodes, time, type of nodes (constraint), and quality of service (QOS). If a script does not specify any of these options then a default may be applied.

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

### Defaults

| Option     | Cori       | Edison     |
|------------|------------|------------|
| nodes      | 1          | 1          |
| time       | 5minutes   | 5minutes   |
| qos        | debug      | debug      |
| constraint | haswell    | ivybridge  |
| account    | set in NIM | set in NIM |

### File system licenses

### Node type

### Quality of Service

### Account

## Submitting jobs

When you submit the job, Slurm responds with the job's ID, which will be used
to identify this job in reports from Slurm.

```console
nersc$ sbatch first-job.sh
Submitted batch job 864933
```

## Monitoring

### sqs
`sqs` is used to view job information for jobs managed by Slurm. This is a custom script 
provided by NERSC which incorportates information from serveral sources.

```console
nersc$ sqs
JOBID   ST  USER   NAME         NODES REQUESTED USED  SUBMIT               PARTITION SCHEDULED_START      REASON
864933  PD  elvis  first-job.*  2     10:00     0:00  2018-01-06T14:14:23  regular   avail_in_~48.0_days  None
```

### Email notification

```
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=user@domain.com
```

## Examples

### Basic

```bash
--8<-- "docs/jobs/examples/first-job.sh"
```

### Annotated Basic

```bash
--8<-- "docs/jobs/examples/first-job-annotated.sh"
```

### MPI jobs

### OpenMP jobs

### Hybrid MPI+OpenMP jobs

### Burst buffer

### Containerized (Docker) applications with Shifter

Detailed information about [how Shifter works](shifter/overview.md) and [how to build images](shifter/how-to-use.md) is avaiable.

```bash
#!/bin/bash
#SBATCH --image=docker:image_name:latest
#SBATCH --nodes=1
#SBATCH --qos=regular

srun -n 32 shifter python myPythonScript.py args
```

### MPMD and multi-program jobs 
