## Jobs

A **job** is an allocation of resources such as compute nodes assigned to a user for an ammount of time. Jobs can be [interactive](interactive.md) or [batch](batch.md) (e.g. a script) scheduled for later execution.

Once a job is assigned a set of nodes, the user is able to initiate parallel work in the form of job steps (sets of tasks) in any configuration within the allocation.

When you login to a NERSC cluster you land on a *login node*. Login nodes are for editing, compiling, preparing jobs. They are not for running jobs. From the login node you can interact with Slurm to submit job scripts or start interactive jobs.

## Slurm

NERSC uses [Slurm](https://slurm.schedmd.com) for cluster/resource management and job scheduling. Slurm is responsible for allocating resources to users, providing a framework for starting, executing and monitoring work on allocated resources and scheduling work for future execution.

### Commands

#### sacct

`sacct` is used to report job or job step accounting information about active or completed jobs.

#### sbatch

`sbatch` is used to submit a job script for later execution. The script will typically contain one or more srun commands to launch parallel tasks.

#### srun 

`srun` is used to submit a job for execution or initiate job steps in real time. A job can contain multiple job steps executing sequentially or in parallel on independent or shared resources within the job's node allocation.

#### sqs

`sqs` is used to view job information for jobs managed by Slurm. This is a custom script provided by NERSC which incorportates information from serveral sources.
