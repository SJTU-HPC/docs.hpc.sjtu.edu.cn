# TaskFarmer

TaskFarmer is a utility developed in-house at NERSC to farm tasks onto a compute 
node - these can be single- or multi-core tasks. It tracks which tasks have 
completed successfully, and allows straightforward re-submission of failed or 
un-run jobs from a task list. 

The base functionality is contained within the runcommands.sh script which is
provided by Taskfarmer.  The script will be added to your path after loading the
Taskfarmer module. This script launches a server on the head node of your
compute allocation that will keep track of the tasks on your list, and the
workers running on cores of the other compute nodes in your batch job (note that
this means that you always need at least 2 nodes in your TaskFarmer batch job -
more on this later). The workers check the $THREADS environmental variable (this
is a TaskFarmer variable, not a SLURM variable) and spins up that many threads
to run tasks. By default, this is set to the number of available cores on a
compute node. Each thread requests a task from the server, is assigned the next
task in the task list, then forks off to run the task. Once the task is complete
it communicates with the server and requests the next task. 

## Things to note: 

* The TaskFarmer server requires a full compute node, so you will need to request minimum 2 nodes in your batch script. 
* The total amount of time you request should be equal to (number of tasks*task time)/$THREADS, not simply the time required to run one task.
* TaskFarmer can handle roughly 5-10 tasks per second before communication between the server and workers becomes a bottleneck (this is why the server requires a full node to run). 
  This will impact how you set up your task list - so for example, if you have tasks that require 100 seconds to run then TaskFarmer can handle about 1000 tasks per second. 
  Note that TaskFarmer is designed to handle jobs that require tens of minutes to hours, and performs optimally with longer jobs. 

## How to run TaskFarmer

### Step 1: Define your task

Write a wrapper that defines one task that you're going to run. 
It should contain the executable and any options required - these will be defined in the next step. 
Note that this example has a script that is sitting in a directory called "taskfarmer" on a users scratch space. 
Please edit this example to point to your own script. 

```shell
cd $SCRATCH/taskfarmer
python calcBinary.py $1 $2 $3
```

### Step 2: Create a task list
This is where you can list all the tasks you need, including all job options. wrapper.sh refers to the wrapper script you created in the previous step. 

    wrapper.sh 0 0 1 
    wrapper.sh 0 1 0 
    wrapper.sh 0 1 1 
    wrapper.sh 1 0 0 
    wrapper.sh 1 0 1 
    wrapper.sh 1 1 1

### Step 3: Batch script
Your batch script will specify the total time required to run all your tasks and how many nodes you want to use. 
Assuming you are running on the Cori Haswell partition, note that you need to specify the "-c 64" option to ensure that all 64 hyperthreads are available to the workers, and export THREADS=32 to ensure that each task runs on one thread. 
The "-N 2" requests two compute nodes - one will run the TaskFarmer server, and the other will run the tasks. 
tasks.txt is the tasklist you created in the previous step. 

```shell
#!/bin/sh
#SBATCH -N 2 -c 64
#SBATCH -p debug
#SBATCH -t 00:05:00
#SBATCH -C haswell 
cd $SCRATCH/taskfarmer 
export PATH=$PATH:/usr/common/tig/taskfarmer/1.5/bin:$(pwd)
export THREADS=32
runcommands.sh tasks.txt
```

### Step 4: Run it
You will need to load the taskfarmer module, then you can simply submit your batch script. 
batch.sl is the batch script you created in the previous step. 

```shell
module load taskfarmer
sbatch batch.sl
```

## Output 

You will find several files appear in your job submission directory. 
Their names will depend on the name of the tasklist you created in step 2. 

* tasks.txt.tfin: Repetition of tasklist, formatted for use by TaskFarmer. 
 progress.tasks.txt.tfin: A space-delimited file that captures tracking information for each successful task. 
 A line count of this file will show you how many tasks have completed.  
The first field is the byte offset into the formatted task list file. 
The second field is the node/thread that ran the task. 
The third field is the runtime in seconds for the task. 
The fourth field can be ignored. 
The fifth field is the completion time in UNIX time format. 
The sixth field can be ignored.
* log.tasks.txt.tfin: Any error messages from the tasks. This is useful for identifying errors for particular tasks.
* fast recovery.tasks.txt.tfin: This is a checkpoint file that tracks which tasks have not yet completed. If your job fails or hits wall time, you can re-run TaskFarmer with the same options and it will re-run any in-flight tasks and resume progress.
* done.tasks.txt.tfin: Produced once all tasks are completed. 

## Resuming and Rerunning a Taskfarmer Job 

TaskFarmer is designed to recovery easily from failures or wall-time limits.  
For example, if a job hits a wall-time limit, you can resubmit the same batch script and it will resume operations.  
A "done" file will be created once the job has finished (all tasks have completed successfully).  
If you wish to re-run the same job, delete the progress and done files (e.g. done.tasks.txt.tfin and progresss.tasks.txt.tfin).

