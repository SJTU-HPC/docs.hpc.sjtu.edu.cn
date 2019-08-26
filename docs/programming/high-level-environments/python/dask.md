# What is Dask?
**Yes, we support Dask on Cori!**

[Dask](https://dask.org) is task-based parallelization framework for Python. It
allows you, the user, to distribute your work among a collection of workers
controlled by a central scheduler.

Dask is [well-documented](https://docs.dask.org/en/latest/), flexible, and
currently under active development. For these reasons we recommend giving Dask
a try if you are trying to scale Python. A good way to start learning about
Dask is to read through or try some of their
[tutorials](https://github.com/dask/dask-tutorial). You can also watch a 15
minute Dask demo [here](https://www.youtube.com/watch?v=ods97a5Pzw0).

**Advantages of Dask**  

* Dask can run on small systems like your laptop all the way up to large
  systems like Cori. The number of workers can be easily adjusted or even
  automatically scaled.  
* It runs on both CPUs and GPUs.  
* It is robust. A nanny process can revive dead workers and the job can continue.  
* It has a very nice Bokeh status monitoring page.  
* It can be used on an interactive node or in a batch script.

# Dask on Cori
We recommend first creating your own conda environment using Python 3 which
we'll call `my_dask_env` in this example (see [here](index.md) for more info)
and then installing Dask distributed via conda-forge:
```
conda create -n my_dask_env python=3.7
source activate my_dask_env
conda config -add channels conda-forge
conda install dask distributed -c conda-forge
```
After
this you may want to remove the conda-forge channel:
```
conda config --remove channels conda-forge
```
since it is better to install from the default channel
whenever possible.

Dask can be used on Cori in either interactive or batch mode. It may be easier
to start learning how to use Dask in interactive mode and eventually switch
to batch mode once you have settled on a suitable workflow.

# Interactive Dask example
Put your client at the top of your file (we'll call it `test_dask.py`)
and tell it where to find the scheduler. Note that you have to spell out the
full path to your scheduler in your Python script rather than using
`$SCRATCH` or `$HOME`.

!!! warning "Run your Dask jobs on $SCRATCH"
    It is better to run your Dask jobs on $SCRATCH.
    Dask will try to lock the files
    associated with each worker which works automatically on
    on $SCRATCH. On $HOME, however, file locking casuses errors and you
    will see many error messages that look like:
    ```
    distributed.diskutils - ERROR - Could not acquire workspace lock on path: /global/u1/s/stephey/worker-klsptdq3.dirlock .Continuing without lock. This may result in workspaces not being cleaned up
    ```

```
from distributed import Client
client = Client(scheduler_file="/global/cscratch1/sd/stephey/scheduler.json")
```

Beneath the client in `test_dask.py`, put your Daskified code:
```

import random

def throw_darts(args):
    darts, seed = args
    random.seed(seed)
    count = 0.0
    for i in range(darts):
        x = random.random()
        y = random.random()
        if x * x + y * y < 1.0:
            count += 1.0
    return (count, darts)

futures = client.map(throw_darts, [(max(1000000, int(random.expovariate(1.0/10000000.0))), i+192837465) for i in range(100)])

results = client.gather(futures)

text_file = open("dask_output.txt", "w")
text_file.write("results is %s" %(results))
text_file.close()

print("dask test completed")
```

Here the client.map is sending parts of your throw_darts function to each Dask
worker. After the futures object is ready, client.gather will return the
results from the workers. We should note that the Dask
[Delayed](https://docs.dask.org/en/stable/delayed.html) API provides
another way to structure this problem, although we prefer the Dask
[client.futures](https://docs.dask.org/en/latest/futures.html)
API (shown in this example) since it is a little cleaner.

Now that you have prepared your Dask program `test_dask.py` we can start up a scheduler and try it out.

Request an interactive node on Cori KNL, module load Python and then
activate your conda environment where you installed Dask Distributed:
```
salloc -A your_repo -N 1 -t 30 -C knl --qos=interactive
module load python
source activate my_dask_env
```
Now you can start your Dask scheduler:
```
python -u $(which dask-scheduler) --scheduler-file $SCRATCH/scheduler.json &
```
You may need to wait 5-10 seconds for your scheduler to get started.
When it does, you'll see something like:
```
(my_dask_env) stephey@nid02304:~> distributed.scheduler - INFO - -----------------------------------------------
distributed.scheduler - INFO - Local Directory:    /tmp/scheduler-lesmw3ax
distributed.scheduler - INFO - -----------------------------------------------
distributed.scheduler - INFO - Clear task state
distributed.scheduler - INFO -   Scheduler at:    tcp://10.128.9.19:8786
distributed.scheduler - INFO -   dashboard at:                     :8787

```
Once your scheduler is ready, hit control C to get your prompt back (the `&` runs
your command in the background.) Now you are ready to start some workers.
Depending on your application you may want to have
fewer workers sharing more threads, or more workers sharing fewer threads. This
will depend on the type of workload that you have and how often you release the
GIL. For more information see
[here](https://docs.dask.org/en/latest/setup/single-machine.html).  In our
example we start 10 workers, each with a single thread.
Finally you'll need to make sure that your scheduler.json
file is the same one that is specified in your client at the top of `test_dask.py`.

```
srun -u -n 10 python -u $(which dask-worker) --scheduler-file $SCRATCH/scheduler.json --nthreads 1 &
```
Depending on how many workers you've asked for, you may need to wait a few seconds to
a few minutes for all of them to be fully ready. As your workers spin up, it will
look like this:

```
(my_dask_env) stephey@nid02316:/global/cscratch1/sd/stephey/dask> distributed.nanny - INFO -         Start Nanny at: 'tcp://10.128.9.31:44133'
distributed.nanny - INFO -         Start Nanny at: 'tcp://10.128.9.31:38545'
distributed.nanny - INFO -         Start Nanny at: 'tcp://10.128.9.31:32889'
distributed.nanny - INFO -         Start Nanny at: 'tcp://10.128.9.31:36749'
distributed.nanny - INFO -         Start Nanny at: 'tcp://10.128.9.31:41949'
distributed.nanny - INFO -         Start Nanny at: 'tcp://10.128.9.31:36999'
distributed.nanny - INFO -         Start Nanny at: 'tcp://10.128.9.31:41767'
distributed.nanny - INFO -         Start Nanny at: 'tcp://10.128.9.31:33775'
distributed.nanny - INFO -         Start Nanny at: 'tcp://10.128.9.31:40993'
distributed.nanny - INFO -         Start Nanny at: 'tcp://10.128.9.31:45925'
distributed.worker - INFO -       Start worker at:    tcp://10.128.9.31:46695
distributed.worker - INFO -          Listening to:    tcp://10.128.9.31:46695
distributed.worker - INFO -          dashboard at:          10.128.9.31:37291
distributed.worker - INFO - Waiting to connect to:     tcp://10.128.9.31:8786
distributed.worker - INFO - -------------------------------------------------
distributed.worker - INFO -               Threads:                          1
distributed.worker - INFO -                Memory:                   60.00 GB
distributed.worker - INFO -       Local Directory: /global/cscratch1/sd/stephey/dask/worker-qocnzvw2
distributed.worker - INFO - -------------------------------------------------
distributed.scheduler - INFO - Register tcp://10.128.9.31:46695
distributed.scheduler - INFO - Starting worker compute stream, tcp://10.128.9.31:46695
distributed.core - INFO - Starting established connection
distributed.worker - INFO -         Registered to:     tcp://10.128.9.31:8786
distributed.worker - INFO - -------------------------------------------------
distributed.core - INFO - Starting established connection

```
This message will continue for many lines depending on how many workers you've
requested. After your workers are ready, hit Control C to get your prompt back.

Now that your client is ready and your workers are alive and communicating via
your scheduler, you are ready to launch your Dask program!

```
python -u test_dask.py
```
Once you start your Dask program, it will look like this:
```
(my_dask_env) stephey@nid00096:/global/cscratch1/sd/stephey/dask> python -u test_dask.py
distributed.scheduler - INFO - Receive client connection: Client-d540277a-bdf5-11e9-a3fa-000101000060
distributed.core - INFO - Starting established connection
dask test completed
distributed.scheduler - INFO - Remove client Client-d540277a-bdf5-11e9-a3fa-000101000060
distributed.scheduler - INFO - Remove client Client-d540277a-bdf5-11e9-a3fa-000101000060
distributed.scheduler - INFO - Close client connection: Client-d540277a-bdf5-11e9-a3fa-000101000060
```
Congratulations! You just ran your first Dask program on Cori!

## Batch Dask example
Ok, so you're comfortable with Dask and you're ready to start submitting
Dask jobs through our batch system. Here is an example that we'll
call `dask_batch.sh` that demonstrates how
to run our same `test_dask.py` program that we showed above as a batch job.

The main difference between the interactive Dask example and the batch script
is that we have to include `sleep` commands in place of the waiting we instructed
you to do earlier. If you don't include these `sleep` commands, Dask may try to start
workers and talk to a scheduler that doesn't yet exist. The other difference
is that we need to spell out the full paths to our files rather than using
`$SCRATCH` or `$HOME`.

```
#!/bin/bash
#SBATCH -J test_dask
#SBATCH -o /global/cscratch1/sd/stephey/dask/test_dask.txt
#SBATCH --constraint=knl
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --time=10

module load python
source activate my_dask_env

python -u $(which dask-scheduler) --scheduler-file /global/cscratch1/sd/stephey/scheduler.json & sleep 30

srun -u -n 10 python -u $(which dask-worker) --scheduler-file /global/cscratch1/sd/stephey/scheduler.json --nthreads 1 &
sleep 60 &
python -u /global/cscratch1/sd/stephey/dask/test_dask.py
```

The main difference between the interactive Dask example and the batch script
is that we have to include `sleep` commands in place of the waiting we instructed
you to do earlier. If you don't include these `sleep` commands, Dask may try to start
workers and talk to a scheduler that doesn't yet exist. The other difference
is that we need to spell out the full paths to our files rather than using
`$SCRATCH` or `$HOME`.

Now just submit your Dask job script using sbatch:
```
sbatch dask_batch.sh
```
and wait for your job to run. Voila!

# Security in Dask

*Coming soon*

# Using the Dask Bokeh visualization page

*Coming soon*
