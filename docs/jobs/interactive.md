## Allocation

`salloc` is used to allocate resources for a job in real
time. Typically this is used to allocate resources and spawn a
shell. The shell is then used to execute srun commands to launch
parallel tasks.


### Edison

Requesting 30 minutes of time on two nodes.

```
edison$ salloc --qos=debug --nodes=2 --time=30
```

### Cori

Cori has an dedicated interactive QOS. This queue is intended to
deliver nodes for interactive use within 6 minutes of the job request.

We have deployed experimental functionality to support medium-length
interactive work on Cori. This queue is intended to deliver nodes for
interactive use within 6 minutes of the job request. To access the
interactive queue add the qos flag to your salloc command.

```
cori$ salloc -N 1 -C haswell -q interactive -t 01:00:00
```

To run on KNL nodes, use "-C knl" instead of "-C haswell".

Users in this queue are limited to a single running job on as many as
64 nodes for up to 4 hours. Additionally, each NERSC allocation (MPP
repo) is further limited to a total of 64 nodes between all their
interactive jobs (KNL or haswell). This means that if UserA in repo
m9999 has a job using 1 haswell node, UserB (who is also in repo
m9999) can have a simultaneous job using 63 haswell nodes or 63 KNL
nodes, but not 64 nodes. Since this is intended for interactive work,
each user can submit only one job at a time (either KNL or
haswell). KNL nodes are currently limited to quad,cache mode only. You
can only run single node job; sub-node jobs like those in the shared
queue are not possible.

We have configured this queue to reject the job if it cannot be
scheduled within a few minutes. This could be because the job violates
the single job per user limit, the total number of nodes per NERSC
allocation limit, or because there are not enough nodes available to
satisfy the request. In some rare cases, jobs may also be rejected
because the batch system is overloaded and wasn't able to process your
job in time. If that happens, please resubmit.

Since there is a limit on number of nodes used per allocation, you may
be unable to run a job because other users who share your allocation
are using it. To see who in your allocation is using the interactive
queue you can use

```
cori$ squeue -q interactive -A <reponame> -O jobid,username,starttime,timelimit,maxnodes,account
```

If the number of nodes in use by your repo sums up to 64 nodes, please
contact the other group members if you feel they need to release
interactive resources.
