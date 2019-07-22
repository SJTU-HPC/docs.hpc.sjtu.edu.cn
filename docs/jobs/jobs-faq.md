# FAQs in Running Jobs

Below are some frequently asked questions and answers for running jobs on Cori.

## General Questions

**Q: How do I check for how many free nodes are available in each
partition?**

**A**: Below is a sample Slurm command `sinfo` with the selected
output fields.  Column 1 shows the partition name, Column 2 shows
the status of this partition, Column 3 shows the max wall time limit
for this partition, and Column 4 shows the number of nodes
Available/Idle/Other/Total in this partition.

```shell
cori$ sinfo -o "%.10P %.10a %.15l %.20F"
 PARTITION      AVAIL       TIMELIMIT       NODES(A/I/O/T)
    system         up      1-12:00:00   11611/412/53/12076
    debug*         up           30:00    11299/77/42/11418
   jupyter         up      4-00:00:00            0/10/0/10
   regular         up      4-00:00:00    11199/15/40/11254
  regularx         up      2-00:00:00    11295/77/42/11414
      resv         up     14-00:00:00   11611/412/53/12076
resv_share         up     14-00:00:00     2159/210/19/2388
 benchmark         up      1-12:00:00    11299/77/42/11418
realtime_s         up        12:00:00      1847/75/12/1934
  realtime         up        12:00:00    11299/89/42/11430
    shared         up      2-00:00:00            59/0/1/60
interactiv         up         4:00:00         93/282/9/384
  genepool         up      3-00:00:00         160/31/1/192
genepool_s         up      3-00:00:00         160/31/1/192
```

**Q: How do I check for how many Haswell and KNL nodes are idle
now?**

**A**: Below is a sample Slurm command `sinfo` with the selected
output fields.  Column 1 shows the available computer node features
(such as Haswell or KNL),  and Column 2 shows the number of nodes
Available/Idle/Other/Total in this partition.  Both `knl` and
`knl,cache,quad` are KNL quad cache nodes.

```shell
cori$ sinfo -o "%.20b %.20F"
     ACTIVE_FEATURES       NODES(A/I/O/T)
                 knl              0/0/6/6
             haswell     2138/231/19/2388
      knl,cache,quad     9412/242/28/9682
```

**Q: How many interactive QOS nodes are available that I can use?**

**A**: Each repo can use up to total 64 nodes (combining Hawell and
KNL nodes).  Run the following command to see how many interactive
nodes are being used by the members of your repo:

```shell
cori$ squeue --qos=interactive --account=<reponame> -O jobid,username,starttime,timelimit,maxnodes,account
```

If the number sums up to 64 nodes, please contact the other group
members if you feel they need to release interactive resources.


**Q: Could I run jobs using both Haswell and KNL compute nodes?**

**A**: Currently only available for certain NERSC and Cray staff
for benchmarking.  We will evaluate whether (and if yes, how) to
make it available for general NERSC users.

**Q: How do I improve my I/O performance?**

**A**: Consider using the Burst Buffer - this is a tier of SSDs
that sits inside the Cori HSN, giving high-performance I/O. See
[this page](../performance/io/bb/index.md) for more
details. If you are using the `$SCRATCH` file system, take a look
at [this page](../performance/io/lustre/index.md)  for
I/O optimization tips.


## Job Errors

Some common errors encountered during submit or run times
and their possible causes are shown in the following table.

### Job submission errors

-   Error message:

    ```
    sbatch: error: Your account has exceeded a filesystem quota and is not permitted to submit batch jobs.  Please run `myquota` for more information.
    ```

    Possible causes/remedies:

    Your filesystem usage is over the quota(s). Please run the
    `myquota` command to see which quota is exceeded. Reduce usage
    and resubmit the job.

-   Error message:

    ```
    sbatch: error: Job request does not match any supported policy.
    sbatch: error: Batch job submission failed: Unspecified error.
    ```

    Possible causes/remedies:

    There is something wrong with your job submission parameters
    and the request (the requested number of nodes, the walltime
    limit, etc.) does not match the policy for the selected qos.
    Please check the [queue policy](policy.md).

    This error also happens when the job submission didn't include
    the `--time-min` line with the `flex`.

-   Error message:

    ```
    sbatch: error: More resources requested than allowed for logical queue shared (XX requested core-equivalents > YY) 
    sbatch: error: Batch job submission failed: Unspecified error
    ```

    Possible causes/remedies:

    The number of logical cores that your job tries to use exceeds
    the the number that you requested for this `shared` qos job.

-   Error message:

    ```
    Job submit/allocate failed: Unspecified
    ```

    Possible causes/remedies:

    This error could happen if a user has no active repo on Cori.
    Please make sure your NERSC account is renewed with an active
    allocation.

-   Error message:

    ```
    Job sumit/allocate failed: Invalid qos specification
    ```

    Possible causes/remedies:

    This error mostly happens if a user has no access to certain
    Slurm qos.  For example, a user who doesn't have access to the
    `realtime` qos would see this error when submitting a job to
    the qos.

-   Error message:


    ```
    sbatch: error: Batch job submission failed: Socket timed out on send/recv operation
    ```

    Possible causes/remedies:

    The job scheduler is busy. Some users may be submitting lots
    of jobs in a short time span. Please wait a little bit before
    you resubmit your job.

    This error normally happens when submitting a job, but can
    happen during runtimes, too.

-   Error message:

    ```
    cori$ salloc ... --qos=interactive
    salloc: Pending job allocation XXXXXXXX 
    salloc: job XXXXXXXX queued and waiting for resources 
    salloc: error: Unable to allocate resources: Connection timed out
    ```

    Possible causes/remedies:

    The interactive job could not start within 6 minutes, and,
    threfore, was cancelled. It is because either the number of
    available nodes left from all the reserved interactive nodes
    or out of the 64 node limit per repository was less than what
    you requested.

-   Error message:

    ```
    sbatch: error: No architecture specified, cannot estimate job costs.
    sbatch: error: Batch job submission failed: Unspecified error
    ```

    Possible causes/remedies:

    Your job didn't specify the type of compute nodes. To run on
    Hawell nodes, add to your batch script:

    ```
    #SBATCH -C haswell
    ```

    To request KNL nodes, add this line:

    ```
    #SBATCH -C knl
    ```

-   Error message:

    ```
    sbatch: error: The scavenger logical queue requires an lower balance than the estimated job cost. Job cost estimated at XX.XX NERSC-Hours, your balance is YYYYYY.YY NERSC-Hours (Repo: YYYYYYY.YY NERSC-Hours). Cannot proceed, please see https://docs.nersc.gov/jobs/policy/ for your options to run this job.
    sbatch: error: Batch job submission failed: Unspecified error
    ```

    Possible causes/remedies:

    You submitted the job to the `scavenger` partition directly.
    When you submit a job with a normal qos (e.g., `regular`,
    `debug`, etc.), requesting more NERSC-Hours than your repo
    balance, it will be automatically routed to the `scavenger`
    queue.

-   Error message:

    ```
    sbatch: error: No available NIM balance information for user xxxxx, account yyyyy. Cannot proceed.
    sbatch: error: Batch job submission failed: Unspecified error
    ```

    Possible causes/remedies:

    You submitted the job using a repo that you are not allowed to use.
    Login in to your NIM account to see which repo you can use.

-   Error message:

    ```
    sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)
    ```

    Possible causes/remedies:

    There may be an issue with Slurm. If the error is still seen after
    a few minutes, report to NERSC.

-   Error message:

    ```
    sbatch: error: Job cost estimated at XXXXXXXX.XX NERSC-Hours, your balance is XXXXXXX.XX NERSC-Hours (Repo: XXXXXXXX.XX NERSC-Hours). Cannot proceed, please see https://docs.nersc.gov/jobs/policy/ for your options to run this job.
    sbatch: error: Job submit/allocate failed: Unspecified error
    ```

    Possible causes/remedies:

    Your remaining repo balance is not big enough to run the job.

-   Error message:

    ```
    srun: error: Unable to create step for job XXXXXXXX: More processors requested than permitted
    ```

    Possible causes/remedies:

    Your `srun` command required more logical cores than available.
    Please check the values for the `-n`, `-c`, etc.


### Runtime errors

-   Error message:

    ```
    srun: error: eio_handle_mainloop: Abandoning IO 60 secs after job shutdown initiated.
    ```

    Possible causes/remedies:

    Slurm is giving up waiting for stdout/stderr to finish. This
    typically happens when some rank ends early while others are
    still wanting to write. If you don't get complete stdout/stderr
    from the job, please resubmit the job.

-   Error message:

    ```
    Tue Jul 17 18:04:24 2018: [PE_3025]:_pmi_mmap_tmp: Warning bootstrap barrier failed: num_syncd=3, pes_this_node=68, timeout=180 secs
    ```

    Possible causes/remedies:

    Use the `sbcast` command to transmit the executable to all
    compute nodes before a srun command.

-   Error message:

    ```
    slurmstepd: error: _send_launch_resp: Failed to send RESPONSE_LAUNCH_TASKS: Resource temporarily unavailable
    ```

    Possible causes/remedies:

    This situation does not affect the job. This issue may have been fixed.

[comment]: <> (-   Error message:)
[comment]: <> ()
[comment]: <> (    ```)
[comment]: <> (    libgomp: Thread creation failed: Resource temporarily unavailable)
[comment]: <> (    ```)
[comment]: <> ()
[comment]: <> (    Possible causes/remedies:)
[comment]: <> ()
-   Error message:

    ```
    srun: fatal: Can not execute vasp_gam
    /var/spool/slurmd/job15816716/slurm_script: line 17: 34559 
    Aborted                 srun -n 32 -c8 --cpu-bind=cores vasp_gam
    ```

    Possible causes/remedies:

    The user does not belong to a VASP group. Please the user needs
    to provide VASP license info following the instructions in
    [here](../applications/vasp/index.md#access).

-   Error message:

    ```
    cori$ sqs
    JOBID     ST  ...  REASON
    XXXXXXXX  PD  ...  Nodes required*
    ...
    ```

    or

    ```
    cori$ scontrol show job XXXXXXXX 
    ... 
    JobState=PENDING Reason=Nodes_required_for_job_are_DOWN,_DRAINED_or_reserved_for_jobs_in_higher_priority_partitions Dependency=(null) 
    ...
    ```

    Possible causes/remedies:

    The job was tentatively scheduled to start as a backfill job.
    But some of the assigned nodes are now down, drained or re-assigned
    to a higher priority job. Wait until Slurm reschedules the job.

-   Error message:

    ```
    srun: error: Unable to create step for job XXXXXXXX: Job/step already completing or completed
    ```

    which appears with or without this message:

    ```
    srun: Job XXXXXXXX step creation temporarily disabled, retrying
    ```

    Possible causes/remedies:

    This may be caused by a system issue. Please report to NERSC.

-   Error message:

    ```
    Tue Sep 18 20:13:26 2018: [PE_5]:inet_listen_socket_setup:inet_setup_listen_socket: bind failed port 63725 listen_sock = 4 Address already in use
    Tue Sep 18 20:13:26 2018: [PE_5]:_pmi_inet_listen_socket_setup:socket setup failed
    Tue Sep 18 20:13:26 2018: [PE_5]:_pmi_init:_pmi_inet_listen_socket_setup (full) returned -1
    ...
    ```

    Possible causes/remedies:

    Typically this error indicates that multiple applications have
    been launched on the same node, and both are using the same PMI
    (Process Management Interface, which supports launching and
    managing the processes that make up the execution of a parallel
    program; see the `intro_pmi` man page for more info) control
    port number. When running multiple applications per node, it
    is the launcher's responsibility to provide PMI with a new
    (available) port nuber by setting the `PMI_CONTROL_PORT` env
    variable. Slurm typically does this.

    You can try either of the following approaches to suppress the
    PMI errors because your code runs on a single node and does not
    communicate between nodes:

    -   Recompile your code with the `craype-network-aries` module
	unloaded, then the PMI library will not be linked into your
	code:

        ```shell
        cori$ module swap craype-network-aries craype-network-none
        ```

    -   Set the fowllowing two env variables:

        ```shell
        cori$ export PMI_NO_FORK=1
        cori$ export PMI_NO_PREINITIALIZE=1
        ```

-   Error message:

    ```
    /some/path/ ./a.out error while loading shared libraries: /opt/gcc/7.3.0/snos/lib64/libgomp.so.1: cannot read file data:
    Input/output error
    ...
    ```

    Possible causes/remedies:

    A possible cause is that the `LD_LIBRARY_PATH` environment
    varialbe has been modified.  Since `libgomp.so.1` is part of
    the Intel libraries, you can try unloading the `gcc` module
    with `module unload gcc` if it is loaded, or reloading the
    `intel` module with `module load intel`.

[comment]: <> (-   Error message:)
[comment]: <> ()
[comment]: <> (    ```)
[comment]: <> (    sacct: error: slurm_persist_conn_open: failed to send persistent connection init message to corique01:6819)
[comment]: <> (    sacct: error: slurmdbd: Getting response to message type 1444)
[comment]: <> (    sacct: error: slurmdbd: DBD_GET_JOBS_COND failure: Unspecified error)
[comment]: <> (    ```)
[comment]: <> ()
[comment]: <> (    or)
[comment]: <> ()
[comment]: <> (    ```)
[comment]: <> (    sacct: error: slurm_persist_conn_open_without_init: failed to open persistent connection to corique01:6819: Connection refused)
[comment]: <> (    ```)
[comment]: <> ()
[comment]: <> (    Possible causes/remedies:)
[comment]: <> ()

-   Error message:

    ```
    slurmstepd: error: Detected zonesort setup failure: Could not open job cpuset (########.#)
    ```

    Possible causes/remedies:

    KNL's MCDRAM cache is prone to cache thrashing because it uses
    direct mapped caching, which can result in slow code performance.
    To alleviate this possiblity, the system's Node Health Check
    tool runs the 'zonesort' kernel module on compute nodes. For
    more info, please see [KNL Cache
    Mode](../performance/knl/cache-mode.md). Note that the zonesort
    module is also run on Haswell nodes although performance
    implication may not be as significant since direct mapped caching
    is not used.

    The error message means that running the zonesort kernel failed
    for some reason. The end result is that your code may have run
    less optimally. Other than that, the message is usually harmless.
    If your job failed because the application ran slowly, please
    resubmit the job.
