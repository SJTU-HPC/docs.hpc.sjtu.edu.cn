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

**(Q) Could I run jobs using both Haswell and KNL compute nodes?**

**A**: Currently only available for certain NERSC and Cray staff
for benchmarking.  We will evaluate whether (and if yes, how) to
make it available for general NERSC users.

**Q: How do I improve my I/O performance?**

**A**: Consider using the Burst Buffer - this is a tier of SSDs
that sits inside the Cori HSN, giving high-performance I/O. See
[this page](https://docs.nersc.gov/performance/io/bb/) for more
details. If you are using the `$SCRATCH` file system, take a look
at [this page](https://docs.nersc.gov/performance/io/lustre/)  for
I/O optimization tips.


## Job Error Messages

Some common error messages encountered during submit or run times
and their possible causes are shown in the following table.

-   **Error message**:

    ```
    Job submit/allocate failed: Unspecified
    ```

    **Possible causes/remedies**:

    Happen during job submission time.

    This error could happen if a user has no active repo on Cori.
    Please make sure your NERSC account is renewed with an active
    allocation.

-   **Error message**:

    ```
    Job sumit/allocate failed: Invalid qos specification
    ```

    **Possible causes/remedies**:

    Happen during job submission time.

    This error mostly happens if a user has no access to certain
    Slurm qos.  For example, when only NESAP users had access to a
    qos allowing job submission to the regular partition for over
    2 hrs of run time on KNL nodes, non-NESAP users would see this
    error when submitting a job longer than 2 hours.

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
