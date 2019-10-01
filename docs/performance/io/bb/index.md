# Burst Buffer
## Performance Tuning
The NERSC Burst Buffer is based on Cray DataWarp that uses flash or
SSD (solid-state drive) technology to significantly increase the I/O
performance on Cori for all file sizes and all access patterns.

### Striping

Currently, the Burst Buffer granularity is 20GB. If you request an
allocation smaller than this amount, your files will sit on one Burst
Buffer node. If you request more than this amount, then your files
will be striped over multiple Burst Buffer nodes. For example, if you
request 19GB then your files all sit on the same Burst Buffer
server. This is important, because each Burst Buffer server has a
maximum possible bandwidth of roughly 6.5GB/s - so your aggregate
bandwidth is summed over the number of Burst Buffer servers you use. If other
people are accessing data on the same Burst Buffer server at the same
time, then you will share that bandwidth and will be unlikely to reach
the theoretical peak.

Therefore:

 * It is better to stripe your data over many Burst Buffer servers,
particularly if you have a large number of compute nodes trying to
access the data.

 * The number of Burst Buffer nodes used by an application should be
scaled up with the number of compute nodes, to keep the Burst Buffer
nodes busy but not over-subscribed. The exact ratio of compute to
Burst Buffer nodes will depend on the I/O load produced by
the application.

### Use Large Transfer Size

We have seen that using transfer sizes less than 512KiB results in
poor performance. In general, we recommend using as large a transfer
size as possible.

### Use More Processes per Burst Buffer Node

We have seen that the Burst Buffer cannot be kept busy with less than
4 processes writing to each Burst Buffer node - less than this will
not be able to achieve the peak potential performance of roughly 6.5
GB/s per node.

## Known Issues
There are a number of known issues to be aware of when using the Burst Buffer on Cori. This page will be updated as problems are discovered, and as they are fixed. 
### General Issues 
* Do not use a decimal point when you specify the burst buffer capacity - slurm does not parse this correctly and will allocate you one grain of space instead of the full request. This is easy to work around - request 3500GB instead of 3.5TB, etc.  
* Data is at risk in a Persistent Reservation if an SSD fails - there is no possibility to recover data from a dead SSD. Please back up your data! 
* If you request a too-small allocation on the Burst Buffer (e.g. request 200GB and actually write out 300GB) your job will fail, and the BB node will go into an undesirable state and need to be recovered manually. Please be careful of how much space your job requires - if in doubt, over-request. 
* If you use "dwstat" in your batch job, you may occasionally run into "[SSL: CERTIFICATE_VERIFY_FAILED]" errors, which may fail your job. If you see this error, it is due to a modulefile issue - please use the full path to the dwstat command: "/opt/cray/dws/default/bin/dwstat". 
* If you have multiple jobs writing to the same directory in a persistent reservation, you will run into race conditions due to the DataWarp caching. The second job will likely fail with "permission denied" or "No such file or directory" messages, as the metadata in the compute node cache does not match the reality of the metadata on the BB.
* If the primary SLURM controller is down, the secondary (backup) controller will be scheduling jobs - and the secondary controller does not know about the Burst Buffer. If you happen to submit a job when the backup scheduler is running your jobs will fail with the message "sbatch: error: burst_buffer/cray: Slurm burst buffer configuration error / sbatch: error: Batch job submission failed: Burst Buffer request invalid". If you receive this error and your submission script is correct, please check [MOTD](https://www.nersc.gov/live-status/motd/) for SLURM downtime/issues, and try again later. 

### Staging Issues

* The command "squeue -l -u username" will give you useful information on how your stage_in process is going. If you see an error message (e.g. "(burst_buffer/cray: dws_data_in: Error staging in session 20868 configuration 6953 path /global/cscratch1/sd/username/stagemein.txt -> /stagemein.txt: offline namespaces: [44223] - ask a system administrator to consult the dwmd log for more information") then you may have a permissions issue with the files you are trying to stage_in, or be trying to stage_in a non-existent file. 

* The Burst Buffer cannot access GPFS for staging data (copying data is fine). If you have data that will be staged in to the BB, you will need to have those files in $SCRATCH. Data in your home or project directories can be transferred using "cp" within your compute job. 
* stage_in and stage_out using access_mode=private does not work (by design). 

* If you have multiple files to stage in, you will need to tar them up and use type=file, or keep them all in one directory and use type=directory. 
* type=directory fails with large directories (>~200,000 files) due to a timeout error. In this case, consider tar-ing your files and staging in the tarball. 
* Symbolic links are not preserved when staging in, the link will be lost. 
* Staging in/out with hard links does not work.


