Temporary, flexible, high-performance SSD file system that sits within the High Speed Network (HSN) on Cori. Accessible only from compute nodes, the Burst Buffer provides per-job (or short-term) storage for I/O intensive codes. 

## Usage

Access to the Burst Buffer resource is integrated with the Scheduler of the system which provides the ability to provision the Burst Buffer resource to be shared by a set of users or jobs. Using the Burst Buffer on Cori can be as simple as adding a single line to your slurm batch script.

### Scratch

The fast IO available with the Burst Buffer makes it an ideal scratch space to store temporary files during the execution of an IO-intensive code. 

!!!note
	*All* files on the Burst Buffer allocation will be removed when the job ends - so you will need to *stage_out* any data that you want to retain at the end of your job. 
	
To use the Burst Buffer as scratch, add a `#DW jobdw` command to your slurm batch script. It is required to specify an *access mode*, *type*, *capacity* and *pool* option for the `#DW jobdw` command.

!!!example
	```
	#!/bin/bash
	#SBATCH -q regular
	#SBATCH -N 1
	#SBATCH -C haswell
	#SBATCH -t 01:00:00
	#DW jobdw capacity=10GB access_mode=striped type=scratch
	#DW stage_in source=/global/cscratch1/sd/username/path/to/filename destination=$DW_JOB_STRIPED/filename type=file
	srun a.out INSERT_YOUR_CODE_OPTIONS_HERE
	```

#### Access Mode

* **Striped access** mode means your data will be striped across several DataWarp nodes. 

* **Private access** mode means each of your compute jobs will have their own, private space on the Burst Buffer that will not be visible to any other compute job. Data will be striped across Burst Buffer nodes in private mode.

!!!note
	All compute nodes will share the allocation. If one compute node fills up all the space then the other compute nodes will run out and you will see "out of space" errors.

#### Pools

The Burst Buffer has two levels of granularity in two different pools - 80GiB (wlm_pool, default) and 20GiB (sm_pool). This is the minimum allocation on each Burst Buffer SSD. 

!!!example
	You wish to use 1.2TiB of Burst Buffer space in the wlm_pool. This will be striped across 15 SSDs, each holding 80GiB (note there is no guarantee that your allocation will be spread evenly between SSDs). Burst Buffer nodes are allocated on a round-robin basis.

#### Type

The only type currently available is type=scratch. 

#### Capacity

This is the minimum allocation on each Burst Buffer SSD is determined by the selected pool (20 GB for the sm_pool and 80 GB for the wlm_pool).

### Stage in

If you will be accessing a file multiple times in your code, or if you have large input files that would benefit from a faster IO, you should consider using the stage_in feature of the Burst Buffer. This will copy the named file or directory into the Burst Buffer, which can then be accessed using `$DW_JOB_STRIPED`. Currently, only the Cori scratch file system is accessible from the Burst Buffer, so only files on `$SCRATCH` can be staged in. Also be aware that you cannot stage_in to a Burst Buffer reservation in private mode.

Stage_in/out differs from a simple filesystem `cp` in that the Burst Buffer nodes transfer the files directly to the Burst Buffer SSDs, without going through the compute nodes (as is the case with `cp`), so it is *significantly* faster. In addition, your files will be staged in before the start of the compute job - so you are not charged compute time for the time taken to stage the data in.

* You can stage_in (and stage_out) both files and directories - use `type=file` or `type=directory` accordingly. 
* You need to have permission to access the files - if you try to stage_in someone else's files without sufficient permissions, you may see errors. 
* Use the full path to your scratch directory and not `$SCRATCH` (or other environmental variables), as the datawarp software will not be able to interpret the environmental variable.
* If you don't specify a destination directory name on the Burst Buffer when using type=directory, you will find the files contained in your source directory have been copied over to the Burst Buffer - not the directory itself. 
* Your compute job will wait in the queue until the data is staged in to the Burst Buffer so that your data will be available as soon as the compute job starts. If you have a large amount of data (e.g. TB-scale) or many (e.g. millions) of files to stage in, this may take tens of minutes, so your compute job could pend for longer than you'd expect based on your queue priority. In this case, *just be patient!*

### Stage out

The Burst Buffer can stage out files to your scratch directory (this is not available for a Burst Buffer allocation in private mode). This stage_out will happen after the end of your compute job, and you are not charged compute time for it. Be aware that your staged-out data may not appear immediately when your compute job finishes.

### Persistent Reservations

If you have multiple jobs that need to access the same files across many jobs, you can use a persistent reservation. This creates a space on the Burst Buffer that will persist after the end of your job, and can be accessed by subsequent jobs. 

!!!warning
	You must delete the reservation at the end of your use of it, to free up the space for other users. The Burst Buffer is not intended for long-term storage and we do not guarantee your files will be recoverable over long periods of time.
	
#### Create a reservation

You can create the named persistent reservation in a standalone job or at the start of your regular batch script. The only type of reservation available is with striped access and of type scratch. Note that creating (and destroying) the persistent reservation uses a "#BB" prefix rather than the "#DW" used in other examples. This is because the command is interpreted by Slurm and not by the Cray Datawarp software. This can result in the batch job behaving in a way you may not expect. As soon as the scheduler reads the job, the Burst Buffer resource is scheduled, even though the job has not yet executed. This means the the persistent reservation will be available shortly after you submit the batch job, even if the job is not scheduled to execute for many hours. It also means that canceling the job after the reservation has been created will not cancel the reservation. If unsure, please use `scontrol show burst` to see what reservations have been created on the Burst Buffer.

* A persistent reservation requires a unique name - this must be a unique name otherwise the allocation will fail. Please avoid using simple names like "offline" which can cause confusion. 
* You can create a persistent reservation using a batch script that does no compute work at all - but you still need to submit it to the batch system and therefore request compute time, even if it's not used.

!!!example
	```
	#!/bin/bash
	#SBATCH -q debug
	#SBATCH -N 1
	#SBATCH -C haswell
	#SBATCH -t 00:05:00
	#BB create_persistent name=myBBname capacity=100GB access_mode=striped type=scratch
	```

#### Use the reservation

Tell slurm that you will be accessing the Burst Buffer reservation using the datawarp command and the name of your persistent reservation with #DW persistentDW name=myBBname. The path to your Burst Buffer reservation can then be accessed during your job using `$DW_PERSISTENT_STRIPED_myBBname`, where "myBBname" is the name you gave your persistent reservation when you created it. In the example below, you could pass this path to your executable as an option (note that you have to actually write your code so that it accepts such an option, it will not magically know where your data sits).

!!!example
	```
	#!/bin/bash
	#SBATCH -q debug
	#SBATCH -N 1
	#SBATCH -C haswell
	#SBATCH -t 00:05:00
	#DW persistentdw name=myBBname
	mkdir $DW_PERSISTENT_STRIPED_myBBname/test1
	srun a.out INSERT_YOUR_CODE_OPTIONS_HERE 
	```

!!!warning
	Be careful when running many jobs that write to the same location - you may inadvertently overwrite your data.

#### Delete the reservation

It is **very important** to remember to delete the persistent reservation once you have finished using it, to avoid inconveniencing other users. At present, we recommend this task be submitted as an independent batch job. Similarly to creating the reservation, the reservation will be destroyed as soon as the scheduler reads the batch job. This means your reservation may be destroyed many hours before your batch job rises to the top of the queue and actually executes. 

!!!example
	```
	#!/bin/bash
	#SBATCH -q debug
	#SBATCH -N 1
	#SBATCH -C haswell
	#SBATCH -t 00:05:00
	#BB destroy_persistent name=myBBname
	```

## Quotas

| Type   | Quota |
|--------|:-----:|
| Space  | 50 TB |
| inodes | none  |

## Performance

The peak bandwidth performance is over 1.7 TB/s. Each Burst Buffer node contributes up to 6.5 GB/s. The number of Burst Buffer nodes depends on the granularity and size of the Burst Buffer allocation. Performance is also dependent on access patter, transfer size and access method (e.g. MPI I/O, shared files).

