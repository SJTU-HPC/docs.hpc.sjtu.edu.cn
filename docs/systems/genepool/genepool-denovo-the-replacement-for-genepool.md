Denovo is the name of the new configuration of the system available exclusively to JGI users on the NERSC Mendel cluster. The previous configuration is referred to as "Old Genepool".

#Logging into Denovo

Logging into Denovo is as simple as typing `ssh denovo` if logged into another NERSC system. Genepool users are all enabled there by default, and because your $HOME is on a global filesystem, your files are in the same place as they were on Old Genepool and on Cori. However, Denovo does not have all the same software modules installed that Old Genepool did, since we expect you to use tools like Shifter or Anaconda to manage software in a more robust manner, whenever possible. There are tutorials on Shifter and Anaconda on the [Genepool Training & Tutorials page](genepool-training-and-tutorials.md), as well as tutorials on Slurm.

#Using Slurm on Denovo/Genepool

Like all NERSC systems, Denovo exclusively uses the open-source Slurm scheduler for its job scheduling. You can view NERSC's pages on Slurm, and the complete documentation for Slurm here.

|Genepool UGE command|	Slurm equivalent|	Description|
|---|---|---|
|qsub yourscript.sh|sbatch yourscript.sh|Submit the shell script "yourscript.sh" as a job
|qlogin	|salloc| Start an interactive session. On Slurm, salloc creates a node allocation, and automatically logs you into the first of the nodes you were allocated.|
|qs|	squeue|	View jobs running on the cluster. Uses a cached dataset, and can be utilized with scripts. "squeue --help" will provide a full list of options.|
|qhold $jobnumber|scontrol hold $jobnumber|Hold a specific job|
|qrls $jobnumber|scontrol release $jobnumber|Release a specific job|
|qhost|sinfo |Get information on the configuration of the cluster. "sinfo --help" provides a full list of options|
||scontrol show $ENTITY|also provides detailed info about various cluster configuration. For example "scontrol show partitions" or "scontrol show nodes". See "scontrol show --help" for a full list of options.|


#Guide to scheduler options.
These options can be placed in the submission command, or the shell script header. You can find more useful info HERE, and a [cheat sheet](https://slurm.schedmd.com/rosetta.pdf).

|UGE option|Slurm equivalent|Description|
|---|---|---|
|-q $queue|-q $qos|On NERSC systems you shoudl request a QOS, and the scheduler direct your jobs to the appropriate partition based on QOS and resource requests. A QOS request is NOT necessary on Denovo, but is required on Cori (On Cori, JGI users should ask for `genepool` or `genepool_shared`)|
| |-N $count|Number of nodes requested. (In UGE, you would request a total number of cpus, and UGE would allocate an appropriate number of nodes to fill the request, so this option was not available.)|
|-pe $options|-n $count|Number of MPI tasks requested per node. ***Be careful if requesting values for both -N and -n (they are multiplicative!!!)***.|
| |-c $count|Number of cpus per task. The number of cpus per task multiplied by the number of tasks per node should not exceed the number of cpus per node!|
|-l h_rt=$seconds|-t h:min:sec|Hard run time limit. Note that in Slurm, "-t 30" is requesting 30 seconds of run time.|
|-l mem_free=$value|--mem=$value|Minimum amount of memory, with units. For example: --mem=120G|
|-l ram.c=$value|--mem-per-cpu=$value|Minimum amount of memory per cpu with units. For example: --mem-per-cpu=5G|
|-o $filename|-o $filename|Standard output filename|
|-e $filename|-e $filename|Standard error filename|
|-m abe|--mail-type=$events|Send email message on events. In Slurm, $events can be BEGIN, END, FAIL, or ALL|
|-M $emailaddress|--mail-user=$emailaddress|Email event messages to $emailaddress|
|-P $project|--A=$project|Project under which to charge the job.|


#Submitting jobs
Slurm is much more flexible than UGE, so we don't need to impose strict limits with lots of queues that leads to inefficient use of resources. To allow Slurm to schedule your jobs efficiently, and to make your batch jobs portable between Denovo and Cori, you should not target particular queues or partitions if you can avoid it. Instead, you should specify the number of CPUs you need, the amount of memory, and the runtime that you need (walltime, not multiplied by number of CPUs).

The more accurately you specify those parameters, the faster your job will schedule, because the scheduler can find resources to match your requirements sooner than it otherwise might. Requesting overly long runtimes, in particular, will cause your jobs to be held in the period before scheduled maintenance, since the scheduler will assume there's not enough time to run your job before the machine goes down for maintenance. Run a few pilot jobs first if you're not sure what you need. Workflow managers can handle retries with adjusted limits in the case of outliers.

Since there is currently no per-user job-submission limit, it's entirely possible that one person can commandeer the entire cluster for an indefinite period. We ask that you be responsible with your job submissions. If you use task arrays, you can limit the degree of concurrency so that you can submit large numbers of jobs without triggering complaints. E.g, this example shows how to submit 1000 jobs, but only allow 10 to run at any one time.

```bash
denovo> sbatch --array=1-1000%10 ...
```

You can request exclusive use of a node with the `--exclusive` flag to the sbatch command. Slurm enforces job-boundaries more rigidly than UGE, so jobs that attempt to use more memory or CPUs than they have been allocated will be killed by the scheduler. Using `--exclusive` overrides the number of CPUs requested and the amount of memory, of course.

#Memory and your jobs
You should take care to specify the amount of memory your job needs. By default, if you don't request memory, your job will go to one of the 128 GB nodes. To use 256 GB or more, you must specify the memory you need. The actual amount of memory available for jobs is slightly less than the total installed, since the operating system takes some of it. So if you actually request 128 GB of memory, with `--mem=128G`, your job will run on a 256 GB node. The sinfo command in the previous section shows you how to list the available memory, in MB, and the current max limits are shown in the table here.

|Nominal system memory|Maximum available memory (MB)|Slurm options (default unit is MB, use K/M/G/T to specify)|
|---|---|---|
|128 GB|121042|`--mem=121042`, or `--mem=118G`|
|256 GB|249239|`--mem=249239`, or `--mem=243G`|
|512 GB|506097|`--mem=506097`, or `--mem=494G`|
|1 TB| 1012160|`--mem=1012160`, or `--mem=988G`|


#Monitoring job progress
The squeue command is very flexible, and can give you a lot of information on currently running or queued jobs. For completed jobs, sacct offers much the same functionality. There are simple options to select the set of jobs by username, by partition, by job-ID etc, and options to specify the output information in ways that are easy for scripts to parse. For example, to see all my currently running or queued jobs, I can do this:

```bash
denovo> squeue --user wildish
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
             33077 productio sleep.sh  wildish  R      10:50      1 mc1211
```

If  you want to monitor your jobs from a script, you can produce output that is much easier for a script to parse:

```bash
denovo> squeue --noheader --user wildish --Format=jobid,state
33077               RUNNING
```

Or even simpler, if you know the ID of the job you want to monitor:

```bash
denovo> squeue --noheader --job 33077 --Format=state
RUNNING
```

For completed jobs, the sacct command is almost completely equivalent:

```bash
denovo> sacct --noheader --user wildish --format=jobid,state
33065         COMPLETED
33065.0       COMPLETED
33066         COMPLETED
33066.0       COMPLETED
33067            FAILED
33067.0          FAILED
33068            FAILED
33068.0          FAILED
33069         COMPLETED
33069.0       COMPLETED
```

Check the man pages for more details of each command's options (e.g. `man sacct`).

#Running with Shifter
Shifter on Denovo behaves much the same as on Cori or Edison. You do not need to load any modules for Shifter, it's in the default $PATH. Shifter implements a limited subset of the Docker functionality which can be safely supported at NERSC. Shifter is able to run images, and volume mount to access filesystems. Specifying ports or environment variables is not supported, nor are more advanced features like linked containers. In general, you're expected to simply run a script or binary, with optional arguments, with input and output mapped to the global filesystems.

You can find more information on Shifter on the Using Shifter and Docker page, here are a few simple example commands:

##Pulling an image from Dockerhub to NERSC:
```bash
denovo> shifterimg pull bioboxes/velvet
2017-09-27T14:24:59 Pulling Image: docker:bioboxes/velvet, status: READY
```

##Listing the images which have been pulled already:
You can list the images which have been pulled to NERSC already with the `shifterimg images` command. E.g. to find images built by the JGI container project so far and available on Denovo, use this command:

```bash
denovo2> shifterimg images | grep jgi
mendel     docker     READY    b6ba49b95b   2017-05-15T15:46:02 registry.services.nersc.gov/jgi/bwa:latest
mendel     docker     READY    a4c075bcfe   2017-05-11T09:47:48 registry.services.nersc.gov/jgi/checkm:latest
mendel     docker     READY    ea246c6191   2017-07-26T11:24:50 registry.services.nersc.gov/jgi/hmmer:latest
mendel     docker     READY    1cd8629c3e   2017-05-15T16:28:46 registry.services.nersc.gov/jgi/macs2:latest
mendel     docker     READY    1a8159327b   2017-05-16T14:43:13 registry.services.nersc.gov/jgi/picard:latest
mendel     docker     READY    b2cf3f787e   2017-09-11T09:30:52 registry.services.nersc.gov/jgi/prodege:latest
mendel     docker     READY    efddc3487b   2017-09-15T10:38:03 registry.services.nersc.gov/jgi/prodigal:latest
mendel     docker     READY    d3de678de9   2017-08-21T16:49:22 registry.services.nersc.gov/jgi/smrtanalysis:2.3.0_p5
mendel     docker     READY    6a8ee03bf2   2017-09-15T10:42:27 registry.services.nersc.gov/jgi/smrtlink:4.0.0.190159
mendel     docker     READY    36b07a3f2d   2017-05-11T11:10:03 registry.services.nersc.gov/jgi/tony-sandbox:latest
mendel     docker     READY    00b175aab6   2017-05-10T12:32:30 registry.services.nersc.gov/jgi/trinity:latest
mendel     docker     READY    22891d8b91   2017-05-05T12:09:38 registry.services.nersc.gov/jgi/usearch:gitlab
mendel     docker     READY    6c728db34a   2017-05-15T15:58:43 registry.services.nersc.gov/jgi/wgsim:latest
```

##Running an image interactively:
Denovo currently supports running images on the login nodes. Cori and Edison do not. You should run on the login nodes only to debug or test images, not to run something that takes longer than a few minutes. Once you know that your container runs, please submit a batch job to run it, rather than use the login nodes.

```bash
denovo> shifter --image=registry.services.nersc.gov/jgi/hmmer:latest hmmscan -h
# hmmscan :: search sequence(s) against a profile database
# HMMER 3.1b2 (February 2015); https://hmmer.org/
# Copyright (C) 2015 Howard Hughes Medical Institute.
# Freely distributed under the GNU General Public License (GPLv3).
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Usage: hmmscan [-options] <hmmdb> <seqfile>
[...]
```

Note that you have to use the full name of the image, `registry.services.nersc.gov/jgi/hmmer:latest`, not just `hmmer:latest` or `hmmer`, like you can with Docker.

##Running an image interactively on a batch node:
If you need to run for more than a few minutes for debugging, you can get an interactive node and run there. Use the salloc command, then run shifter. Salloc takes many of the same options that sbatch does, in that you can ask for more than one node, specify the time you want the allocation for etc. Salloc immediately logs you into the first node of your allocation.

```bash
denovo> salloc
salloc: Granted job allocation 33066
bash-4.1$ shifter --image=registry.services.nersc.gov/jgi/hmmer:latest hmmscan -h
# hmmscan :: search sequence(s) against a profile database
# HMMER 3.1b2 (February 2015); https://hmmer.org/
# Copyright (C) 2015 Howard Hughes Medical Institute.
# Freely distributed under the GNU General Public License (GPLv3).
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Usage: hmmscan [-options] <hmmdb> <seqfile>
[...]
bash-4.1$ exit
salloc: Relinquishing job allocation 33066
denovo>
```

##Running a shifter image in a batch script:
Shifter is integrated with Slurm on Cori, Edison and Denovo, which means that you can tell Slurm to pull the shifter image you need, and keep the body of your script cleaner. For example, you can submit the following script to Slurm:

```bash
denovo> cat shifter.sh
#!/bin/bash
#SBATCH -t 00:01:00
#SBATCH --image=alpine:3.5

shifter cat /etc/os-release
```

and it will produce this output:

```bash
denovo> cat slurm-33075.out
NAME="Alpine Linux"
ID=alpine
VERSION_ID=3.5.2
PRETTY_NAME="Alpine Linux v3.5"
HOME_URL="https://alpinelinux.org"
BUG_REPORT_URL="https://bugs.alpinelinux.org"
```

So, in this example we specified the image in the #SBATCH directive, and just used `shifter` in the script body to run a command from that container. That's a bit cleaner than having `shifter --image=...` sprinkled throughout the batch script, but we can go one step further. By specifying the image on the `sbatch` submit command, and not in the script, we can make a script that works with several different versions of the container without change. This example simply tells you what OS the container thinks its running, and we can tell the script to run different containers at submit-time:

```bash
denovo> cat shifter.sh
#!/bin/bash
#SBATCH -t 00:01:00

shifter cat /etc/os-release

denovo> sbatch --image=alpine:3.5 shifter.sh
Submitted batch job 33076
[...]

denovo> cat slurm-33076.out
NAME="Alpine Linux"
ID=alpine
VERSION_ID=3.5.2
PRETTY_NAME="Alpine Linux v3.5"
HOME_URL="https://alpinelinux.org"
BUG_REPORT_URL="https://bugs.alpinelinux.org"
```

 now run the same script with a different image:

```bash
wildish@mc1218: default:~> sbatch --image=ubuntu:16.04 shifter.sh
Submitted batch job 89
[...]

denovo> cat slurm-89.out
premount hook
CAP_LAST_CAP: 37, possibleMaxCap: 37
NAME="Ubuntu"
VERSION="16.04.2 LTS (Xenial Xerus)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 16.04.2 LTS"
VERSION_ID="16.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
VERSION_CODENAME=xenial
UBUNTU_CODENAME=xenial
```

##Interactive nodes, 'gpint' replacements
We have a few nodes available as replacements for the group-specific gpint nodes, so-called 'dints'. These are configured exactly like the other login nodes, but are not part of the round-robin login, they can only be logged into by connecting first to a standard Denovo login node and then on to the dint. We can allocate these to groups for testing purposes, and once your group is happy that everything works, we will co-ordinate with you to migrate your gpints to dints. Let us know if you need one.

##Interactive nodes for batch debugging
There are 5 nodes reserved for interactive work, specifically for debugging batch scripts, not for running long-term services or for exploratory analysis.

You can access them by using the `--qos=interactive` flag:
```bash
denovo> salloc --qos=interactive --time 01:00:00
salloc: Granted job allocation 283255
salloc: Waiting for resource configuration
salloc: Nodes mc1535 are ready for job
```

Please use these nodes reasonably, don't stay logged in if you're not making proper use of them.
