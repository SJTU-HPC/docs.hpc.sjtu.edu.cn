NERSC is pleased to provide compute capacity on its flagship supercomputer, Cori, to JGI users. Burst buffer, Shifter, and all other features available to Haswell Cori nodes are available via new JGI-specific "quality of service" (QOS).

#How to access JGI Cori Compute Capacity
Access to Cori compute capacity is now available to all JGI users. The JGI capacity is now considered to be in production use as of January 2018. As this change was only recently made, we are requesting that users give us feedback on their experience to help us through this learning period.

JGI staff and Genepool-authorized researchers are provided special access to Cori via two different "quality of service" or QOS.

* For jobs requiring one or more whole nodes, use `--qos=genepool`.
* For shared jobs ("slotted" jobs in UGE nomenclature), use `--qos=genepool_shared`.
* You will also need to specify the SLURM account under which the job will run (with `-A <youraccount>`).
* Do not request other standard Cori QOS settings (ie debug, premium, etc).

!!! note
	Jobs run under the Cori "genepool" and "genepool_shared" QOS are not charged. Fairshare rules will apply to prevent a single user or project from monopolizing the resource.

!!! note
	The JGI's Cori capacity is entirely housed on standard Haswell nodes: 64 hyperthreaded cores (32 physical) and 128GB memory. It is not necessary to request Haswell nodes via Slurm (for example, with `-C haswell`). KNL nodes are NOT available via the "genepool" or "genepool_shared" qos. To use KNL nodes, you must submit to one of Cori's standard queues, and use the "m342" account - though be aware that normal Cori job charging will apply.

!!! example
	For a single-slotted job, you would minimally need:

	`sbatch --qos=genepool_shared -A <youraccount> yourscript.sh`

	To request an interactive session on a single node with all CPUs (and thus memory):

	`salloc --qos=genepool -A <youraccount> -N 1 -n 64`

Note that 'youraccount' in the above example is the project name you submit to, not your login account. So fungalp, gentechp etc. Unlike during early access, you should use the same project account that you used on Genepool. If you don't know what accounts you belong to, you can check with:

`sacctmgr show associations where user=$USER`


#Modules and Bioinformatics Software

Unlike Genepool, very few NERSC software modules are available for user software. We encourage users to use alternatives wherever possible, which will be more maintainable and portable to other platforms. If you have Genepool module software that is critical to migrating your workflow to Cori, please talk to a consultant.

##Alternatives to using software modules are:

* Shifter, with the software installed in a Docker container
* Anaconda virtual environments allow you to install software in a clean and reproducible manner. See this tutorial for more information about using Anaconda at NERSC. Note that while Anaconda started life as a python package manager, now R, Perl and many other computing languages and associated modules can also be installed using it.
* You are, as always, welcome to install and maintain any software you like on NERSC systems within your own user disk space. While scratch space is subject to NERSC's purge policy, many groups hold a "sandbox" space which is equally performant to $BSCRATCH and shared among group members. (Running software from $HOME is strongly discouraged.)

#Cori Features and Other Things to Know
Cori offers many additional useful features and capabilities that will be of use to JGI researchers:

##SLURM
Unlike Genepool, Cori uses the SLURM job scheduler, which is incompatible with UGE. We've prepared a separate page to get you started on converting submission scripts and commands to SLURM here, and the NERSC webpages on SLURM for Cori are here. Complete SLURM documentation is here, and you may also find this cheatsheet useful.

The batch queues on Cori and Denovo are not configured identically. Cori and Denovo have different capabilities and maintenance cycles. If you need to write scripts that know which machine they're running on, you can use the $NERSC_HOST environment variable to check where you are.

##Cori scratch
Cori scratch is a Lustre filesystem, accessible through Cori and Edison, but not on Genepool or Denovo. The \$CSCRATCH environment variable will point to your Cori scratch directory. Like /projectb/scratch (\$BSCRATCH), Cori scratch is purged periodically, so take care to back up your files. You can find information on how to do that on the HPSS Data Archive page.

\$BSCRATCH is also mounted on Cori and Edison, so you can use that if you need to see your files on all machines.

!!! note
	The performance of the different filesystems may vary, depending partly on what your application is doing. It's worth experimenting with your data in different locations to see what gives the best results.

##The Burst Buffer
The Burst buffer is a fast filesystem optimized for applications demanding high I/O. The Burst Buffer is particularly suitable for applications that perform lots of random-access I/O, or that read files more than once.

To access the Burst Buffer you need to add directives to your batch job to make a reservation. A reservation can be dynamic or persistent. A dynamic reservation lasts only as long as the job that requested it, the disk space is reclaimed once the job ends. A persistent reservation outlives the job that created it, and can be shared among many jobs.

Use dynamic reservations for checkpoint files, for files that will be accessed randomly (i.e. not read through in a streaming manner) or just for local scratch space. Cori batch nodes don't have local disk, unlike the Genepool batch nodes, so a dynamic reservation serves that role well.

Use persistent reservations to store data that is shared between jobs and heavily used, e.g. reference DBs or similar data. The data on a persistent reservation is stored with normal unix filesystem permissions, and anyone can mount your persistent reservation in their batch job, so you can use them to share heavily used data among workflows belonging to a group, not just for your own private work.

You can access multiple persistent reservations in a single batch job, but any batch job can have only one dynamic reservation.

The per-user limit on Burst Buffer space is 50 TB. If the sum of your persistent and dynamic reservations reaches that total, further jobs that require Burst Buffer space will not start until some of those reservations are removed.

!!! example
	This is a simple example showing how to use the Burst Buffer. See the links below for full documentation on how to use it.

	```bash
	#!/bin/bash
	#SBATCH --time=00:10:00
	#SBATCH -N 1
	#SBATCH --constraint haswell
	#DW jobdw capacity=240GB access_mode=striped type=scratch

	echo "My BB reservation is at $DW_JOB_STRIPED"
	cd $DW_JOB_STRIPED
	df -h .
	```

	The output from a particular run of this script is below:

	```bash
	My BB reservation is at /var/opt/cray/dws/mounts/batch/6501112_striped_scratch/
	Filesystem                                    Size  Used Avail Use% Mounted on
	/var/opt/cray/dws/mounts/registrations/24301  242G   99M  242G   1% /var/opt/cray/dws/mounts/batch/6501112_striped_scratch
	```

More information on getting started with Burst Buffer is here. There are slides from a training session on the Burst Buffer on the Genepool training page.

##Shifter

Shifter is Docker-like software for running containers on NERSC systems. More information on Shifter is available here, and Shifter training slides are also available. The main advantages of using Shifter are:

* Using containers makes your workflow portable, across Cori, Denovo, Edison, and to cloud resources
* You no longer need to depend on system features, such as specific compiler versions, software libraries or other tools
* Because Shifter uses Docker containers, you can build and debug containers on your laptop or desktop, then be sure they will run the same way on Cori or other NERSC platforms.
* Shifter exists because Docker cannot be safely run on NERSC machines. Docker requires too much access to the system, anyone who can run a container can essentially access the entire machine. Shifter implements a subset of the same functionality that Docker provides, and can run Docker containers unmodified.

The process for building a container and running it with Shifter is roughly as follows:

1. use Docker on a laptop or desktop machine to build a Docker container for your software
2. push that container to Dockerhub or another Docker container registry
use Shifter on Cori, Edison or Denovo to pull that image to the NERSC Shifter registry
3. use Shifter on a batch node to run that container, and perform useful work.
Note that Shifter is not available on Genepool, the kernel version there is too old to support containers. This is just one reason why we're commissioning Denovo.

!!! note
	The JGI has a containerization project, intended to provide standardized containers for the majority of JGI use-cases. If you need a container for a tool or pipeline, check with your consultants - you may find it's already been done for you. Documentation for the containerization project will be made available soon.

#Current Settings for Cori's JGI Partition
|Setting|Value|
|---|---|
|Job limits|5000 exclusive jobs, or 10000 shared jobs|
|Run time limits|72 h|
|Partition size|192 nodes|
|Node configuration|32-core Haswell CPUs (64 hyperthreads), 128GB memory|
