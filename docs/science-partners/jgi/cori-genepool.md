# Cori for JGI

A subset of nodes on Cori, the flagship supercomputer at NERSC, are
reserved for exclusive use by JGI users. The Burst Buffer, Shifter,
and all other features available on Haswell Cori nodes are available
by using the JGI-specific "quality of service" (QOS).

## Access

All JGI affiliated individuals have access to the JGI reserved fraction
of Cori compute capacity. This service first became available in
January 2018. 

JGI staff and affiliates are provided special
access to Cori via a number of "quality of service" or QOS arguments
which are passed to slurm job submissions.

* All JGI users must specify the Slurm account under which the job
  will run (with `-A <youraccount>`). Unlike other NERSC users, JGI
  users do not have a default account.
* For jobs requiring one or more whole nodes, use `--qos=genepool`.
* For jobs which can share a node with other jobs, use `--qos=genepool_shared`.
- Each of the following items first require `module load esslurm`:
	* For large memory batch jobs use `--qos=jgi_exvivo`.
	* For large memory shared batch jobs use `--qos=jgi_shared`.
	* For large memory interactive jobs use `--qos=jgi_interactive`.
	* For transfer jobs which write to the Data and Archive filesystem 
        use `--qos=xfer_dna`.

!!! note
	Jobs run under the Cori "genepool", "genepool_shared",
        "jgi_exvivo", "jgi_shared", "jgi_interactive", and "xfer_dna" QOS
	are not charged. Resources are scheduled on a first come
        first served basis; please be a good citizen to your fellow
        researchers. Users violating the spirt of this policy
        will find themselves less able to do so.

!!! note
	The JGI's Cori capacity is entirely housed on standard
	Haswell nodes: 32 physical cores, each core with 2 hyperthreads,
        no local hard drives, and 128GB	memory. It is not necessary to
        request `-c Haswell` via Slurm if using a JGI QOS. KNL nodes are
        NOT available via a JGI QOS. To use KNL nodes, submit to one
        of Cori's standard QOS (such as `regular`), and use the "m342"
        account. Be aware that jobs run with "m342" will charge
        NERSC allocation hours to JGI.

!!! example
	For a single core shared job, you would minimally need:

	`sbatch --qos=genepool_shared -A <youraccount> yourscript.sh`

	To request an interactive session on a single node with all CPUs and memory:

	`salloc --qos=genepool -A <youraccount>`

        Don't forget that if the Cori genepool QOS is full, the previous command
        can take a long time to give you a node.

In the earlier examples, 'youraccount' is the project name you
submit to, not your username or file group name.  If you don't know what
accounts you belong to, you can check with:

`sacctmgr show associations where user=$USER`

## Cori Features and Other Things to Know

Cori offers additional features and capabilities that can
be of use to JGI researchers:

### Slurm

Cori uses the Slurm job scheduler. Documentation and examples
for using Slurm at NERSC can be found [here.](../../jobs/index.md) 

The batch queues on Cori and Denovo are not configured
identically. Cori and Denovo have different capabilities and
maintenance cycles. If you write scripts that need to know which
machine they're running on, you can use the $NERSC_HOST environment
variable to check the current host.

### Cori Scratch
 
Cori scratch is a Lustre filesystem, accessible through Cori and
Cori ExVivo, but not Denovo. This directory can be found at
`/global/cscratch1/sd/$USER` or by using the \$CSCRATCH environment
variable. Like `/global/projectb/scratch` (\$BSCRATCH), Cori scratch is
purged periodically; backing up data stored there is your responsibility.
The [HPSS Tape Data Archive](../../filesystems/archive.md) can be
used for for this purpose, or the JGI JAMO system. See
[the NERSC Data Management Policy](../../data/policy.md) for more
information on topics such as automatic file backups and 
scratch directory purge frequency. 

\$BSCRATCH is also mounted on Cori, Cori genepool, and Cori ExVivo.
This is useful for a workload needing to see files from all machines.

!!! note
	The performance of the different filesystems will vary
	depending significantly on what your application is doing. It's worth
	experimenting with your data in different locations to see what
	gives the best results.

### Burst Buffer

The Burst buffer is a fast filesystem optimized for applications
demanding large amounts of I/O bandwidth and operations. This system 
is particularly suitable for applications that perform lots of
random-access, or that read files more than once.

To use the Burst Buffer add directives to your batch
job to either scheduling staging in/out of data or to make a
persistent reservation. The dynamic reservation lasts only as long
as the job that requested it and the disk space is reclaimed once
the job ends. A persistent reservation outlives the job that created it,
and can be accessed by multiple jobs.

Use dynamic reservations for checkpoint files, for files that will be
accessed randomly (i.e. not read through in a streaming manner) or
just for local scratch space. Cori batch nodes don't have local disk,
so a dynamic reservation can serve that role.

Use persistent reservations to store data that is shared between jobs
and heavily used such as reference databases. The data on a
persistent reservation is stored with normal unix filesystem
permissions, and anyone can mount your persistent reservation in their
batch job, so you can use them to share heavily used data among
workflows belonging to a group.

You can access multiple persistent reservations in a single batch job,
but any batch job can have only one dynamic reservation.

The per-user limit on Burst Buffer space is 50 TB. If the sum of your
persistent and dynamic reservations reaches that total, further jobs
that require Burst Buffer space will not start until some of those
reservations are removed.

!!! example
	This is a simple example showing how to use the Burst
	Buffer. See the links below for full documentation on how to use
	it.

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

More information on getting started with Burst Buffer can be found
[here](../../filesystems/cori-burst-buffer.md). There
are slides from a training session on the Burst Buffer on the
[JGI training page](training.md).

## JGI Partition configuration

|Setting|Value|
|---|---|
|Job limits|5000 exclusive jobs, or 10000 shared jobs|
|Run time limits|72 h|
|Partition size|192 nodes|
|Node configuration|32-core Haswell CPUs (64 hyperthreads), 128GB memory|
