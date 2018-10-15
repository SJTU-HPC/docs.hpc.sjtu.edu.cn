#Data Management

It is the responsibility of every user to organize and maintain software and data critical to their science. Managing your data requires understanding NERSC's storage systems.

##Filesystems
NERSC provides several filesystems, and to make efficient use of NERSC computing facilities, it's critical to have an understanding of the strengths and limitations of each filesystem.

!!! note
	JAMO is JGI's in-house-built hierarchical file system, which has functional ties into NERSC's filesystems and tape archive. JAMO is not maintained by NERSC. To learn more about JAMO, click here.

All NERSC filesystems have per-user size limitations, or quotas. To check your filesystem usage use the `myquota` command from any login node, check the [Data Dashboard](https://my.nersc.gov/data-mgt.php) on [my.nersc.gov](https://my.nersc.gov) or log into the [NERSC Information Management (NIM) website](https://nim.nersc.gov) and click on the "Account Usage" tab.

###\$HOME
Your home directory is mounted across all NERSC systems.  You should refer to this home directory as \$HOME wherever possible, and you should not change the environment variable \$HOME.

Each user's \$HOME directory has a quota of 40GB and 1M inodes. In most cases, \$HOME quotas cannot be increased, and users should make use of other filesystems for storage of large volumes of data and for computation.

!!! warning
	As a global filesystem, \$HOME is shared by all users, and is *not* configured for performance. Do not write scripts or run software in a way that will cause high bandwidth I/O to \$HOME. High volumes of reads and writes can cause congestion on the filesystem metadata servers, and can slow NERSC systems significantly for all users. Large I/O operations should *always* be directed to scratch filesystems.

###Projectb
projectb is a 2.7PB GPFS based file system dedicated to the JGI's active projects.  There are two distinct user spaces in the projectb filesystem: projectb/sandbox and projectb/scratch.  The projectb filesystem is available on most NERSC systems.

| 	|projectb Scratch|projectb Sandbox|
|---|---|---|
|Location|/global/projectb/scratch/\$username|/global/projectb/sandbox/$program|
|Quota|20TB, 5M inodes by default; 40TB upon request|Defined by agreement with JGI Management|
|Backups|Not backed up|Not backed up|
|File Purging|Files not accessed for 90 days are automatically deleted|Files are not automatically purged|

projectb "Scratch" and "Sandbox" space is intended for staging and running JGI calculations on the NERSC systems, including Denovo, Cori, and Edison.  On Denovo, the projectb scratch space is the recommended filesystem for performing file IO during all your calculations, and the environment variable \$BSCRATCH points to the user's projectb scratch space.  If you have access to the Genepool resources, you should have space on projectb scratch. If you don't, please [file a Consulting ticket](https://help.nersc.gov).

The Sandbox areas were allocated by program.  If you have questions about your program's space, please see your group lead. New Sandbox space must be allocated with JGI management approval.

###DnA (Data n' Archive)
DnA is a 1PB GPFS based file system for the JGI's archive, shared databases and project directories.

| 	|DnA Projects|DnA Shared|DnA DM Archive|
|---|---|---|---|
|Location|/global/dna/projectdirs/|/global/dna/shared|/global/dna/dm_archive|
|Quota|5TB default|Defined by agreement with the JGI Management|Defined by agreement with the JGI Management|
|Backups|Daily, only for projectdirs with quota <= 5TB|Backed up by JAMO|Backed up by JAMO|
|Files are not automatically purged|Files are not automatically purged|Purge policy set by users of the JAMO system|Files are not automatically purged|

The intention of the DnA "Project" and "Shared" space is to be a place for data that is needed by multiple people collaborating on a project to allow for easy access for data sharing. The "Project" space is owned and managed by the JGI.  The "Shared" space is a collaborative effort between the JGI and NERSC.

If you would like a project directory, please use the [Project Directory Request Form](https://www.nersc.gov/users/storage-and-file-systems/file-systems/project-directory-request-form/).

The "DM Archive" is a data repository maintained by the JAMO system.  Files are stored here during migration using the JAMO system.  The files can remain in this space for as long as the user specifies.  Any file that is in the "DM Archive" has also been placed in the HPSS tape archive.  This section of the file system is owned by the JGI data management team.

###$SCRATCH
Each user has a "scratch" directory.  Scratch directories are NOT backed up and file are purged if they have not been accessed for 90 days.  Access your scratch directory with the environment variable "$SCRATCH" for example:

cd $SCRATCH
Scratch environment variables:

|Environment Variable|Value|NERSC Systems|
|---|---|---|
\$SCRATCH|Best-connected file system|All NERSC computational systems|
\$BSCRATCH|/global/projectb/scratch/$username|Denovo only|
\$CSCRATCH|/global/cscratch[1,2,3]/sd/$username|Cori and Edison|

\$BSCRATCH points to your projectb scratch space if you have a BSCRATCH allocation.  \$SCRATCH will always point to the best-connected scratch space available for the NERSC machine you are accessing.  For example, on Denovo \$SCRATCH will point to \$BSCRATCH, whereas on Cori \$SCRATCH will point to \$CSCRATCH.

The intention of scratch space is for staging, running, and completing your calculations on NERSC systems.  Thus these filesystems are designed to allow wide-scale file reading and writing from many compute nodes.  The scratch filesystems are not intended for long-term file storage or archival, and thus data is not backed-up, and files not accessed for 90 days will be automatically purged.

###Other file systems
Other file systems used by JGI may also be mounted on NERSC systems:

* SeqFS - file system used exclusively by the Illumina sequencers, SDM and Instrumentation groups at the JGI.
* /usr/common - is a file system where NERSC staff build software for user applications.  This is the principal site for the modular software installations.
* /global/project - is a GPFS based file system that is accessible on almost all of NERSC's other compute systems used by all the other NERSC users.  The projectdir portion of projectb should be favored by JGI users instead of /global/project.



#Archiving your data with HPSS
HPSS is the NERSC tape archival system available to all NERSC users. You can read more about HPSS and find its documentation here.

Here, we provide some basic examples of data transfer and access with HPSS.

###Using HSI from a NERSC Production System
All of the NERSC computational systems available to users have the hsi client already installed.  To access the Archive storage system you can type hsi with no arguments:
`% hsi`
That is, the utility is set up to connect to the Archive system by default.  This is equivalent to typing:
`% hsi -h archive.nersc.gov`
Just typing hsi will enter an interactive command shell, placing you in your home directory on the Archive system.  From this shell, you can run the ls command to see your files, cd into storage system subdirectories, put files into the storage system and get files from it.

!!! note
	You can run hsi commands in several different ways:

	* From a command line:	`% hsi`
	* Single-line execution:	`% hsi "mkdir run123;  cd run123; put bigdata.0311`
	* Read commands from a file:	`% hsi "in command_file"`
	* Read commands from standard input:	`% hsi < command_file`
	* Read commands from a pipe:	`% cat command_file | hsi`

Specifying local and HPSS file names when storing or retrieving files
The HSI `put` command stores files from your local file system into HPSS and the `get` command retrieves them.  The command:
`% put myfile`
will store the file named "myfile" from you current local file system directory into a file of the same name into your current HPSS directory.  So, in order to store "myfile" into the "run123" subdirectory of your home in HPSS, you can type:

``` bash
% hsi
A:/home/j/joeuser-> cd run123
A:/home/j/joeuser-> put myfile
```
or
``` bash
% hsi "cd run123; put myfile"
```
The hsi utility uses a special syntax to specify local and HPSS file names when using the put and get commands: The local file name is always on the left and the HPSS file name is always on the right.
Use a ":" (colon character) to separate the names

``` bash
% put local_file : hpss_file
% get local_file : hpss_file
```
This format is convenient if you want to store a file named "foo" in the local directory as "foo_2010_09_21" in HPSS:

`% hsi "put foo : foo_2010_09_21"`

You can also use this method to specify the full or relative pathnames of files in both the local and HPSS file systems:

`% hsi "get bigcalc/hopper/run123/datafile.0211 : /scratch2/scratchdirs/joeuser/analysis/data"`

###Archiving your data with HTAR
HTAR is a command line utility that creates and manipulates HPSS-resident tar-format archive files.  It is ideal for storing groups of files in HPSS.  Since the tar file is created directly in HPSS, it is generally faster and uses less local space than creating a local tar file then storing that into HPSS.  However, there is a file size limit of 64GB for an individual file within the archive (archives themselves can be much larger).  So if you have individual files that are larger than 64GB that you need to back up, use hsi for those files.

HTAR is useful for storing groups of related files that you will probably want to access as a group in the future.  Examples include:

* archiving a source code directory tree
* archiving output files from a code simulation run
* archiving files generated by the run of an experiment

!!! note
	If stored individually, the files will likely be distributed across a collection of tapes, requiring possibly long delays (due to multiple tape mounts) when fetching them from HPSS.  On the other hand, an HTAR archive file will likely be stored on a single tape, requiring only a single tape mount when it comes time to retrieve the data.

The basic syntax of HTAR is similar to the standard tar utility:

`htar -{c|K|t|x|X} -f tarfile [directories] [files] `

As with the standard unix tar utility the "-c" "-x" and "-t" options respectively function to create, extract, and list tar archive files. The "-K" option verifies an existing tarfile in HPSS and the "-X" option can be used to re-create the index file for an existing archive.  Please note, you cannot add or append files to an existing archive.

!!! note
	When HTAR creates an archive, it places an additional file (with a strange name) at the end of the archive.  Just ignore the file, it is for HTAR internal use and will not be retrieved when you extract the files from the archive.

!!! example
	``` bash
	# Create an archive with directory "nova" and file "simulator"
	% htar -cvf nova.tar nova simulator
	HTAR: a   nova/
	HTAR: a   nova/sn1987a
	HTAR: a   nova/sn1993j
	HTAR: a   nova/sn2005e
	HTAR: a   simulator
	HTAR: a   /scratch/scratchdirs/joeuser/HTAR_CF_CHK_61406_1285375012
	HTAR Create complete for nova.tar. 28,396,544 bytes written for 4 member files, max threads: 4 Transfer time: 0.420 seconds (67.534 MB/s)
	HTAR: HTAR SUCCESSFUL

	# Now List the contents
	% htar -tf nova.tar
	HTAR: drwx------  joeuser/joeuser          0 2010-09-24 14:24  nova/
	HTAR: -rwx------  joeuser/joeuser    9331200 2010-09-24 14:24  nova/sn1987a
	HTAR: -rwx------  joeuser/joeuser    9331200 2010-09-24 14:24  nova/sn1993j
	HTAR: -rwx------  joeuser/joeuser    9331200 2010-09-24 14:24  nova/sn2005e
	HTAR: -rwx------  joeuser/joeuser     398552 2010-09-24 17:35  simulator
	HTAR: -rw-------  joeuser/joeuser        256 2010-09-24 17:36  /scratch/scratchdirs/joeuser/HTAR_CF_CHK_61406_1285375012
	HTAR: HTAR SUCCESSFUL

	# now, as an example, using hsi remove the nova.tar.idx index file from HPSS
	# (Note: you generally do not want to do this)
	% hsi "rm nova.tar.idx"
	...
	rm: /home/j/joeuser/nova.tar.idx (2010/09/24 17:36:53 3360 bytes)

	# Now try to list the archive contents without the index file:
	% htar -tf nova.tar
	ERROR: No such file: nova.tar.idx
	ERROR: Fatal error opening index file: nova.tar.idx
	HTAR: HTAR FAILED

	# Here is how we can rebuild the index file if it is accidently deleted
	% htar -Xvf nova.tar
	HTAR: i nova
	HTAR: i nova/sn1987a
	HTAR: i nova/sn1993j
	HTAR: i nova/sn2005e
	HTAR: i simulator
	HTAR: i /scratch/scratchdirs/joeuser/HTAR_CF_CHK_61406_1285375012
	HTAR: Build Index complete for nova.tar, 5 files 6 total objects, size=28,396,544 bytes
	HTAR: HTAR SUCCESSFUL

	#
	% htar -tf nova.tar
	HTAR: drwx------  joeuser/joeuser          0 2010-09-24 14:24  nova/
	HTAR: -rwx------  joeuser/joeuser    9331200 2010-09-24 14:24  nova/sn1987a
	HTAR: -rwx------  joeuser/joeuser    9331200 2010-09-24 14:24  nova/sn1993j
	HTAR: -rwx------  joeuser/joeuser    9331200 2010-09-24 14:24  nova/sn2005e
	HTAR: -rwx------  joeuser/joeuser     398552 2010-09-24 17:35  simulator
	HTAR: -rw-------  joeuser/joeuser        256 2010-09-24 17:36  /scratch/scratchdirs/joeuser/HTAR_CF_CHK_61406_1285375012
	HTAR: HTAR SUCCESSFUL
	```

For more examples, please go to the HPSS page.
