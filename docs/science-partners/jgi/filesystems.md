# Filesystems

It is the responsibility of every user to organize and maintain
software and data critical to their science. Managing your data
requires understanding NERSC's storage systems.

## Filesystems

NERSC provides several filesystems. Making efficient use of NERSC
computing facilities requires an understanding of the
strengths and limitations of each filesystem.

!!! note
	JAMO is JGI's in-house-built hierarchical file system, which
	has functional ties to NERSC's filesystems and tape
	archive. JAMO is not maintained by NERSC.

All NERSC filesystems have per-user quotas on total storage and
number of files and directories (known as `inodes`). To
check your filesystem usage use the `myquota` command from any login
node, check the [Data Dashboard](https://my.nersc.gov/data-mgt.php)
on [my.nersc.gov](https://my.nersc.gov) or log into
the [NERSC Information Management (NIM) website](https://nim.nersc.gov)
and click on the "Account Usage" tab.

### $HOME

Your home directory is mounted across all NERSC systems.  You should
refer to this home directory as `$HOME` wherever possible, and you
should not change the environment variable `$HOME`.

Each user's `$HOME` directory has a quota of 40GB and 1M inodes. \$HOME
quotas will not be increased. Other filesystems should be used for
storage of large volumes of data which are either being stored long term
or are a part of computation in progress.

!!! warning
	As a global filesystem, \$HOME is shared by all users, and
	is *not* configured for performance. Do not write scripts or run
	software in a way that will cause high bandwidth I/O to
	\$HOME. High volumes of reads and writes can cause congestion on
	the filesystem metadata servers, and can slow NERSC systems
	significantly for all users. Large I/O operations should *always*
	be directed to scratch filesystems.

### Projectb

projectb is a 2.7PB GPFS based file system dedicated to the JGI's
active projects.  There are three distinct user spaces in the projectb
filesystem: "projectb/sandbox", "project/software", and "projectb/scratch".
The projectb filesystem is available on most NERSC systems.

| 	|projectb Scratch|projectb Sandbox|projectb Software|
|---|---|---|---|
|Location|/global/projectb/scratch/\$username|/global/projectb/sandbox/$program|/global/projectb/software/$group|
|Quota|20TB, 5M inodes by default; 40TB upon request|Defined by agreement with JGI Management|500GB, 500K inodes|
|Backups|Not backed up|Not backed up|Backed up|
|File Purging|Files not accessed for 90 days are automatically deleted|no purge|no purge|

The projectb "Scratch" space is intended for staging and
running JGI computation by individual users of NERSC systems.
The environment variable \$BSCRATCH points to a user's projectb scratch space.
These scratch directories are not automatically granted to new users; to request
space please [file a Consulting ticket](https://help.nersc.gov) asking for 
an initial projectb scratch allocation.

Sandbox directories on projectb are allocated by program. Data and software
stored in projectb sandbox is not subject to purging.  If you have questions
about your program's space, please see your group lead. New Sandbox
space or quota increase requests must be approved by JGI management.

The projectb software allocations are intended for storage of
Conda environments, source code, and binaries being used by individual
groups at JGI. At very large production scale, projectb performance,
may degrade; consider moving such software to "/usr/common/software", 
the Data and Archive filesystem, or a Shifter container.

### DnA (Data n' Archive)

DnA is a 2.4PB GPFS filesystem for the JGI's archive, shared
databases, and project directories.

| 	|DnA Projects|DnA Shared|DnA DM Archive|
|---|---|---|---|
|Location|/global/dna/projectdirs/|/global/dna/shared|/global/dna/dm_archive|
|Quota|5TB default|Defined by agreement with the JGI Management|Defined by agreement with the JGI Management|
|Backups|Daily, only for projectdirs with quota <= 5TB|Backed up by JAMO|Backed up by JAMO|
|Files are not automatically purged|Files are not automatically purged|Purge policy set by users of the JAMO system|Files are not automatically purged|

The intention of the DnA "Project" and "Shared" space is to be a place
for data that is needed by multiple users collaborating on a project
which allows for easy reading of shared data. The "Project" space is
owned and managed by the JGI.  The "Shared" space is a collaborative
effort between the JGI and NERSC. Write access to DnA is restricted
to protect high performance; data can only be written to DnA from
[Data Transfer Nodes](../../systems/dtn/index.md)
or by using the `--qos=dna_xfer` QOS.

If you would like a project directory, please use the
[Project Directory Request Form](https://www.nersc.gov/users/storage-and-file-systems/file-systems/project-directory-request-form/).

The "DM Archive" is a data repository maintained by the JAMO system.
Files are stored here during migration using the JAMO system.  The
files can remain in this space for as long as the user specifies.  Any
file that is in the "DM Archive" has also been placed in the HPSS tape
archive.  This section of the file system is owned by the JGI data
management team.

### $SCRATCH

Each user has a "scratch" directory.  Scratch directories are NOT
backed up and files can be purged if they have not been accessed for 90
days. Find your scratch directory using the environment variable
"$SCRATCH" for example:

```
elvis@cori02:~> cd $SCRATCH
elvis@cori02:/global/cscratch1/sd/elvis> 
```

Scratch environment variables:

|Environment Variable|Value|NERSC Systems|
|---|---|---|
\$SCRATCH|Best-connected file system|All NERSC computational systems|
\$CSCRATCH|/global/cscratch[1,2,3]/sd/$username|Cori|

\$BSCRATCH points to your projectb scratch space if you have a
BSCRATCH allocation.  \$SCRATCH will always point to the
best-connected scratch space available for the NERSC machine you are
accessing.  

The intention of scratch space is for staging, running, and completing
calculations on NERSC systems.  Thus these filesystems are
configured for best performance when usage is wide-scale file reading
and writing from many compute nodes.  The scratch filesystems are not
intended for long-term file storage or archival. Data is not backed-up,
and files not accessed for a significant time period can be purged.

Policies for \$SCRATCH are described at [NERSC Data Management Policy](../../data/policy.md#scratch-file-systems).

### Other file systems
Other file systems used by JGI may also be mounted on NERSC systems:

* SeqFS - file system used exclusively by the Illumina sequencers, SDM
  and Instrumentation groups at the JGI.
* /usr/common - is a file system where NERSC staff build software for
  user applications.  This is the principal site for the modular
  software installations.
* /global/project - is a GPFS based file system that is accessible on
  almost all of NERSC's other compute systems used by all the other
  NERSC users.  The projectdir portion of projectb should be favored
  by JGI users instead of /global/project.
