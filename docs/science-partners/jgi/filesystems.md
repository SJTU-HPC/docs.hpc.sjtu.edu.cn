# Filesystems

It is the responsibility of every user to organize and maintain
software and data critical to their science. Managing your data
requires understanding NERSC's storage systems.

## Filesystems

NERSC provides several filesystems, and to make efficient use of NERSC
computing facilities, it's critical to have an understanding of the
strengths and limitations of each filesystem.

!!! note
	JAMO is JGI's in-house-built hierarchical file system, which
	has functional ties into NERSC's filesystems and tape
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

Each user's `$HOME` directory has a quota of 40GB and 1M inodes. In
most cases, \$HOME quotas cannot be increased, and users should make
use of other filesystems for storage of large volumes of data and for
computation.

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

projectb "Scratch" and "Sandbox" space is intended for staging and
running JGI calculations on the NERSC systems. On Denovo, the projectb
scratch space is the recommended filesystem for performing file IO
during all your applications. The environment variable \$BSCRATCH points
to the user's projectb scratch space. If you don't,
please [file a Consulting ticket](https://help.nersc.gov).

The Sandbox areas are allocated by program.  If you have questions
about your program's space, please see your group lead. New Sandbox
space must be allocated with JGI management approval.

The projectB software allocations are intended for storage of
Conda environments, source code, and binaries being used by individual
groups at JGI. At very large production scale, projectB performance,
may degrade; consider moving such software to "/usr/common/software"
or the Data and Archive filesystem.

### DnA (Data n' Archive)

DnA is a 1PB GPFS based file system for the JGI's archive, shared
databases and project directories.

| 	|DnA Projects|DnA Shared|DnA DM Archive|
|---|---|---|---|
|Location|/global/dna/projectdirs/|/global/dna/shared|/global/dna/dm_archive|
|Quota|5TB default|Defined by agreement with the JGI Management|Defined by agreement with the JGI Management|
|Backups|Daily, only for projectdirs with quota <= 5TB|Backed up by JAMO|Backed up by JAMO|
|Files are not automatically purged|Files are not automatically purged|Purge policy set by users of the JAMO system|Files are not automatically purged|

The intention of the DnA "Project" and "Shared" space is to be a place
for data that is needed by multiple people collaborating on a project
to allow for easy reading of shared data. The "Project" space is
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
backed up and files are purged if they have not been accessed for 90
days.  Access your scratch directory with the environment variable
"$SCRATCH" for example:

```
cd $SCRATCH
```

Scratch environment variables:

|Environment Variable|Value|NERSC Systems|
|---|---|---|
\$SCRATCH|Best-connected file system|All NERSC computational systems|
\$BSCRATCH|/global/projectb/scratch/$username|Denovo only|
\$CSCRATCH|/global/cscratch[1,2,3]/sd/$username|Cori|

\$BSCRATCH points to your projectb scratch space if you have a
BSCRATCH allocation.  \$SCRATCH will always point to the
best-connected scratch space available for the NERSC machine you are
accessing.  For example, on Denovo \$SCRATCH will point to \$BSCRATCH,
whereas on Cori \$SCRATCH will point to \$CSCRATCH.

The intention of scratch space is for staging, running, and completing
your calculations on NERSC systems.  Thus these filesystems are
designed to allow wide-scale file reading and writing from many
compute nodes.  The scratch filesystems are not intended for long-term
file storage or archival, and thus data is not backed-up, and files
not accessed for a significant time period will be automatically purged.

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
