# The HPSS Archive System

## Intro

The High Performance Storage System (HPSS) is a modern, flexible,
performance-oriented mass storage system. It has been used at NERSC
for archival storage since 1998. HPSS is intended for long term
storage of data that is not frequently accessed.

HPSS is Hierarchical Storage Management (HSM) software developed by a
collaboration of DOE labs, of which NERSC is a participant, and
IBM. The HPSS system is a tape system that uses HSM software to ingest
data onto a high performance disk cache and automatically migrate it
to a very large enterprise tape subsystem for long-term retention. The
disk cache in HPSS is designed to retain many days worth of new data
and the tape subsystem is designed to provide the most cost-effective
long-term scalable data storage available.

NERSC's HPSS system can be accessed at `archive.nersc.gov` through a
variety of clients such as hsi, htar, ftp, pftp, and Globus. By
default every user has an HPSS account.

## Accessing HPSS
You can access NERSC's HPSS in a variety of different ways. Hsi and
htar are the best ways to transfer data in and out of HPSS within
NERSC. Globus is recommended for transfers to or from outside
NERSC. We also offer access via gridFTP, pftp, and ftp.

### Automatic Token Generation
The first time you try to connect from a NERSC system (Cori, DTNs,
etc.) using a NERSC provided client like hsi, htar, or pftp you will
be prompted for your NERSC password + one-time password which will
generate a token stored in $HOME/.netrc. After completing this step
you will be able to connect to HPSS without typing a password:

```
nersc$ hsi
Generating .netrc entry...
Password + OTP:
```

If you are having problems connecting see the [Troubleshooting
section](#trouble-connecting) below.


### Session Limits
Users are limited to 15 concurrent sessions. This number can be
temporarily reduced if a user is impacting system usability for
others.

### Hsi
Hsi is a flexible and powerful command-line utility to access the
NERSC HPSS storage system. You can use it to store and retrieve files
and it has a large set of commands for listing your files and
directories, creating directories, changing file permissions, etc. The
command set has a UNIX look and feel (e.g. mv, mkdir, rm, cp, cd,
etc.) so that moving through your HPSS directory tree is close to what
you would find on a UNIX file system. Hsi can be used both
interactively or in batch scripts.

The hsi utility is available on all NERSC production computer systems
and it has been configured on these systems to use high-bandwidth
parallel transfers.

#### Hsi Usage Examples
All of the NERSC computational systems available to users have the hsi
client already installed. To access the Archive storage system you can
type hsi with no arguments. This will put you in an interactive
command shell, placing you in your home directory on the Archive
system. From this shell, you can run the `ls` command to see your
files, `cd` into storage system subdirectories, `put` files
into the storage system and `get` files from it.

In addition to command line, you can run hsi commands several
different ways:

  * Single-line execution: `hsi "mkdir run123;  cd run123; put bigdata.0311"`
  * Read commands from a file: `hsi "in command_file"`
  * Read commands from standard input: `hsi < command_file`
  * Read commands from a pipe: `cat command_file | hsi`

The hsi utility uses a special syntax to specify local and HPSS
file names when using the put and get commands. The local file
name is always on the left and the HPSS file name is always on the
right and a ":" (colon character) is used to separate the names

There are some shortcuts, for instance the command `put
myfile.txt` will store the file named `myfile` from your current
local file system directory into a file of the same name into your
current HPSS directory. If you wanted to put it into a specific
HPSS directory, you can also do something like `hsi "cd run123;
put myfile.txt"`

Most of the standard Linux commands work in hsi (`cd`,
`ls`,`rm`,`chmod`,etc.). There are a few commands that
are unique to hsi

| Command | Function  |
|-------------|-----------|
| get, mget   | Copy one or more HPSS-resident files to local files |
| cget | Conditional get - get the file only if it doesn't already exist on the target |
| put, mput | Copy one or more local files to HPSS |
| cput | Conditional put - copy the file into HPSS unless it is already there |

Hsi also has a series of "local" commands, that act on the
non-HPSS side of things:

| Command | Function |
|-------------|-----------|
| lcd | Change local directory |
| lls | List local directory |
| lmkdir | Make a local directory |
| lpwd | Print current local directory |
| command | Issue shell command |

#### Removing Older Files
You can find and remove older files in HPSS using the `hsi find`
command. This may be useful if you're doing periodic backups of
directories (this is not recommended for software version control,
instead use a versioning system like git) and want to delete older
backups. Since you can't use a linux pipe ("|") in hsi, you need a
multi-step process. The example below will find files older than 10
days and delete them from HPSS.

```
hsi -q "find . -ctime 10" > temp.txt 2>&1
cat temp.txt | awk '{print "rm -R",$0}' > temp1.txt
hsi in temp1.txt
```

#### Removing Entire Directories
To recursively remove a directory and all of its contained
sub-directories and files, use `hsi rm -R <directory_name>`.

### Htar
Htar is a command line utility that is ideal for storing groups of
files in HPSS. Since the tar file is created directly in HPSS, it is
generally faster and uses less local space than creating a local tar
file then storing that into HPSS. Htar also does inline compression so
compressing the data beforehand is unnecessary and it preserves the
directory structure of stored files. Furthermore, htar creates an
index file that (by default) is stored along with the archive in
HPSS. This allows you to list the contents of an archive without
retrieving it from tape first. The index file is only created if the
htar bundle is successfully stored in the archive.

Htar is installed and maintained on all NERSC production systems. If
you need to access the member files of an htar archive from a system
that does not have the htar utility installed, you can retrieve the
tar file to a local file system and extract the member files using the
local tar utility.

If you have a collection of files and store them individually with
hsi, the files will likely be distributed across a collection of
tapes, requiring long delays (due to multiple tape mounts) when
fetching them from HPSS. Instead, grouping these files in an htar
archive file that will likely be stored on a single tape, requiring
only a single tape mount when it comes time to retrieve the data.

The basic syntax of htar is similar to the standard tar utility:

```
htar -{c|K|t|x|X} -f tarfile [directories] [files]
```

As with the standard unix tar utility the `-c` `-x` and `-t` options
create, extract, and list tar archive files. The `-K` option verifies
an existing tarfile in HPSS and the `-X` option can be used to
re-create the index file for an existing archive.

Please note, you cannot add or append files to an existing htar file.

If your htar files are 100 GB or larger and you only want to extract
one or two small member files, you may find faster retrieval rates by
skipping staging the file to the HPSS disk cache by adding the
`-Hnostage` option to your htar command.

#### Htar Usage Examples

Create an archive with directory `nova` and file `simulator`

```
nersc$ htar -cvf nova.tar nova simulator
HTAR: a   nova/
HTAR: a   nova/sn1987a
HTAR: a   nova/sn1993j
HTAR: a   nova/sn2005e
HTAR: a   simulator
HTAR: a   /scratch/scratchdirs/elvis/HTAR_CF_CHK_61406_1285375012
HTAR Create complete for nova.tar. 28,396,544 bytes written for 4 member files, max threads: 4 Transfer time: 0.420 seconds (67.534 MB/s)
HTAR: HTAR SUCCESSFUL
```

Now list the contents:

```
nersc$ htar -tf nova.tar
HTAR: drwx------  elvis/elvis          0 2010-09-24 14:24  nova/
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1987a
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1993j
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn2005e
HTAR: -rwx------  elvis/elvis     398552 2010-09-24 17:35  simulator
HTAR: -rw-------  elvis/elvis        256 2010-09-24 17:36  /scratch/scratchdirs/elvis/HTAR_CF_CHK_61406_1285375012
HTAR: HTAR SUCCESSFUL
```

As an example, using hsi remove the `nova.tar.idx` index file from HPSS
(Note: you generally do not want to do this)
```
nersc$ hsi "rm nova.tar.idx"
rm: /home/e/elvis/nova.tar.idx (2010/09/24 17:36:53 3360 bytes)
```

Now try to list the archive contents without the index file:
```
nersc$ htar -tf nova.tar
ERROR: No such file: nova.tar.idx
ERROR: Fatal error opening index file: nova.tar.idx
HTAR: HTAR FAILED
```

Here is how we can rebuild the index file if it is accidently deleted
```
nersc$ htar -Xvf nova.tar
HTAR: i nova
HTAR: i nova/sn1987a
HTAR: i nova/sn1993j
HTAR: i nova/sn2005e
HTAR: i simulator
HTAR: i /scratch/scratchdirs/elvis/HTAR_CF_CHK_61406_1285375012
HTAR: Build Index complete for nova.tar, 5 files 6 total objects, size=28,396,544 bytes
HTAR: HTAR SUCCESSFUL

nersc$ htar -tf nova.tar
HTAR: drwx------  elvis/elvis          0 2010-09-24 14:24  nova/
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1987a
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1993j
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn2005e
HTAR: -rwx------  elvis/elvis     398552 2010-09-24 17:35  simulator
HTAR: -rw-------  elvis/elvis    256 2010-09-24 17:36  /scratch/scratchdirs/elvis/HTAR_CF_CHK_61406_1285375012
HTAR: HTAR SUCCESSFUL
```

Here is how we extract a single file from a htar file

```
nersc$ htar -xvf nova.tar simulator
```

##### Using ListFiles to Create an htar Archive

Rather than specifying the list of files and directories on the
command line when creating an htar archive, you can place the list of
file and directory pathnames into a ListFile and use the `-L` option.
The contents of the ListFile must contain exactly one pathname per
line.

```
nersc$ find nova -name 'sn19*' -print > novalist
nersc$ cat novalist
nova/sn1987a
nova/sn1993j
```

Now create an archive containing only these files
```
nersc$ htar -cvf nova19.tar -L novalist
HTAR: a   nova/sn1987a
HTAR: a   nova/sn1993j
nersc$ htar -tf nova19.tar
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1987a
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1993j
```

##### Soft Delete and Undelete

The `-D` option can be used to "soft delete" one or more member files
or directories from an htar archive. The files are not really
deleted, but simply marked in the index file as deleted. A file that
is soft-deleted will not be retrieved from the archive during an
extract operation. If you list the contents of the archive, soft
deleted files will have a `D` character after the mode bits in the
listing:

```
nersc$ htar -Df nova.tar nova/sn1993j
HTAR: d  nova/sn1993j
HTAR: HTAR SUCCESSFUL
```

Now list the files and note that sn1993j is marked as deleted:

```
nersc$ htar -tf nova.tar
HTAR: drwx------   elvis/elvis          0 2010-09-24 14:24  nova/
HTAR: -rwx------   elvis/elvis    9331200 2010-09-24 14:24  nova/sn1987a
HTAR: -rwx------ D elvis/elvis    9331200 2010-09-24 14:24  nova/sn1993j
HTAR: -rwx------   elvis/elvis    9331200 2010-09-24 14:24  nova/sn2005e
```
To undelete the file, use the -U option:

```
nersc$ htar -Uf nova.tar nova/sn1993j
HTAR: u  nova/sn1993j
HTAR: HTAR SUCCESSFUL
```

List the file and note that the 'D' is missing
```
nersc$ htar -tf nova.tar nova/sn1993j
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1993j
```

#### Htar Archive Verification

You can request that htar compute and save checksum values for each
member file during archive creation. The checksums are saved in the
corresponding htar index file. You can then further request that htar
compute checksums of the files as you extract them from the archive
and compare the values to what it has stored in the index file.

```
nersc$ htar -Hcrc -cvf nova.tar nova
HTAR: a   nova/
HTAR: a   nova/sn1987a
HTAR: a   nova/sn1993j
HTAR: a   nova/sn2005e
```

Now, in another directory, extract the files and request verification
```
nersc$ htar -Hverify=crc -xvf nova.tar
HTAR: x nova/
HTAR: x nova/sn1987a, 9331200 bytes, 18226 media blocks
HTAR: x nova/sn1993j, 9331200 bytes, 18226 media blocks
```

#### Htar Limitations
Htar has several limitations to be aware of:

* **Member File Path Length:** File path names within an htar
aggregate of the form prefix/name are limited to 154 characters for
the prefix and 99 characters for the file name. Link names cannot
exceed 99 characters.  
* **Member File Size:** The maximum file size the NERSC archive will
support is approximately 20 TB. However, we recommend you aim for htar
aggregate sizes of several hundred GBs. Member files within an htar
aggregate are limited to approximately 68GB.
* **Member File Limit:** Htar aggregates have a default soft limit of
1,000,000 (1 million) member files. Users can increase this limit to a
maximum hard limit of 5,000,000 member files.

### Globus
[Globus](../services/globus.md) is recommended for transfers between
sites (i.e. non-NERSC to NERSC).

To access the HPSS system using Globus, you first need to create a
Globus account. Once you've created an account you can log in either
with your Globus information or with your NERSC account information.

The NERSC HPSS endpoint is called "NERSC HPSS". You can use the web
interface to transfer files. Currently, there is no explicit ordering
by tape of file retrievals for Globus. 

!!! caution
    **If you're retrieving a large data set from HPSS with Globus,
    please see [this
    page](../services/globus.md#transfer-files-from-nerscs-hpss-archive-to-another-location)
    for instructions on how to best order files using hsi and then
    retrieve files using the command line interace for Globus in tape
    order.**

### GridFTP, pftp, and ftp
Files can be transferred between HPSS and remote sites via the
standard internet protocol ftp, however, being non-parallel the
performance of ftp will probably not be as good as other methods such
as Globus. Note that on NERSC systems ftp translates to pftp so it
is in fact parallel. There is no sftp (secure ftp) or scp access.

As standard ftp clients only support authentication via the
transmission of unencrypted passwords, which NERSC does not permit,
special procedures must be used with ftp on remote sites, see HPSS
Passwords.

## Best Practices
HPSS is intended for long term storage of data that is not frequently
accessed.

The best guide for how files should be stored in HPSS is how you might
want to retrieve them. If you are backing up against accidental
directory deletion / failure, then you would want to store your files
in a structure where you use htar to separately bundle up each
directory. On the other hand, if you are archiving data files, you
might want to bundle things up according to month the data was taken
or detector run characteristics, etc. The optimal size for htar
bundles is 100 - 500 GBs, so you may need to do several htar bundles
for each set depending on the size of the data.

### Group Small Files Together
HPSS is optimized for file sizes of 100 - 500 GB. If you need to store
many files smaller than this, please use htar to bundle them together
before archiving. HPSS is a tape system and responds differently than
a typical file system. If you upload large numbers of small files they
will be spread across dozens or hundreds of tapes, requiring multiple
loads into tape drives and positioning the tape. Storing many small
files in HPSS without bundling them together will result in extremely
long retrieval times for these files and will slow down the HPSS
system for all users.

### Order Large Retrievals
If you are retrieving many (> 100 files) from HPSS, you need to
order your retrievals so that all files on a single tape will be
retieved in a single pass in the order they are on the tape. NERSC has
a script to help you generate an ordered list for retrieval called
`hpss_file_sorter.script`.

??? tip "Generating a sorted list for retrieval"
    To use the script, you first need a list of fully qualified
	file path names and/or directory path names. If you do not
	already have such a list, you can query HPSS using the
	following command:

	```
	hsi -q 'ls -1 <HPSS_files_or_directories_you_want_to_retrieve>' 2> temp.txt
	```

	(for csh replace "2>" with ">&"). Once you have the list of files, feed it to the sorting script:

	```
	hpss_file_sorter.script temp.txt > retrieval_list.txt
	```

    The best way to retrieve this list from HPSS is with the `cget`
    command, which will get the file from HPSS only if it isn't
    already in the output directory. You also should take advantage of
    the `hsi in <file_of_hsi_commands.txt>` to run an entire set of
    HPSS commands in one HPSS session. This will avoid HPSS doing a
    sign in procedure for each file, which can add up to a significant
    amount of time if you are retrieving many files. To do this,
    you'll need to add a little something to the retrieval_list.txt
    file you already generated:

    ```
	awk '{print "cget",$1}' retrieval_list.txt > final_retrieval_list.txt
	```

    Finally, you can retrieve the files from HPSS with

    ```
	hsi "in final_retrieval_list.txt"
	```

    This procedure will return all the files you're retrieving in a
    single directory. You may want to preserve some of the directory
    structure you have in HPSS. If so, you could automatically
    recreate HPSS subdirectories in your target directory with this
    command

    ```
	sed 's:^'<your_hpss_directory>'/\(.*\):\1:' temp.txt | xargs
    -I {} dirname {} | sort | uniq | xargs -I {} mkdir -p {}
	```

    where <your_hpss_directory> is the root directory you want to
    harvest subdirectories from, and temp.txt holds the output from
    your `ls -1` call.

### Avoid Very Large Files
Files sizes greater than 1 TB can be difficult for HPSS to work with
and lead to longer transfer times, increasing the possibility of
transfer interruptions. Generally it's best to aim for file sizes in
the 100 - 500 GB range. You can use `tar` and `split` to break up
large aggregates or large files into 500 GB sized chunks:

```
nersc$ tar cvf - myfiles* | split -d --bytes=500G - my_output_tarname.tar.
```

This will generate a number of files with names like
`my_output_tarname.tar.00`, `my_output_tarname.tar.01`, which you can
use "hsi put" to archive into HPSS. When you retrieve these files, you
can recombine them with cat

```
nersc$ cat my_output_tarname.tar.* | tar xvf -
```

### Accessing HPSS Data Remotely
We recommend a two-stage process to move data to / from HPSS and a
remote site. Use Globus to transfer the data between NERSC and the
remote site (your scratch directory would make a useful temporary
staging point) and use hsi or htar to move the data into HPSS.

When connecting with HPSS via ftp or pftp, it is not uncommon to
encounter problems due to firewalls at the client site. Often you will
have to configure your client firewall to allow connections to
HPSS and generate a token for accessing HPSS remotely.

#### Manual Token Generation
You can generate a string for access to NERSC HPSS from outside the
NERSC network by logging to NIM and selecting "Generate an HPSS token"
from the "Actions" menu. Ignore the password provided and select
"Please use this link to specify a different IP address". Then enter
the IP address of the system from which you wish to connect to
HPSS. Note that this prefills the box with the IP address that the
browser is running on and this may not be the system you intend to
access HPSS from. Enter the correct IP address and select "Generate
Token".

#### Firewalls and External Access
Most firewalls are configured to deny incoming network connections
unless access is explicitly granted. Systems running htar or hsi that
want to connect to the archive at NERSC must accept network
connections which are initiated by the HPSS Movers (helper machines
that initiate multi-stream data movement into and out of the
archive). By default hsi is configured with Firewall Mode set to on
and will usually work without any firewall changes. To configure your
system to allow connections from HPSS Movers at NERSC, you will need
to grant access for TCP connections originating from the
`128.55.32.0/22`, `128.55.80.0/21`, `128.55.88.0/24`, `128.55.136.0/22`, and
`128.55.207.0/24` subnets.


### Use the Xfer Queue
User the dedicated [xfer queue](../../jobs/examples/#xfer-queue) for
long-running transfers to / from HPSS. You can also submit jobs to the
xfer queue after your computations are done. The xfer queue is
configured to limit the number of running jobs per user to the same
number as the limit of HPSS sessions.

## HPSS Usage Charging
DOE's Office of Science awards an HPSS quota to each NERSC project
every year. Users charge their HPSS space usage to the HPSS repos of
which they are members.

Users can check their HPSS usage and quotas with the hpssquota command
on Cori. You view usages on a user level:

```
nersc$ hpssquota -u usgtest
HPSS Usage for User usgtest
REPO                          STORED [GB]      REPO QUOTA [GB]     PERCENT USED [%]
-----------------------------------------------------------------------------------
nstaff                             144.25              49500.0                  0.3
matcomp                              10.0                950.0                  1.1
-----------------------------------------------------------------------------------
TOTAL USAGE [GB]                   154.25
```

Here, "Stored" shows you how much data you have stored in HPSS. Data
stored in HPSS could potentially be charged to any repo that you are a
member of (see below for details). The "Repo Quota" shows you the
maximum amount your PI has allocated for you to store data, and the
"Percent Used" shows the percentage of the quota you've used.

You can also view usage on a repo level:

```
nersc$ hpssquota -r ntrain
HPSS Usage for Repo ntrain

USER                          STORED [GB]           USER QUOTA [GB]          PERCENT USED [%]
---------------------------------------------------------------------------------------------
train1                             100.00                     500.0                      20.0
train2                               0.35                      50.0                       0.1
train47                              0.12                     500.0                       0.0
train28                              0.09                     500.0                       0.0
---------------------------------------------------------------------------------------------

TOTAL USAGE [GB]         TOTAL QUOTA [GB]              PERCENT USED
100.56                              500.0                     20.11
```

"Stored" shows how much data each user has in HPSS that is charged to
this repo. "User Quota" shows how much total space the PI has
allocated for that user (by default this is 100%, PIs may want to
adjust these for each user, see below for more info) and the "Percent
Used" is the percentage of allocated quota each user has used. The
totals at the bottom shows the total space and quota stored for the
whole repo.

You can also check the HPSS quota for a repo by logging in to the NIM
and clicking on their "Account Usage" tab.

### Apportioning User Charges to Repositories: Project Percents
If a user belongs to only one HPSS repo all usage is charged to that
repo. If a user belongs to multiple repos daily charges are
apportioned among the repos using the project percents for that login
name. Default project percents are assigned based on the size of each
repo's storage allocation. Users (only the user, not the project
managers) can change their project percents by selecting Change SRU
Proj Pct (this is a historic name based on the old charging model)
from the Actions pull-down list in the NIM main menu. Users should try
to set project percents to reflect their actual use of HPSS for each
of the projects of which they are a member. Note that this is quite
different from the way that computational resources are charged.

On each computational system each job is charged to a specific
repository. This is possible because the batch system has accounting
hooks that handle charging to repos. The HPSS system has no
notion of repo accounting but only of user accounting. Users must
say "after the fact" how to distribute their HPSS usage charges to the
HPSS repos to which they belong. For a given repo the MPP
repository and the HPSS repository usually have the same name.

### Adding or Removing Users
If a user is added to a new repo or removed from an existing repo the
project percents for that user are adjusted based on the size of the
quotas of the repos to which the user currently belongs. However, if
the user has previously changed the default project percents the
relative ratios of the previously set project percents are respected.

As an example user u1 belongs to repos r1 and r2 and has changed the
project percents from the default of 50% for each repo to 40% for r1
and 60% for r2:

| Login	| Repo	| Allocation (GBs) | Project % |
|-------|-------|------------------|-----------|
| u1 	| r1 	| 500 	   | 40 |
| u1 	| r2 	| 500 	   | 60 |

If u1 then becomes a new member of repo r3 which has a storage
allocation of 1,000 GBs the project percents will be adjusted as
follows (to preserve the old ratio of 40:60 between r1 and r2 while
adding r3 which has the same SRU allocation as r1+r2):

| Login  | Repo	| Allocation (GBs) |	    Project % |
|--------|------|------------------|------------------|
| u1     | r1 	| 500 	   | 20 |
| u1     | r2 	| 500 	   | 30 |
| u1     | r3 	| 1,000 	  | 50 |

If a repo is retired, the percentage charged to that repo is spread
among the remaining repos while keeping their relative values the
same.


## HPSS Project Directories
A special "project directory" can be created in HPSS for groups of
researchers who wish to easily share files. The file in this directory
will be readable by all members of a particular unix file group. This
file group can have the same name as the repository (in which case all
members of the repository will have access to the project directory)
or a new name can be requested (in which case only those users added
to the new file group by the requester will have access to the project
directory).

HPSS project directories have the following properties:

  * located under /home/projects
  * owned by the PI, a PI Proxy, or a Project Manager of the associated repository
  * have suitable group attribute (include "setgid bit")

To request creation of an HPSS project directory the PI, a PI Proxy or
a Project Manager of the requesting repository should fill out the
[HPSS Project Directory Request
Form](https://www.nersc.gov/users/storage-and-file-systems/hpss/request-form).

## Troubleshooting
Some frequently encountered issues and how to solve them.

### Trouble connecting
The first time you try to connect using a NERSC provided client like
hsi, htar, or PFTP you will be prompted for your NERSC password +
one-time password which will generate a token stored in
$HOME/.netrc. This allows you to connect to HPSS without typing a
password. However, sometimes this file can become out of date or
otherwise corrupted. This generates errors that look like this:

```
nersc$ hsi
result = -11000, errno = 29
Unable to authenticate user with HPSS.
result = -11000, errno = 9
Unable to setup communication to HPSS...
*** HSI: error opening logging
Error - authentication/initialization failed
```

If this error occurs try moving $HOME/.netrc file to
$HOME/.netrc_temp. Then connect to the HPSS system again and enter
your NERSC password + one-time password when prompted. A new
$HOME/.netrc file will be generated with a new entry/token. If the
problem persists contact account support.

### Cannot transfer files using htar
Htar requires the node you're on to accept incoming connections from
its movers. This is not possible from a compute node at NERSC, so htar
transfers will fail. Instead we recommend you use our special [xfer
queue](../../jobs/examples/#xfer-queue) for data transfers

### Globus transfer errors
Globus transfers will fail if you don't have permission to read the
source directory or space to write in the target directory. One common
mistake is to make the files readable, but forget to make the
directory holding them readable. You can check directory permissions
with `ls -ld`. At NERSC you can make sure you have enough space to
write in a directory by using the `myquota` command.