# The HPSS Archive System

## Intro

The High Performance Storage System (HPSS) is a modern, flexible,
performance-oriented mass storage system. It has been used at NERSC
for archival storage since 1998. HPSS is intended for long term
storage of data that is not frequently accessed.

HPSS is Hierarchical Storage Management (HSM) software developed by a
collaboration of DOE labs, of which NERSC is a participant, and
IBM. The HPSS system is a tape system that uses HSM software to ingest data onto high
performance disk arrays and automatically migrate it to a very large
enterprise tape subsystem for long-term retention. The disk cache in
HPSS is designed to retain many days worth of new data and the tape
subsystem is designed to provide the most cost-effective long-term
scalable data storage available.

NERSC's HPSS system can be accessed at archive.nersc.gov through a
variety of clients such as hsi, htar, ftp, pftp, and globus. By
default every user has an HPSS account.

## Getting Started

### Accessing HPSS

You can access HPSS from any NERSC system. Inside of NERSC, files can
be archived to HPSS individually with the "hsi" command or in groups
with the "htar" command (similar to the way "tar" works). HPSS is also
accessible via Globus, gridFTP, ftp, and pftp. Please see the
[Accessing HPSS](archive_access.md) page for a list of all possible
way to access HPSS and details on their use.

HPSS uses NIM to create an "hpss token" for user authentication. On a
NERSC system, typing "hsi" or "htar" will usually be enough to create
this token. If you are access HPSS remotely (using ftp, pftp, or
gridFTP), you may need to manually generate a token. Please see the
[Accessing HPSS](archive_access.md) page for more details.

### Best Practices

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

#### Group Small Files Together

HPSS is optimized for file sizes of 100 - 500 GB. If you need to store
many files smaller than this, please use htar to bundle them together
before archiving. HPSS is a tape system and responds differently than
a typical file system. If you upload large numbers of small files they
will be spread across dozens or hundreds of tapes, requiring multiple
loads into tape drives and positioning the tape. Storing many small
files in HPSS without bundling them together will result in extremely
long retrieval times for these files and will slow down the HPSS
system for all users.

Please see the [Accessing HPSS](archive_access.md) for more details on
how to use htar.

#### Order Large Retrievals

If you are retrieving many (> 100 files) from HPSS, you need to
order your retrievals so that all files on a single tape will be
retieved in a single pass in the order they are on the tape. NERSC has
a script to help you generate an ordered list for retrieval called
```hpss_file_sorter.script```.

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

    The best way to retrieve this list from HPSS is with the ```cget```
    command, which will get the file from HPSS only if it isn't
    already in the output directory. You also should take advantage of
    the ```hsi in <file_of_hsi_commands.txt>``` to run an entire set of
    HPSS commands in one HPSS session. This will avoid HPSS doing a
    sign in procedure for each file, which can add up to a significant
    amount of time if you are retrieving many files. To do this,
    you'll need to add a little something to the retrieval_list.txt
    file you already generated:

    ```awk '{print "cget",$1}' retrieval_list.txt > final_retrieval_list.txt```

    Finally, you can retrieve the files from HPSS with

    ```hsi "in final_retrieval_list.txt"```

    This procedure will return all the files you're retrieving in a
    single directory. You may want to preserve some of the directory
    structure you have in HPSS. If so, you could automatically
    recreate HPSS subdirectories in your target directory with this
    command

    ```sed 's:^'<your_hpss_directory>'/\(.*\):\1:' temp.txt | xargs
    -I {} dirname {} | sort | uniq | xargs -I {} mkdir -p {} ```

    where <your_hpss_directory> is the root directory you want to
    harvest subdirectories from, and temp.txt holds the output from
    your "ls -1" call.

#### Avoid Very Large Files

Files sizes greater than 1 TB can be difficult for HPSS to work with
and lead to longer transfer times, increasing the possibility
of transfer interruptions. Generally it's best to aim for file
sizes in the 100 - 500 GB range. You can use "tar" and "split"
to break up large aggregates or large files into 500 GB sized
chunks:

```tar cvf - myfiles* | split -d --bytes=500G -
my_output_tarname.tar.```

This will generate a number of files with names like
"my_output_tarname.tar.00", "my_output_tarname.tar.01", which
you can use "hsi put" to archive into HPSS. When you retrieve
these files, you can recombine them with cat

```cat my_output_tarname.tar.* | tar xvf -```

#### Accessing HPSS Data Remotely

We recommend a two-stage process to move data to / from HPSS and a
remote site. Use globus to transfer the data between NERSC and the
remote site (your scratch directory would make a useful temporary
staging point) and use hsi or htar to move the data into HPSS.

When connecting with HPSS via ftp or pftp, it is not uncommon to
encounter problems due to firewalls at the client site. Often you will
have to configure your client firewall to allow connections to
HPSS. See the HPSS firewall page for more details.

#### Use the Xfer Queue

User the dedicated [xfer queue](../../jobs/examples/#xfer-queue) for
long-running transfers to / from HPSS. You can also submit jobs to the
xfer queue after your computations are done. The xfer queue is
configured to limit the number of running jobs per user to the same
number as the limit of HPSS sessions.

### Session Limits

Each HPSS user is limited to no more than 15 concurrent HPSS sessions.

### HPSS Usage Charging

DOE's Office of Science awards an HPSS quota to each NERSC project
every year. Users charge their HPSS space usage to the HPSS repos of
which they are members.

Users can check their HPSS usage and quotas with the hpssquota command
on Cori or Edison. You view usages on a user level:

```
cori03> hpssquota -u usgtest
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
edison03> hpssquota -r ntrain
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

#### Apportioning User Charges to Repositories: Project Percents

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

#### Adding or Removing Users

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


### HPSS Project Directories

A special "project directory" can be created in HPSS for groups of
researchers who wish to easily share files. The file in this directory
will be readable by all members of a particular unix file group. This
file group can have the same name as the repository (in which case all
members of the repository will have access to the project directory)
or a new name can be requested (in which case only those users added
to the new file group by the requester will have access to the project
directory.

HPSS project directories have the following properties:

  * located under /home/projects
  * owned by the PI, a PI Proxy, or a Project Manager of the associated repository
  * have suitable group attribute (include "setgid bit")

To request creation of an HPSS project directory the PI, a PI Proxy or
a Project Manager of the requesting repository should fill out the
[HPSS Project Directory Request
Form](https://www.nersc.gov/users/storage-and-file-systems/hpss/request-form).


### Troubleshooting

Some frequently encountered issues and how to solve them.

#### Trouble connecting

The first time you try to connect using a NERSC provided client like
HSI, HTAR, or PFTP you will be prompted for your NIM password +
one-time password which will generate a token stored in
$HOME/.netrc. This allows you to connect to HPSS without typing a
password. However, sometimes this file can become out of date or
otherwise corrupted. This generates errors that look like this:

```
% hsi
result = -11000, errno = 29
Unable to authenticate user with HPSS.
result = -11000, errno = 9
Unable to setup communication to HPSS...
*** HSI: error opening logging
Error - authentication/initialization failed
```

If this error occurs try moving $HOME/.netrc file to
$HOME/.netrc_temp. Then connect to the HPSS system again and enter
your NIM password + one-time password when prompted. A new
$HOME/.netrc file will be generated with a new entry/token. If the
problem persists contact account support.

#### Cannot transfer files using htar

Htar requires the node you're on to accept incoming connections from
its movers. This is not possible from a compute node at NERSC, so htar
transfers will fail. Instead we recommend you use our special xfer
queue for data transfers

#### Globus transfer errors

Globus transfers will fail if you don't have permission to read the
source directory or space to write in the target directory. One common
mistake is to make the files readable, but forget to make the
directory holding them readable. You can check directory permissions
with ```ls -ld```. At NERSC you can make sure you have enough space to
write in a directory by using the ```myquota``` command.
