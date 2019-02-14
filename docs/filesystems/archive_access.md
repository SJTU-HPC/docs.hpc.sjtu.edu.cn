# Accessing HPSS
You can access NERSC's HPSS in a variety of different ways. HSI and
HTAR are the best ways to transfer data in and out of HPSS within
NERSC. Globus is recommended for transfers to or from outside
NERSC. We also offer access via gridFTP, pftp, and ftp.

HSI, HTAR, pftp, and some ftp clients will look for a file name
".netrc" in your home directory. This will enable automated
authentication of access to HPSS. You will not be prompted for a
username/password pair. A sample file showing entries is provided
below:

```
machine archive.nersc.gov
login elvis
password 02V02zwoA3kI5sZ2VysafaFyZABi8K7Tz+iJj4jJ99EdyMjFMZcUyw==
```

Users are limited to 15 concurrent sessions. This number can be
temporarily reduced if a user is impacting system usability for others.



## HSI
HSI is a flexible and powerful command-line utility to access
the NERSC HPSS storage systems. You can use it to store and
retrieve files and it has a large set of commands for listing
your files and directories, creating directories, changing file
permissions, etc. The command set has a UNIX look and feel (e.g. mv,
mkdir, rm, cp, cd, etc.) so that moving through your HPSS directory
tree is close to what you would find on a UNIX file
system. HSI can be used both interactively or in batch scripts.

The HSI utility is available on all NERSC production computer systems
and it has been configured on these systems to use high-bandwidth
parallel transfers.

### HSI Usage Examples
All of the NERSC computational
systems available to users have the hsi client already
installed. To access the Archive storage system you can type hsi
with no arguments. This will put you in an interactive command
shell, placing you in your home directory on the Archive system.
From this shell, you can run the ```ls``` command to see your
files, ```cd``` into storage system subdirectories, ```put```
files into the storage system and get files from it.

In addition to command line, you can run hsi commands several different ways:

  * Single-line execution: ```hsi "mkdir run123;  cd run123; put bigdata.0311```
  * Read commands from a file: ```hsi "in command_file"```
  * Read commands from standard input: ```hsi < command_file```
  * Read commands from a pipe: ```cat command_file | hsi```

The hsi utility uses a special syntax to specify local and HPSS
file names when using the put and get commands. The local file
name is always on the left and the HPSS file name is always on the
right and a ":" (colon character) is used to separate the names

There are some shortcuts, for instance the command ```put
myfile.txt``` will store the file named "myfile" from your current
local file system directory into a file of the same name into your
current HPSS directory. If you wanted to put it into a specific
HPSS directory, you can also do something like ```hsi "cd run123;
put myfile.txt"```

Most of the standard Linux commands work in hsi (```cd```,
```ls```,```rm```,```chmod```,etc.). There are a few commands that
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

You can find and remove older files in HPSS using the ```hsi find```
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
sub-directories and files, use ```rm -R <directory_name>```.


## HTAR

HTAR is a command line utility that creates and manipulates
HPSS-resident tar-format archive files. It is ideal for storing groups
of files in HPSS. Since the tar file is created directly in HPSS, it
is generally faster and uses less local space than creating a local
tar file then storing that into HPSS. Furthermore, HTAR creates an
index file that (by default) is stored along with the archive in
HPSS. This allows you to list the contents of an archive without
retrieving it from tape first. The index file is only created
if the HTAR bundle is successfully stored in the archive.

HTAR is installed and maintained on all NERSC production systems. If
you need to access the member files of an HTAR archive from a system
that does not have the HTAR utility installed, you can retrieve the
tar file to a local file system and extract the member files using the
local tar utility.

HTAR is useful for storing groups of related files that you will
probably want to access as a group in the future. Examples include:

  * archiving a source code directory tree
  * archiving output files from a code simulation run
  * archiving files generated by an experimental run

If stored individually, the files will likely be distributed across a
collection of tapes, requiring long delays (due to multiple tape
mounts) when fetching them from HPSS. On the other hand, an HTAR
archive file will likely be stored on a single tape, requiring only a
single tape mount when it comes time to retrieve the data.

The basic syntax of HTAR is similar to the standard tar utility:

``` htar -{c|K|t|x|X} -f tarfile [directories] [files]```

As with the standard unix tar utility the "-c" "-x" and "-t" options
respectively function to create, extract, and list tar archive
files. The "-K" option verifies an existing tarfile in HPSS and the
"-X" option can be used to re-create the index file for an existing
archive.

Please note, you cannot add or append files to an existing archive.

When HTAR creates an archive, it places an additional file (with an
idx postfix) at the end of the archive. This is an index file that
HTAR can use to more quickly retrieve individual files from your
bundle. It is only created if the htar completed successfully.

If your htar files are 100 GBs or larger and you only want to extract
one or two small member files, you may find faster retrieval rates by
skipping staging the file to the HPSS disk cache by adding the
"-Hnostage" option to your htar command.

### HTAR Usage Examples

Create an archive with directory "nova" and file "simulator"

```
htar -cvf nova.tar nova simulator
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
htar -tf nova.tar
HTAR: drwx------  elvis/elvis          0 2010-09-24 14:24  nova/
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1987a
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1993j
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn2005e
HTAR: -rwx------  elvis/elvis     398552 2010-09-24 17:35  simulator
HTAR: -rw-------  elvis/elvis        256 2010-09-24 17:36  /scratch/scratchdirs/elvis/HTAR_CF_CHK_61406_1285375012
HTAR: HTAR SUCCESSFUL
```

As an example, using hsi remove the nova.tar.idx index file from HPSS (Note: you generally do not want to do this)
```
hsi "rm nova.tar.idx"
rm: /home/j/elvis/nova.tar.idx (2010/09/24 17:36:53 3360 bytes)
```

Now try to list the archive contents without the index file:
```
htar -tf nova.tar
ERROR: No such file: nova.tar.idx
ERROR: Fatal error opening index file: nova.tar.idx
HTAR: HTAR FAILED
```

Here is how we can rebuild the index file if it is accidently deleted
```
htar -Xvf nova.tar
HTAR: i nova
HTAR: i nova/sn1987a
HTAR: i nova/sn1993j
HTAR: i nova/sn2005e
HTAR: i simulator
HTAR: i /scratch/scratchdirs/elvis/HTAR_CF_CHK_61406_1285375012
HTAR: Build Index complete for nova.tar, 5 files 6 total objects, size=28,396,544 bytes
HTAR: HTAR SUCCESSFUL

htar -tf nova.tar
HTAR: drwx------  elvis/elvis          0 2010-09-24 14:24  nova/
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1987a
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1993j
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn2005e
HTAR: -rwx------  elvis/elvis     398552 2010-09-24 17:35  simulator
HTAR: -rw-------  elvis/elvis    256 2010-09-24 17:36  /scratch/scratchdirs/elvis/HTAR_CF_CHK_61406_1285375012
HTAR: HTAR SUCCESSFUL
```

Here is how we extract a single file from a htar file

```htar -xvf nova.tar simulator```

#### Using ListFiles to Create an HTAR Archive

Rather than specifying the list of files and directories on the
command line when creating an HTAR archive, you can place the list of
file and directory pathnames into a ListFile and use the "-L" option.
The contents of the ListFile must contain exactly one pathname per
line.

```
find nova -name 'sn19*' -print > novalist

cat novalist
nova/sn1987a
nova/sn1993j
```

Now create an archive containing only these files
```
htar -cvf nova19.tar -L novalist
HTAR: a   nova/sn1987a
HTAR: a   nova/sn1993j

htar -tf nova19.tar
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1987a
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1993j
```

#### Soft Delete and Undelete

The "-D" option can be used to "soft delete" one or more member files
or directories from an HTAR archive. The files are not really
deleted, but simply marked in the index file as deleted. A file that
is soft-deleted will not be retrieved from the archive during an
extract operation. If you list the contents of the archive, soft
deleted files will have a 'D' character after the mode bits in the
listing:

```
htar -Df nova.tar nova/sn1993j
HTAR: d  nova/sn1993j
HTAR: HTAR SUCCESSFUL
```

Now list the files and note that sn1993j is marked as deleted:

```
htar -tf nova.tar
HTAR: drwx------   elvis/elvis          0 2010-09-24 14:24  nova/
HTAR: -rwx------   elvis/elvis    9331200 2010-09-24 14:24  nova/sn1987a
HTAR: -rwx------ D elvis/elvis    9331200 2010-09-24 14:24  nova/sn1993j
HTAR: -rwx------   elvis/elvis    9331200 2010-09-24 14:24  nova/sn2005e
```
To undelete the file, use the -U option:

```
htar -Uf nova.tar nova/sn1993j
HTAR: u  nova/sn1993j
HTAR: HTAR SUCCESSFUL
```

List the file and note that the 'D' is missing
```
htar -tf nova.tar nova/sn1993j
HTAR: -rwx------  elvis/elvis    9331200 2010-09-24 14:24  nova/sn1993j
```

### HTAR Archive Verification

You can request that HTAR compute and save checksum values for each
member file during archive creation. The checksums are saved in the
corresponding HTAR index file. You can then further request that HTAR
compute checksums of the files as you extract them from the archive
and compare the values to what it has stored in the index file.

```
htar -Hcrc -cvf nova.tar nova
HTAR: a   nova/
HTAR: a   nova/sn1987a
HTAR: a   nova/sn1993j
HTAR: a   nova/sn2005e
```

Now, in another directory, extract the files and request verification
```
htar -Hverify=crc -xvf nova.tar
HTAR: x nova/
HTAR: x nova/sn1987a, 9331200 bytes, 18226 media blocks
HTAR: x nova/sn1993j, 9331200 bytes, 18226 media blocks
```

### HTAR Limitations

HTAR has several limitations to be aware of:

#### Member File Path Length

File path names within an HTAR aggregate of the form prefix/name are
limited to 154 characters for the prefix and 99 characters for the
file name. Link names cannot exceed 99 characters.

#### Member File Size

The maximum file size the NERSC archive will support is approximately
20 TB. However, we recommend you aim for HTAR aggregate sizes of
several hundred GBs. Member files within an HTAR aggregate are limited
to approximately 68GB.

#### Member File Limit

HTAR aggregates have a default soft limit of 1,000,000 (1 million)
member files. Users can increase this limit to a maximum hard limit of
5,000,000 member file.

## Globus

Globus is recommended for transfers between sites (i.e. non-NERSC to NERSC).

To access the HPSS system using [Globus](https://www.globus.org/), you
first need to create a Globus account. Once you've created an
account you can log in either with your Globus information or
with your NERSC account information. The first time you log in using
your NERSC account you'll be asked to enter your Globus account
information as well.

The NERSC HPSS endpoint is called "NERSC HPSS". You can use the GUI to
transfer files. Currently, there is no explicit ordering by tape of
file retrievals for Globus. If you're retrieving a large data
set with Globus, we recommend that users see [this page](archive.md#order-large-retrievals) for
instructions on how to best order files using HSI and then retrieve
files using the command line interace for Globus in tape
order.

## GridFTP, pftp, and ftp

Files can be transferred between HPSS and remote sites via the
standard internet protocol ftp, however, being non-parallel the
performance of ftp will probably not be as good as other methods such
as gridFTP. Note that on NERSC systems ftp translates to pftp so it
is in fact parallel. There is no sftp (secure ftp) or scp access.

As standard ftp clients only support authentication via the
transmission of unencrypted passwords, which NERSC does not permit,
special procedures must be used with ftp on remote sites, see HPSS
Passwords.

## HPSS Passwords

The HPSS systems use NIM and the NERSC LDAP server to create an "hpss
token" for user authentication. The HPSS token does not expire and
users may generate new tokens as often as they wish and old tokens
will still be honored. If a user wishes to disable previously
generated tokens for security reasons contact the NERSC help desk.

Because HPSS passwords do not expire it is only necessary to generate
a password one time for continued use of HPSS. This password is placed
in a file name ".netrc" for use by hsi, htar, pftp, and most ftp clients.

### Automatic Token Generation for use at NERSC

The first time you try to connect from a NERSC system (Cori, Edison,
etc.) using a NERSC provided client like HSI, HTAR, or pftp you will
be prompted for your NIM password + one-time password which will
generate a token stored in $HOME/.netrc. After completing this step
you will be able to connect to HPSS without typing a password:

```
hsi
Generating .netrc entry...
elvis@auth2.nersc.gov's password:
```

If you have an existing $HOME/.netrc file and you are having problems
connecting to either HPSS system try moving this file to temp.netrc
and re-connect to HPSS. If the problem persists contact NERSC account
support.

You can log into NIM to manually generate an HPSS token by selecting
"Generate an HPSS token" from the "Actions" menu. This will provide
you with a token (an encrypted string) in the pale yellow highlighted
box that may be used on any machine in the NERSC network by any
supported HPSS client (hsi, htar, pftp, or ftp). Below the pale yellow
highlighted box you are also provided with a sample .netrc file with
your updated password. Creating a .netrc and place it in your home
directory to enable pftp, hsi, htar and some ftp clients to read it
upon starting a new session to HPSS and avoid the need to enter your
username/password. Permission on your .netrc file should be set to 600
(chmod 600 ~/.netrc).

You can generate a string for access to NERSC HPSS from outside the
NERSC network by logging to NIM and selecting "Generate an HPSS token"
from the "Actions" menu. Ignore the password provided and select
"Please use this link to specify a different IP address". Then enter
the IP address of the system from which you wish to connect to
HPSS. Note that this prefills the box with the IP address that the
browser is running on and this may not be the system you intend to
access HPSS from. Enter the correct IP address and select "Generate
Token".

## Firewalls and External Access

Most firewalls are configured to deny incoming network connections
unless access is explicitly granted. Systems running HTAR or HSI that
want to connect to the archive at NERSC must accept network
connections which are initiated by the HPSS Movers (helper machines
that initiate multi-stream data movement into and out of the
archive). By default HSI is configured with Firewall Mode set to on
and will usually work without any firewall changes. To configure your
system to allow connections from HPSS Movers at NERSC, you will need
to grant access for TCP connections originating from the
128.55.32.0/22, 128.55.80.0/21, 128.55.88.0/24, 128.55.136.0/22, and
128.55.207.0/24 subnets.
