# Using bbcp at NERSC

Bbcp is a point-to-point network file copy application with excellent
network transfer rates. The application was originally written for
transferring large files of the data-intensive High-Energy Physics
community. Bbcp is available on the NERSC Data Transfer Nodes.


## Requirements

To transfer files into/out of NERSC using bbcp, you need an SSH
client. For most Unix-like systems (Linux//MacOS/Cygwin), the command
ssh is sufficient and the bbcp executable. You can download the bbcp
executable from the [maintainer's
site](http://www.slac.stanford.edu/~abh/bbcp/).

## Usage

All example commands below are executed on your local machine, not the
NERSC machine:

The syntax of bbcp is similar to the syntax of scp, with some special
options specifying how to run ssh/bbcp.

Get a file from Data Transfer Node:

```shell
bbcp -S "ssh -x -a -oFallBackToRsh=no %I -l %U %H /usr/common/usg/bin/bbcp" "user_name@dtn01.nersc.gov:/remote/path/file" /local/path
```

Send a file to Data Transfer Node:

```shell
bbcp -T "ssh -x -a -oFallBackToRsh=no %I -l %U %H /usr/common/usg/bin/bbcp" /local/path/file "user_name@dtn01.nersc.gov:/remote/path/"
```

Get a file from an outside host to NERSC:

```shell
bbcp -S "ssh -x -a -oFallBackToRsh=no %I -l %U %H /path/to/bbcp/on/remote/host" "user_name@remote.host.com:/remote/path/file" /local/path
```
Send a file from NERSC to an outside host:

```shell
bbcp -T "ssh -x -a -oFallBackToRsh=no %I -l %U %H /path/to/bbcp/on/remote/host" /local/path/file "user_name@remote.host.com:/remote/path/"
```

Note the difference between "-S" and "-T" option, "-S" means the source (where the data come from), "-T" means the target (where the data goes to).

In case you get the following error or similar, add the "-z" option to your command line (right after bbcp).

```shell
bbcp: Accept timed out on port 5031
bbcp: Unable to allocate more than 0 of 8 data streams.
Killed by signal 15.
```
