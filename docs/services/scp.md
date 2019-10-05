# Using scp at NERSC

Secure Copy (scp) is used to securely transfer files between two hosts
using the Secure Shell (ssh) protocol.

!!! tip
    Scp is suggested for smaller files (<~10GB), otherwise use
    [Globus](../services/globus.md).

To transfer files into/out of NERSC using scp, you need an ssh
client. On Linux/Unix and MacOS these should be installed by default,
but on Windows you will need a GUI tool such as WinSCP.

## Usage
Get a file from Data Transfer Node

```shell
scp user_name@dtn01.nersc.gov:/remote/path/myfile.txt /local/path
```

Send a file to Data Transfer Node

```shell
scp /local/path/myfile.txt user_name@dtn01.nersc.gov:/remote/path
```

Use a pre-existing ssh key (like one made by sshproxy)

```shell
scp -i ~/.ssh/nersc user_name@dtn01.nersc.gov:/remote/path/myfile.txt /local/path
```

### Using tar+ssh

When you want to transfer many small files in a directory, we
recommend [Globus](../services/globus.md). If you don't wish to use
Globus, you can consider using ssh piped with tar.

Send a directory to Data Transfer Node:

```shell
tar cz /loca/path/dirname | ssh user_name@dtn01.nersc.gov tar zxv -C /remote/path
```

Get a directory from Data Transfer Node:

```shell
ssh user_name@dtn01.nersc.gov tar cz /remote/path/dirname | tar zxv -C /local/path
```