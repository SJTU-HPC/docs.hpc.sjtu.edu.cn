# GridFTP Data Transfer

GridFTP is a command line service for parallel movement of data. You
may find it easier to use [Globus](globus.md), which uses the same
underlying gridFTP but adds reliability, performance, and ease of use.

## Availability

GridFTP provides a high performance transfer mechanism to move data in
and out of NERSC. GridFTP is available on the following systems:

| Hostname | Description |Recommended Use
| --- | --- | ---
| `dtn0[1-4].nersc.gov` | High performance data transfer nodes with access to all NERSC Global File systems (NGF) as well as Cori Scratch |  Almost all data transfers needs into & out of NERSC
| `garchive.nersc.gov` | Single node system connected directly to the NERSC HPSS tape archive | Remote transfers into & out of HPSS

## Transferring Data with GridFTP

Use `globus-url-copy` to move data with GridFTP:

```shell
Syntax: globus-url-copy [-help | -usage] [-version[s]] [-vb]
[-dbg] [-b | -a] [-q] [-r] [-rst] [-f <filename>] [-s <subject>] [-ds
<subject>] [-ss <subject>] [-tcp-bs <size>] [-bs <size>] [-p
<parallelism>] [-notpt] [-nodcau] [-dcsafe | -dcpriv] <sourceURL>
<destURL> 
```

Initialize your proxy certificate:

```shell
myproxy-logon -s nerscca.nersc.gov
```

Copy a file from your workstation to dtn01:

```shell
globus-url-copy file:///path/to/file \ 
gsiftp://dtn01.nersc.gov//path/file
```

Copy a file from your workstation to HPSS:

```shell
globus-url-copy gsiftp://garchive.nersc.gov/path/file \
file:///path/to/file
```

Copy a file from dtn01 to HPSS ("third party copy" without directly
logging in to either system)

```shell
globus-url-copy gsiftp://dtn01.nersc.gov/path/to/file \
gsiftp://garchive.nersc.gov/path/to/file
```

## Performance Optimization

For optimal data transfer perfomance, you may need to tune certain
parameters for your network. In the example below we are using 4
parallel streams with a TCP block size of 4MB:

```shell
globus-url-copy -p 4 -tcp-bs 4MB file:///path/to/file \
gsiftp://dtn01.nersc.gov//path/file
```

## Firewalls

If you have problems using GridFTP across a firewall (eg. your
transfer hangs without moving any data), you may need to ask your
network administrator to open a range of ports in your firewall. Once
this is done, you will need to set this range in your environment so
that GridFTP clients are aware of this.

For example, to use the port range 60000 to 60064 set the following
environment variable, before starting your client:

```bash
export GLOBUS_TCP_PORT_RANGE=60000,60064
```
