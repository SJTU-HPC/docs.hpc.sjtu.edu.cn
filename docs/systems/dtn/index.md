# Data Transfer Nodes

## Overview

The data transfer nodes are NERSC servers dedicated to
performing transfers between NERSC data storage resources such as HPSS
and the NERSC Global Filesystem (NGF), and storage resources at other
sites. These nodes are being managed (and monitored for performance)
as part of a collaborative effort between ESnet and NERSC to enable
high performance data movement over the high-bandwidth 100Gb ESnet
wide-area network (WAN).  

## Configuration

All DTNs have :

* Two 100-gigabit ethernet links for transfers to internet
* Two 10-gigabit ethernet links to transfer to NERSC internal resources (HPSS)
* Two FDR IB connections to the global filesystem
* One FDR IB connection to Cori scratch /global/cscratch1

Similar to other NERSC systems, shell configuration files ("dot
files") are under the control of NERSC; users should only modify
".ext" files.

There are four data transfer nodes available for interactive use
(dtn0[1-4].nersc.gov), and a pool of nodes used for automated transfer
services like Globus.

## Access

All NERSC users are automatically given access to the data transfer
nodes. The nodes support both interactive use via SSH (direct login)
or data transfer using GridFTP services; however, we recommend using
[Globus](https://www.globus.org) for large data transfers.

## Available File Systems

The NERSC data transfer nodes provide access to global home, global common, the
project and projecta directories, and Cori scratch system.

!!! tip
        Please note that /tmp is very small. Although certain common
        tools (e.g., vi) use /tmp for temporary storage, users should
        never explicitly use /tmp for data.


## File Transfer Software

For most cases, we recommend [Globus](https://www.globus.org) because it provides the best
transfer rates.  It makes transfers trivial so users do not have to
learn command line options for manual performance tuning. Globus also
does automatic performance tuning and has been shown to perform
comparable to -- or even better (in some cases) than -- expert-tuned
GridFTP. Users can also use GridFTP to transfer large files by hand.

For smaller files you can use Secure Copy (`scp`) or Secure FTP
(`sftp`) or `rsync` to transfer files between two hosts.

## Restrictions

In order to keep the data transfer nodes performing optimally for data
transfers, we request that users restrict interactive use of these
systems to tasks that are related to preparing data for transfer or
are directly related to data transfer of some form or fashion.
Examples of intended usage would be running python scripts to download
data from a remote source, running client software to load data from a
file system into a remote database, or compressing (gzip) or bundling
(tar) files in preparation for data transfer. Examples of what should
not be done include running a database on the server to do data
processing, or running tests that saturate the nodes resources. The
goal is to maintain data transfer systems that have adequate memory
and CPU available for interactive user data transfers.

If you need a space for a more complex workflow, NERSC has dedicated
workflow nodes. If you would like access to these nodes, please fill
out the [Workflow Access Request
Form](https://nersc.service-now.com/com.glideapp.servicecatalog_cat_item_view.do?v=1&sysparm_id=f71e186fdb129700b259fb0e0f961935&sysparm_link_parent=e15706fc0a0a0aa7007fc21e1ab70c2f&sysparm_catalog=e0d08b13c3330100c8b837659bba8fb4&sysparm_catalog_view=catalog_default).

