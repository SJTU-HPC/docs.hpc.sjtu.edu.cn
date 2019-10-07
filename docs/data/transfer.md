# Transferring Data

## Overview

NERSC partners with ESNet to provide a high speed connection to the
outside world. NERSC also provides several tools and systems optimized
for data transfer.


## External Data Transfer

!!! tip
    **NERSC recommends transferring data to and from
    NERSC using [Globus](../services/globus.md)**

[Globus](../services/globus.md) is a web-based service that solves
many of the challenges encountered moving data between systems. Globus
provides the most comprehensive, efficient, and easy to use
service for most NERSC users.

However, there are other tools available to transfer data between
NERSC and other sites:

* [scp](../services/scp.md): standard Linux utilities suitable for smaller files (<1GB)
* [GridFTP](../services/gridftp.md): parallel transfer software for large files

## Transferring Data Within NERSC

!!! tip
    **"Do you need to transfer at all?"  If your data is on NERSC
    Global File Systems (`/global/project`, `/global/projecta`,
    `/global/cscratch`), data transfer may not be necessary because
    these file systems are mounted on almost all NERSC
    systems. However, if you are doing a lot of IO with these files,
    you will benefit from staging them on the most performant file
    system. Usually that's the local scratch file system or the Burst
    Buffer.**

* Use the the unix command `cp`, `tar` or `rsync` to copy files within
   the same computational system. For large amounts of data use Globus
   to leverage the automatic retry functionality


## Data Transfer Nodes

The [Data Transfer Nodes (DTNs)](../systems/dtn/index.md) are servers
dedicated for data transfer based upon the ESnet Science DMZ
model. DTNs are tuned to transfer data efficiently, optimized for
bandwidth and have direct access to most of the NERSC file
systems. These transfer nodes are configured within Globus as managed
endpoints available to all NERSC users.

## NERSC FTP Upload Service
NERSC maintains [an FTP upload
service](https://www.nersc.gov/users/storage-and-file-systems/transferring-data/nersc-ftp-upload-service/)
designed for external collaborators to be able to send data to NERSC
staff and users.