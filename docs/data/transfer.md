# Transferring Data

## Advice and Overview

NERSC provides many facilities for storing data and performing analysis. 
However, transferring data - whether over the wide area network or 
within NERSC - can be expensive and time consuming. 
A good strategy, once your data is resident at NERSC, is to perform your 
analysis in situ, rather than transferring the data elsewhere for analysis.

In addition, NERSC also provides an easy way for research teams to share data through the web via from groups 
project directories.

## External Data Transfer

**NERSC generally recommends transferring data to and from NERSC using 
Globus Online**: a web-based service that solves many of the challenges 
encountered moving data between systems. Globus Online provides the most comprehensive, efficient, and easy 
to use service for most NERSC users.

However, there are other tools people use at NERSC to move data between NERSC and other sites.  Such tools and protocols include:

* SCP/SFTP: for smaller files (<1GB).
* BaBar Copy (bbcp): for large files
* GridFTP: for large files
* xrootd

## Transferring Data Within NERSC

!!! tip "Do you need to transfer at all?"
	If your data is on NERSC Global File Systems (`/global/project`, `/global/projecta`, `/global/cscratch`), it's available at high performance center-wide and data transfer may not be necessary because these file systems are mounted on almost all NERSC systems.

*  Use the the unix command `cp`, `tar` or `rsync` to copy files within the same computational system.
*  To transfer files between computational systems (e.g. Edison local scratch to Hopper local scratch), use SCP/SFTP to transfer smaller files (<10GB), and BaBar Copy (bbcp) or GridFTP for bigger files.
*  HPSS can also be used as an intermediary to transfer files within NERSC.

## Data Transfer Nodes

The Data Transfer Nodes (DTN) are servers dedicated for data transfer based upon the ESnet Science DMZ model. DTNs are tuned to transfer data efficiently, optimized for bandwidth and have direct access to most of the NERSC file systems. These transfer nodes are configured within Globus Online as managed endpoints available to all NERSC users. 

## NERSC File Systems

NERSC has a number of shared file systems that are available from all computers:

* Project
* Home
* Global Scratch

These file systems are ideal for sharing data among different platforms.
