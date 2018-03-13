# Overview

NERSC file systems can be divided into two categories: local and global. Local file systems are only accessible on a single platform and provide the best performance; global file systems are accessible on multiple platforms, simplifying data sharing between platforms. 

File systems are configured for different purposes. On each machine you have access to at least three different file systems with different levels of performance, permanence and available space.

## Global storage

### [Global Home](global-home.md)

Permanent, relatively small storage for data like source code, shell scripts that you want to keep. This file system is not tuned for high performance for parallel jobs. Referenced by the environment variable `$HOME`.

### [Global Common](global-common.md)

A performant platform to install software stacks and compile code. Mounted read-only on compute nodes.

### [Project](project.md)

Large, permanent, medium-performance file system. Project directories are inten\
ded for sharing data within a group of researchers.
	
### HPSS

A high capacity tape archive intended for long term storage of inactive and imp\
ortant data. Accessible from all systems at NERSC. Space quotas are allocation dependent

The High Performance Storage System (HPSS) is a modern, flexible, performance-oriented mass storage system. It has been used at NERSC for archival storage since 1998. HPSS is intended for long term storage of data that is not frequently accessed.

## Local storage

### [Edison scratch](edison-scratch.md)

Edison has three local scratch file systems optimized for high-bandwidth, large-block-size access to large files named /scratch1, /scratch2, and /scratch3. The scratch file systems are available from all nodes (login, and compute nodes) and are tuned for high performance. We recommend that you run your jobs, especially data intensive ones, from the scratch file systems.

### [Cori scratch](cori-scratch.md)

!!!tip
	`/global/cscratch1` is also mounted on Edison

Cori has one scratch file system named /global/cscratch1 optimized for high-bandwidth, large-block-size access to large files. The scratch file system is available from all nodes (login, MOM, and compute nodes) and is tuned for high performance. We recommend that you run your jobs, especially data intensive ones, from the burst buffer or the scratch file system.
	
### [Cori Burst Buffer](cori-burst-buffer.md)

Temporary, flexible, high-performance SSD file system that sits within the High Speed Network (HSN) on Cori. Accessible only from compute nodes, the Burst Buffer provides per-job (or short-term) storage for I/O intensive codes. 
