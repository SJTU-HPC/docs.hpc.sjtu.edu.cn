# Overview

NERSC file systems can be divided into two categories: local and global. Local file systems are only accessible on a single platform and provide the best performance; global file systems are accessible on multiple platforms, simplifying data sharing between platforms. 

File systems are configured for different purposes. On each machine you have access to at least three different file systems with different levels of performance, permanence and available space.

## Global storage

### Global Home

!!! info
	Permanent, relatively small storage for data like source code, shell scripts that you want to keep. This file system is not tuned for high performance for parallel jobs. Referenced by the environment variable `$HOME`.

Global home directories (or "global homes") provide a convenient means for a user to have access to dotfiles, source files, input files, configuration files, etc., regardless of the platform the user is logged in to. 

Performance of global homes is optimized for small files. This is suitable for compiling and linking executables, for example. Home directories are not intended for large, streaming I/O.
**User applications that depend on high-bandwidth for streaming large files should not be run in your home directory.**

Wherever possible, you should refer to your home directory using the environment variable `$HOME`. The absolute path to your home directory may change, but the value of `$HOME` will always be correct.

!!!danger
	For security reasons, you should never allow "world write" access to your `$HOME` directory or your `$HOME/.ssh` directory. NERSC scans for such security weakness, and, if detected, will change the permissions on your directories.

### Global Common

!!! info
	A performant platform to install software stacks and compile code.

The global common file system is a global file system available on all NERSC computational systems. It offers a performant platform to install software stacks and compile code. Directories are provided by default to every MPP project. Additional global common directories can be provided upon request.

### Project

!!! info
	Large, permanent, medium-performance file system. Project directories are inten\
ded for sharing data within a group of researchers.

The project file system is a global file system available on all NERSC computational systems. It allows sharing of data between users, systems, and/or (via science gateways) the "outside world".  Default project directories are provided to every MPP project. The system has a peak aggregate bandwidth of 130 GB/sec bandwidth for streaming I/O, although actual performance for user applications will depend on a variety of factors. Because NGF is a distributed network filesystem, performance typically will be less than that of filesystems that are local to a specific compute platform. This is usually an issue only for applications whose overall performance is sensitive to I/O performance.

!!!note
	Additional project directories can be provided upon request.	
	
### HPSS

!!! info
	A high capacity tape archive intended for long term storage of inactive and imp\
ortant data. Accessible from all systems at NERSC. Space quotas are allocation dependent

The High Performance Storage System (HPSS) is a modern, flexible, performance-oriented mass storage system. It has been used at NERSC for archival storage since 1998. HPSS is intended for long term storage of data that is not frequently accessed.

HPSS is Hierarchical Storage Management (HSM) software developed by a collaboration of DOE labs, of which NERSC is a participant, and IBM. The HSM software enables all user data to be ingested onto high performance disk arrays and automatically migrated to a very large enterprise tape subsystem for long-term retention. The disk cache in HPSS is designed to retain many days worth of new data and the tape subsystem is designed to provide the most cost-effective long-term scalable data storage available.

## Local storage

### Edison scratch

!!! info 
	Optimized for high-bandwidth, large-block-size access to large files.

Edison has three local scratch file systems named /scratch1, /scratch2, and /scratch3. The first two file systems have 2.1 PB disk space and 48 GB/sec IO bandwidth each, while the third one has 3.2 PB disk, the peak IO bandwidth is 72G/s. Users are assigned to either /scratch1 or /scratch2 in a round-robin fashion, so a user will be able to use one or the other but not both. The third file system is reserved for users who need large IO bandwidth, and the access is granted by request. If you need large IO bandwidth to conduct more efficient computations and data analysis at NERSC, please submit your request by filling out the [SCRATCH3 Directory Request Form](http://www.nersc.gov/users/computational-systems/edison/file-storage-and-i-o/edison-scratch3-directory-request-form/).

The /scratch1 or /scratch2 file systems should always be referenced using the environment variable `$SCRATCH` (which expands to /scratch1/scratchdirs/YourUserName or /scratch2/scratchdirs/YourUserName on Edison). The scratch file systems are available from all nodes (login, and compute nodes) and are tuned for high performance. We recommend that you run your jobs, especially data intensive ones, from the scratch file systems.

All users have 10 TB of quota for the scratch file system. 

!!!warning
	If your `$SCRATCH` usage exceeds your quota, you will not be able to submit batch jobs until you reduce your usage. We have not set the quotas on the /scratch3 file system. The batch job submit filter checks only the usage of the /scratch1 or /scratch2, but not /scratch3.

The `myquota` command will display your current usage and quota.  NERSC sometimes grants temporary quota increases for legitimate purposes. To apply for such an increase, please use the [Disk Quota Increase Form](http://www.nersc.gov/users/storage-and-file-systems/file-systems/data-storage-quota-increase-request/).

The scratch file systems are subject to purging. Files in your $SCRATCH directory that are older than 12 weeks (defined by last access time) are removed. Please make sure to back up your important files (e.g. to HPSS).   Instructions for HPSS are here.

The /scratch3 file system is subject to purging as well - files that are older than 8 weeks will be deleted from the /scartch3 file systems.


### Cori cratch

!!! info 
	Optimized for high-bandwidth, large-block-size access to large files.
	
Cori has one scratch file system named /global/cscratch1. The file system has 30 PB disk space and >700 GB/sec IO bandwidth. 

The /global/cscratch1 file system should always be referenced using the environment variable $SCRATCH (which expands to /global/cscratch1/sd/YourUserName). The scratch file system is available from all nodes (login, MOM, and compute nodes) and is tuned for high performance. We recommend that you run your jobs, especially data intensive ones, from the burst buffer or the scratch file system.

All users have 20 TB of quota for the scratch file system. If your $SCRATCH usage exceeds your quota, you will not be able to submit batch jobs until you reduce your usage.  The batch job submit filter checks the usage of the /global/cscratch1.

The `myquota` command will display your current usage and quota.  NERSC sometimes grants temporary quota increases for legitimate purposes. To apply for such an increase, please use the Disk Quota Increase Form.

!!! warning
	The scratch file system is subject to purging.
	
### Cori Burst Buffer

Temporary, flexible, high-performance SSD file system that sits within the High Speed Network (HSN) on Cori. Accessible only from compute nodes, the Burst Buffer provides per-job (or short-term) storage for I/O intensive codes. 
