# Filesystem overview

## Summary

NERSC file systems can be divided into two categories: local and
global. Local file systems are only accessible on a single platform
and provide the best performance; global file systems are accessible
on multiple platforms, simplifying data sharing between platforms.

File systems are configured for different purposes. Each machine has
access to at least three different file systems with different levels
of performance, permanence and available space.

| file system     | snapshots | backup | purging | access          |
|-----------------|-----------|--------|---------|-----------------|
| project         | yes       | no     | no      | repository      |
| home            | yes       | yes    | no      | user            |
| common          | no        | no     | no      | repository      |
| Cori scratch    | no        | no     | yes     | user            |
| Edison scratch  | no        | no     | yes     | user            |
| Edison scratch3 | no        | no     | yes     | special request |
| HPSS            | no        | no     | no      | user            |

!!! note
	See [quotas](quotas.md) for detailed information about inode,
	space quotas and file system purge policies.

## Global storage

### [Global Home](global-home.md)

Permanent, relatively small storage for data like source code, shell
scripts that you want to keep. This file system is not tuned for high
performance for parallel jobs. Referenced by the environment variable
`$HOME`.

### [Common](global-common.md)

A performant platform to install software stacks and compile
code. Mounted read-only on compute nodes.

### [Project](project.md)

Large, permanent, medium-performance file system. Project directories
are intended for sharing data within a group of researchers.

### [Archive](archive.md) (HPSS)

A high capacity tape archive intended for long term storage of
inactive and important data. Accessible from all systems at
NERSC. Space quotas are allocation dependent

The High Performance Storage System (HPSS) is a modern, flexible,
performance-oriented mass storage system. It has been used at NERSC
for archival storage since 1998. HPSS is intended for long term
storage of data that is not frequently accessed.

## Local storage

### Scratch

[Edison](/systems/edison/index.md) and [Cori](/systems/cori/index.md)
each have dedicated, large, local, parallel scratch file systems based
on Lustre. The scratch file systems are intended for temporary uses
such as storage of checkpoints or application input and output.

* [Cori scratch](/filesystems/cori-scratch.md)
* [Edison scratch](/filesystems/edison-scratch.md)

### [Burst Buffer](/filesystems/cori-burst-buffer.md)

Cori's [Burst Buffer](/filesystems/cori-burst-buffer.md) provides very
high performance I/O on a per-job or short-term basis. It is
particularly useful for codes that are I/O-bound, for example, codes
that produce large checkpoint files, or that have small or random I/O
reads/writes.
