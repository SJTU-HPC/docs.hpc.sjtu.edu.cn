# NERSC Data Management Policy

## Introduction

NERSC provides its users with the means to store, manage, and share
their research data products.

In addition to systems specifically tailored for data-intensive
computations, we provide a variety of storage resources optimized for
different phases of the data lifecycle; tools to enable users to
manage, protect, and control their data; high-speed networks for
intra-site and inter-site (ESnet) data transfer; gateways and portals
for publishing data for broad consumption; and consulting services to
help users craft efficient data management processes for their
projects.  

## OSTP/Office of Science Data Management Requirements

Project Principal Investigators are responsible for meeting OSTP
(Office of Science and Technology Policy) and DOE Office of Science
data management requirements for long-term data sharing and
preservation. The OSTP has issued a memorandum on Increasing Access to
the Results of Federally Funded Scientific Research
(https://obamawhitehouse.archives.gov/sites/default/files/microsites/ostp/public_access_report_to_congress_apr2016_final.pdf)
and the DOE has issued a [Statement on Digital Data Management](https://science.osti.gov/Funding-Opportunities/Digital-Data-Management/).

NERSC resources are intended for users with active allocations, and as
described below, NERSC cannot guarantee long-term data access without
a prior, written, service-level agreement.  Please carefully consider
these policies, including their limitations and restrictions, as you
develop your data management plan.

## Storage Resources

### NERSC Global Filesystem (NGF)

NGF is a collection of centerwide file systems, based on IBM’s
Spectrum Scale, available on all the systems at NERSC. There are three
main file systems that comprise NGF: one providing home directories
(global homes), one providing a common login user environment for all
our systems (global common), and one for sharing data among
collaborators on a science project or team (project). The main focus
of NGF is data sharing, ease of workflow management (i.e., not moving
data around or maintaining unnecessary copies of data), and data
analysis.

### Scratch File Systems

Our primary computational system, Cori, has a dedicated parallel file system
based on Lustre that is optimized for short-term storage of application output
and checkpoints.

!!! warning "Purging on Scratch File Systems"
    Running commands with the intent of circumventing purge policies 
    on scratch filesystems is **not allowed**.

### Burst Buffer

Cori has a Burst Buffer, an additional layer of high-performance SSD
storage, that is available to users on a per-job or short-term basis.

### Archival Storage (HPSS)

HPSS provides long-term archival storage to users at the facility.
The main focus of the system is data stewardship and preservation, but
it also supports some data sharing needs.

## Data Retention Policy

Data in a group's project directory will be retained as long as the
group has an allocation at NERSC. Allocations are renewed on a yearly
basis (see our section on accounts for information on renewal). If a
group's allocation is not renewed, any data in their project directory
will be automatically archived into HPSS. Please see the [project file
system](https://docs.nersc.gov/filesystems/project/) page for details
on the timing of this process.

Currently NERSC has no plans to delete data in HPSS. Should NERSC ever
need to remove data from HPSS, we will attempt to notify the PI of the
group that owns the data first. However, it is strongly recommended
that if you no longer have an allocation at NERSC, you keep a separate
copy of your data somewhere that you have access.

Any data stored in a scratch filesystem is subject to purging and will
be deleted according to the individual system policies.

## Data Transfer, Data Analysis, and Collaborative Capabilities

### Data Transfer Nodes

To enable high speed data transfer, we provide a set of parallel
transfer servers tuned primarily for WAN data movement (into and out
of the facility), and secondarily for high speed local transfers
(e.g., NGF to HPSS). Staff from NERSC and ESnet with extensive data
transfer experience are available to aid scientists in organizing,
packaging, and moving their data to the most appropriate storage
resource at our facility.  

### Globus

Our facility provides numerous Globus endpoints (access points) into
our various storage systems. The key benefit of using
[Globus](https://www.globus.org) software is its ease of use in
providing third party transfers and reliable fire-and-forget
self-managed transfers.

### Direct Web Access to Data

For rapid and simple data sharing, users can enable web access to
files in their project file system. See the [Science Gateways
page](https://docs.nersc.gov/services/science-gateways/#getting-started).

### Science Gateways

For more complex collaborative data, analytics, and computing
projects, NERSC offers users the ability to create web gateways that
can access NERSC computing, NGF, and HPSS resources. This includes
rapid data subselection, deep search including machine learning, and
simulation science in addition to bulk data movement. More details are
[here](https://docs.nersc.gov/services/science-gateways).

### NERSC Web Toolkit (NEWT)

[NEWT](http://newt.nersc.gov/) is an easy to program RESTful web API
for HPC that brings traditional HPC functions of working with batch
jobs and data to the web browser. Science gateways and web data
portals can use NEWT to build powerful web applications using nothing
but HTML and Javascript.

## User Responsibilities

While NERSC provides users with the systems and services that aid them
in managing their data, users have ultimate responsibility for
managing their data:

* Select the most appropriate resources to meet their individual needs.
* Use shared resources responsibly.
* Set appropriate access control limits.
* Archive and back up critical data.
* Never keep a single copy of critical data; NERSC is not responsible for data loss.
* Follow proper use policies, described in the NERSC User Agreement.

## Data Confidentiality and Access Control

The NERSC network is an open research network, intended primarily for
fundamental research. We cannot guarantee complete confidentiality for
data that resides at NERSC. It is your responsibility to set access
controls appropriate to your needs, understanding that your data is
stored on a semi-public system. While we take care to secure our
systems, security breaches could expose your data to others.

Files are protected only using UNIX file permissions based on NIM user
and group IDs. It is the user’s responsibility to ensure that file
permissions and umasks are set to match their needs.

NERSC system administrators with “root” privileges are not constrained
by the file permissions, and they have the ability to open and/or copy
all files on the system. They can also assume a user’s identity on the
system. NERSC HPC consultants also have the capability to assume a
user’s identity to help users troubleshoot application or user
environment problems. Vendor support personnel acting as agents of
NERSC may also have administrative privileges.

Administrators only use these elevated privileges under certain highly
restricted situations and, generally speaking, they only do so when
requested, or if there is a suspected problem/security
issue. Following are specific instances where we might look at your
files:

* We keep copies of all error, output, and job log files and may
review them to determine if a job failure was due to user error or a
system failure.

* If you request our assistance via any mechanism, e.g., help ticket,
direct personal email, in person, etc., we interpret that request to
be explicit permission to view your files if we think doing so will
aid us in resolving your issue.

Users may encrypt data to provide extra measures of privacy if desired.

Under ordinary circumstances, our staff will not copy, expose,
discuss, or in any other way communicate your project information to
anyone outside of your project or NERSC. There are two key exceptions:

* When an account expires or a user leaves a project, NERSC will honor
requests to change file ownership when instructed by the original user
or the most recent principal investigator (or designated PI proxy) of
the sponsoring project.  
* NERSC is required to address, safeguard
against, and report misuse, abuse, security violations, and criminal
activities. NERSC therefore retains the right, at its discretion, to
disclose any and all data files or records of network traffic to
appropriate cyber security organizations and law enforcement
officials.

## Selecting the Appropriate Data Storage Resource

NERSC provides a broad range of storage solutions to address different
needs. The following describes each offered storage solution: its
intended use (capabilities), data protection (backup), data retention,
and available allocations. Users should familiarize themselves with
these solutions, and select the most appropriate for their individual
needs. Most users will use a combination of all file systems, depending on
usage, performance, and data retention needs.

### Home File System

#### Intent

The [home file
system](https://docs.nersc.gov/filesystems/global-home/) is intended
to hold source files, configuration files, etc., and is optimized for
small to medium sized files. It is NOT meant to hold the output from
your application runs; the scratch or project file systems should be
used for computational output.

#### Stability

This file system has redundancy in the servers and storage, and
duplicate copies of metadata, so we expect good stability and the
highest availability in this space.

#### Backup

There are nightly tape backups performed to enable recovery of files
older than one day after file creation. Backups are kept for 90 days.

#### Retention

Data for active users is not purged from this space. A user is
considered inactive if they do not have an active allocation and have
not accessed their data for at least one year. All files in your home
file system will be archived to tape and maintained on disk for one
year from the date your account is deactivated.

#### Default Allocation

Each user is allocated a directory with a 40 GB hard quota in the home
file system. This is the default allocation granted to all users.

### Project File System

#### Intent

The [project file system](https://docs.nersc.gov/filesystems/project/)
is primarily intended for sharing data within a team or across
computational platforms. The project file systems are parallel file
systems optimized for high-bandwidth, large-block-size access to large
files. Once any active production and/or analysis is completed and you
don't need regular access (> 1 year) to the data any longer, you
should either archive the data in the HPSS data archive (below) or
transfer it back to your home institution.


#### Stability

These file systems have redundancy in the servers and storage and
duplicate copies of metadata, so we expect good stability and
reliability. With high demand and capacity shared usage, we do expect
some degradation on availability of the system (97% is the target
overall availability).

#### Backup

Project directories are not backed up. Instead they use a snapshot
capability to provide users a seven-day history of their project
directories. Users can use this capability to recover accidentally
deleted files. It is recommended that users back up any essential data
in our tape archive system or at another location.


#### Retention

Data for active users is not purged from this space. A user or project
will be considered inactive if they do not have an active allocation
and have not accessed the data in at least one year. All project
directories will be archived to tape and maintained on disk for one
year from the date your account is deactivated. For details on the
process, see project file system.

#### Default Allocation

Each repository is allocated a directory in the project file
system. The default quota for project directories is 1 TB and the
default name of a project directory is the repository name
(i.e. m767). These directories are owned by the Principal
Investigators (PIs) and are accessible to everyone in the Unix group
associated with the repository. If files need to be shared with a
group that is different from a repository group, PIs and PI Proxies
can request a special project directory by filling out the Project
Directory Request form in the "Request Forms" section at the [NERSC
help portal](https://nersc.service-now.com/).

NERSC is working to deploy a new file system to replace the project
file system. After this is deployed (expected late 2019), quotas on
this system will be determined by DOE allocations managers based on
information you provide on your ERCAP form.

### Global Common File System

#### Intent

The [global common file system](https://docs.nersc.gov/filesystems/global-common/)
is primarily intended for sharing software installations within a team or across
computational platforms. The global common file system is a parallel file
system optimized for small-block-size access to small
files.

#### Stability

These file systems have redundancy in the servers and storage and
duplicate copies of metadata, so we expect good stability and
reliability.

#### Backup

Global common directories are not backed up. A user may also opt to
archive their data in HPSS, but it is also recommended that software
stacks be managed by a versioning control system such as git.

#### Retention

Data for active users is not purged from this space. A user or project
will be considered inactive if they do not have an active allocation
and have not accessed the data in at least one year.

#### Default Allocation

Each repository is allocated a directory in the global common file
system. The default quota for directories is 10 GB and the default
name of a directory is the repository name (i.e. m767). These
directories are owned by the Principal Investigators (PIs) and are
accessible to everyone in the Unix group associated with the
repository. If files need to be shared with a group that is different
from a repository group, PIs and PI Proxies can request a special
directory by contacting NERSC consulting.

### Scratch File Systems

#### Intent

Cori has a large, local, parallel scratch file system dedicated to the
users of the system. The scratch file system is intended for temporary
uses such as storage of checkpoints or application result output. If
you need to retain files longer than the purge period (see below), the
files should be copied to the project file systems or to HPSS.

!!! warning "Purging on Scratch File Systems"
    Running commands with the intent of circumventing purge policies 
    on scratch filesystems is **not allowed**.

#### Stability

These file systems have redundancy in the servers and storage, so we
expect good stability and reliability. The extreme high demand placed
on these storage systems results in some degradation of availability
for the systems (97% is the target overall availability).

#### Backup

Due to the extremely large volume of data and its temporary nature,
backups are not performed.

#### Retention

These file systems are purged on a regular interval as communicated to
users of the system (e.g., all files not accessed within 12 weeks).
All files in scratch are eligible for deletion one week after your
account is deactivated.

#### Default Allocation

Each user is allocated a directory in the Cori's scratch file
system. The default quota is 20 TB.

### Burst Buffer

#### Intent

Cori has a layer of SSD storage sitting within the high-speed network
that provides very high-performance I/O for user applications. In
particular, it performs well for large checkpoint/restart codes and
for applications that are IOPs-heavy, read/write many small files or
random reads/writes. The Burst Buffer can be used on a per-job basis,
or for a short-term persistent reservation. It is not a long-term
storage layer - the user is responsible for transferring data to other
file systems for longer retention.

#### Stability

The Burst Buffer has very high stability - since it sits within the
high-speed interconnect of Cori is it available whenever Cori compute
nodes are available. We have seen no instances of data corruption, but
we warn users that data is not guaranteed on the Burst Buffer due to
the risk of SSD failure.

#### Backup

No backups are performed.

#### Retention

Data is removed at the end of a Burst Buffer reservation. This is
either at the end of a batch job (for a scratch reservation) or when a
persistent reservation is destroyed. It is the responsibility of the
user to stage or copy data that needs to be retained. We ask that
users remove a persistent reservation after 6 weeks - after 8 weeks
the reservation may be removed by an administrator.

#### Default Allocation

Users must request usage of the Burst Buffer from their job submission
script. All users are limited to a total of 50TB of Burst Buffer
storage. Larger requests may be considered in special circumstances.

### HPSS Data Archive

#### Intent

The HPSS data archive is intended for long-term, offline storage
(tape) of results that you wish to retain, but you do not need
immediate access to. The primary HPSS access is via HSI, or HTAR for
small files.  We also have access to HPSS via gridFTP or Globus. Tape
backups of the other file systems are stored in a special partition of
the archive.

#### Stability

The HPSS service is configured to be stable and reliable, but is
routinely taken offline for short periods for planned maintenance.

#### Backup

By default, a single copy of the data will be written to tape. Data
loss due to hardware faults can occur, but is very rare. All tapes are
currently stored at the NERSC facility. There is no off-site backup,
and in principle there is a chance of permanent data loss in the case
of a site disaster such as a fire. Critical data should be manually
protected by making an explicit second copy: You can make another copy
within the data archive, or you can copy the data to another location.


#### Retention

We do not actively remove data from HPSS, and will communicate with
users or data owners should a need arise to reclaim space. Note that
in NERSC’s history, we have not deleted files in the HPSS archive due
to storage limitations. Projects requiring a firm commitment for data
retention should contact NERSC.

#### Default Allocation

Projects receive HPSS allocations at the same time that computational
resources are allocated. Each group recieves a default allocation of 1
TB.

#### Appropriate Use

The Archive is a shared multi-user resource. Use of the Archive is
subject to interruption if it is determined that a user's method or
pattern of storing or retrieving files significantly impacts other
users' activities or undermines the long term viability of the system.

### Requests for Data Storage Beyond Defaults

NERSC establishes defaults for allocations, retention, and backup to
serve the needs of the majority of our users, avoid oversubscription,
and place practical limits on the costs of storage provided. We are
willing to work with individual projects to accommodate special
needs. If you need additional space, please fill out the "Request a
Disk Quota Change" in form in the "Request Forms" section at the
[NERSC help portal](https://nersc.service-now.com/).


NERSC is working to deploy a new file system to replace the project
file system. After this is deployed (expected late 2019), quotas on
this system will be determined by DOE allocations managers based on
information you provide on your ERCAP form.

### Proper Use of Software and Data

Please consult the NERSC User Agreement for the policies regarding
appropriate use (and limitations on use) of software and data. NOTE:
The following classes of software and data raise red flags, and may be
prohibited or restricted in some way. Users should carefully consult
the User Agreement before moving such information to NERSC systems:

* Classified or controlled military or defense information
* Software without proper licenses
* Export controlled or ITAR software or data
* Personally identifiable information
* Medical or health information
* Proprietary information
