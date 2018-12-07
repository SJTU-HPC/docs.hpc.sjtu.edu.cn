# Project filesystem

The project file system is a global file system available on all NERSC
computational systems. It allows sharing of data between users,
systems, and the "outside world".

## Usage

Every MPP repository has an associated project directory and unix
group. Project directories are created in `/project/projectdirs`.
All members of the project have access through their membership in the
unix group.

Occasionally there are cases where the above model is too
limiting. For example:

* large projects with multiple MPP repositories
* long-term projects which outlive specific MPP repositories

In these cases, a project directory administrator may request the
creation of a "designer" project directory with a specific name. This
will result in the creation of a new Unix group with that name,
consisting solely of the project directory administrator, followed by
the creation of the project directory itself. The project directory
administrator must then use NIM to add users to the newly-created Unix
group.

!!! info
	If you are a _PI_ or a _PI Proxy_, you can request a designer project
	directory in NIM.

	1. Search for the MPP repository name you wish this designer
	   project directory to be attached to.
	1. Scroll to the bottom of the "Project Information" tab and you
	   will see a link that says "Request a custom project directory".

## Quotas

| space quota | inode quota | purge time | backups       |
|-------------|-------------|------------|---------------|
| 1 TB        | 1 M         | none       | snapshots     |

## Performance

The system has a peak aggregate bandwidth of 130 GB/sec bandwidth for
streaming I/O.

## Snapshots

Global homes use a *snapshot* capability to provide users a seven-day
history of their global home directories. Every directory and
sub-directory in global homes contains a ".snapshots" entry.

* `.snapshots` is invisble to `ls`, `ls -a`, `find` and similar commands
* Contents are visible through `ls -F .snapshots`
* Can be browsed normally after `cd .snapshots`
* Files cannot be created, deleted or edited in snapshots
* Files can *only* be copied *out* of a snapshot

## Lifetime

Project directories will remain in existence as long as the owning
project is active. Projects typically "end" at the end of a NERSC
Allocation Year. This happens when the PI chooses not to renew the
project, or DOE chooses not to provide an allocation for a renewal
request. In either case, the following steps will occur following the
termination of the project:

1. **-365 days** - The start of the new Allocation Year and no Project
   renewal

	The data in the project directory will remain available on the
    project file system until the start of the next Allocation
    Year. Archival process begins.

1. **+0 days** - The start of the following Allocation Year

	Users notified that the affected project directory will be
    archived, and then removed from the file system in 90 days.

1. **+30 days**

	The project directory will become read-only.

1. **+60 days**

    The full pathname to the project directory will be
    modified. Automated scripts will likely fail.

1. **+90 days**

	User access to the directory will be terminated. The directory
    will then be archived in HPSS, under ownership of the PI, and
    subsequently removed from the file system.
