The project file system is a global file system available on all NERSC computational systems. It allows sharing of data between users, systems, and (via science gateways) the "outside world".

## Usage

Project directories are created in `/project/projectdirs`. The name of a *default* project directory is the same as its associated MPP repository. Default project directories are provided to every MPP project. There is also a Unix group with the same name; all members of the repository are also members of the group.  Access to the project directory is controlled by membership in this group.

Occasionally there are cases where the above model is too limiting. 
Some large projects might want a project directory to be accessible by members of multiple repositories or some long-term projects outlive the specific repositories that constitute them
In these cases, a project directory administrator may request the creation of a "designer" project directory with a specific name. This will result in the creation of a new Unix group with that name, consisting solely of the project directory administrator, followed by the creation of the project directory itself. The project directory administrator must then use NIM to add users to the newly-created Unix group (this is a very simple operation). Only these users will be able to access the project directory. If you are a PI or a PI Proxy, you can request a designer project directory in NIM. Search for the repository name you wish this designer project directory to be attached to (the user who can access this directory do not need to be members of this repository, but the directory must be attached to a repository where you are a PI or PI Proxy). Scroll to the bottom of the "Project Information" tab and you will see a link that says "Request a custom project directory".


## Quotas

| Type   | Quota |
|--------|:-----:|
| Space  | 1 TB  |
| inodes | 1 M   |

!!! warning
	Quota increases in global homes are approved only in *extremely* unusual circumstances.

!!! note "Purge policy"
	This filesystem is not subject to purging.

## Performance

The system has a peak aggregate bandwidth of 130 GB/sec bandwidth for streaming I/O, although actual performance for user applications will depend on a variety of factors. Because NGF is a distributed network filesystem, performance typically will be less than that of filesystems that are local to a specific compute platform. This is usually an issue only for applications whose overall performance is sensitive to I/O performance.

## Snapshots

Global homes use a *snapshot* capability to provide users a seven-day history of their global home directories. Every directory and sub-directory in global homes contains a ".snapshots" entry. 

* `.snapshots` is invisble to `ls`, `ls -a`, `find` and similar commands
* Contents are visible through `ls -F .snapshots`
* Can be browsed normally after `cd .snapshots`
* Files cannot be created, deleted or edited in snapshots
* Files can *only* be copied *out* of a snapshot

## Backup/Restore

No managed backups or project directories are done by NERSC. Instead, users can recover files lost in the last seven days by using the snapshot functionality.

!!! warning
	All NERSC users should back up important files to HPSS on a regular basis.  Ultimately, it is your responsibility to protect from data loss.
	
## Lifetime

Project directories will remain in existence as long as the owning project is active. Projects typically "end" at the end of a NERSC Allocation Year. This happens when the PI chooses not to renew the project, or DOE chooses not to provide an allocation for a renewal request. In either case, the following steps will occur following the termination of the project:

1. **Day -365** - The start of the new Allocation Year and no Project renewal

	The data in the project directory will remain available on the project file system until the start of the next Allocation Year. After that time, we will start the archiving process to move the data off of the project file system into the HPSS tape archive.
	
1. **Day 0** - The start of the following Allocation Year

	Users of the affected project directory will be notified that the project directory will be archived, and then removed from the file system, in 90 days.
	
1. **Day 30**

	The project directory will become read-only.  Users will still be able to retrieve data from the project directory, but will not be able to store data into the directory.

1. **Day 60**
   
    The full pathname to the project directory will be modified.  This will most likely cause automated scripts to fail.
   
1. **Day 90**

	User access to the directory will be terminated. The directory will then be archived in HPSS, under ownership of the PI, and subsequently removed from the file system.
