# Global common

The global common file system is a global file system available on all
NERSC computational systems. It offers a performant platform to
install software stacks and compile code. Directories are provided by
default to every MPP project. Additional global common directories can
be provided upon request.

| space	quota | inode quota | purge time | backups |
|-------------|-------------|------------|---------|
| 10 GB       | 1 M         | none       | yes     |

Your groups usage can be viewed with the ```cmnquota <proj_name>```
command.

## Usage

Global common directories are created in
`/global/common/software`. The name of a "default" project directory
is the same as its associated MPP repository. There is also a Unix
group with the same name; all members of the repository are also
members of the group. Access to the global common directory is
controlled by membership in this group. Because this directory is
shared across all systems, you may want to install your software
stacks into separate subdirectories depending on the system or the
processing architecture. For some general programs you can use the
same installs across all systems, but for best performance, we
recommend separate installs for each system and architecture.

!!! warning 
     Global common is mounted read-only on the computes to
     improve performance. Software installs should be done on the
     login nodes.


## Performance

The global common system is optimized for software installation. It
has a smaller block size and is mounted read-only on the
computes. This allows us to turn on client-side caching which
dramatically increases the read time of shared libraries across many
nodes. The cache expires after five minutes, so it is not recommended
you compute against code that is being actively (i.e. within five
minutes) changed.

## Backup/Restore

!!! warning
	No managed backups of global common directories are done
	by NERSC.
