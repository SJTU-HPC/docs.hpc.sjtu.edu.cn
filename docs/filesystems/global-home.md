## Summary

Home directories provide a convenient means for a user to have
access to files such as dotfiles, source files, input files,
configuration files regardless of the platform.

| space	quota | inode quota | purge time | backups |
|-------------|-------------|------------|---------|
| 40 GB       | 1 M         | none       | yes     |

## Usage

Refer to your home directory using the environment variable `$HOME`
whenever possible. The absolute path may change, but the value of
`$HOME` will always be correct.

## Quotas

!!! warning
	Quota increases in global homes are approved only in
	*extremely* unusual circumstances.

## Performance

Performance of global homes is optimized for small files and is
suitable for compiling and linking executables. Global home
directories are not intended for large, streaming I/O. **User
applications that depend on high-bandwidth for streaming large files
should not be run in your `$HOME` directory.**

## Backups

All NERSC users should backup important files on a regular
basis. Ultimately, it is the user's responsibility to prevent data
loss. However, NERSC provides mechanisms to support user's in
protecting against data loss.

### Snapshots

A *snapshot* capability is used to provide users a seven-day history
of their home directories. Every directory and sub-directory in
`$HOME` contains a `.snapshots` entry.

* `.snapshots` is invisble to `ls`, `ls -a`, `find` and similar commands
* Contents are visible through `ls -F .snapshots`
* Can be browsed normally after `cd .snapshots`
* Files cannot be created, deleted or edited in snapshots
* Files can *only* be copied *out* of a snapshot

### Archive

Global homes are backed up to [HPSS](archive.md) monthly.

If the snapshot capability does not meet your need
contact [NERSC Consulting](/help/index.md) with pathnames and
timestamps of the missing data.

!!! note
	Restore requests may take several days to complete.
