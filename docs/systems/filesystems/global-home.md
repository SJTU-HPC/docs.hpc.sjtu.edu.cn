Global home directories provide a convenient means for a user to have access to files such as dotfiles, source files, input files, configuration files regardless of the platform.

## Usage

Wherever possible, you should refer to your home directory using the environment variable `$HOME`. The absolute path to your home directory may change, but the value of `$HOME` will always be correct.

For security reasons, you should never allow "world write" access to your `$HOME` directory or your `$HOME/.ssh` directory. NERSC scans for such security weakness, and, if detected, will change the permissions on your directories.

## Quotas

| Type   | Quota |
|--------|:-----:|
| Space  | 40 GB |
| inodes | 1 M   |

!!! warning
	Quota increases in global homes are approved only in *extremely* unusual circumstances.

!!! note "Purge policy"
	This filesystem is not subject to purging.

## Performance

Performance of global homes is optimized for small files. This is suitable for compiling and linking executables. Global home directories are not intended for large, streaming I/O. **User applications that depend on high-bandwidth for streaming large files should not be run in your `$HOME` directory.**

## Snapshots

Global homes use a *snapshot* capability to provide users a seven-day history of their global home directories. Every directory and sub-directory in global homes contains a ".snapshots" entry. 

* `.snapshots` is invisble to `ls`, `ls -a`, `find` and similar commands
* Contents are visible through `ls -F .snapshots`
* Can be browsed normally after `cd .snapshots`
* Files cannot be created, deleted or edited in snapshots
* Files can *only* be copied *out* of a snapshot

## Backup/Restore

Global homes are backed up to HPSS on a regular basis.  If you require a restoration of lost data that cannot be accomplished via the snapshots capability, please contact NERSC Consulting with pathnames and timestamps of the missing data.  Such restore requests may take a few days to complete.

!!! warning
	 All NERSC users should back up important files to HPSS on a regular basis.  Ultimately, it is your responsibility to protect yourself from data loss.
