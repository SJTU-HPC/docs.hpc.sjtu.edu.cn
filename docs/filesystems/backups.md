## Backups

!!! danger 
	All NERSC users should back up important files to HPSS on
    a regular basis.  **Ultimately, it is your responsibility to
    protect yourself from data loss.**

### Snapshots

Global homes and project use a *snapshot* capability to provide users
a seven-day history of their directories. Every directory and
sub-directory in global homes contains a ".snapshots" entry.

* `.snapshots` is invisble to `ls`, `ls -a`, `find` and similar
  commands
* Contents are visible through `ls -F .snapshots`
* Can be browsed normally after `cd .snapshots`
* Files cannot be created, deleted or edited in snapshots
* Files can *only* be copied *out* of a snapshot

### Backup/Restore

Global homes are backed up to HPSS on a regular basis.  If you require
a restoration of lost data that cannot be accomplished via the
snapshots capability, please contact NERSC Consulting with pathnames
and timestamps of the missing data.  Such restore requests may take a
few days to complete.

## Purging

Files in `$SCRATCH` directories may be purged if they are older than
12 weeks (defined by last access time).

!!! warning
	`$SCRATCH` directories are **not** backed up
