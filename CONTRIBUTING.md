# Contribution guide

This outlines standards which should be observed when adding documentation to this repositiory.

## Command prompts

1. when showing a command and sample result, include a prompt indicating where the command is run, eg for a command valid on any NERSC system, use `nersc$`:

    ```console
    nersc$ sqs
    JOBID   ST  USER   NAME         NODES REQUESTED USED  SUBMIT               PARTITION SCHEDULED_START      REASON
    864933  PD  elvis  first-job.*  2     10:00     0:00  2018-01-06T14:14:23  regular   avail_in_~48.0_days  None
    ```

    But if the command is cori-specific, use `cori$`:
    ```console
    cori$ sbatch -Cknl ./first-job.sh
    Submitted batch job 864933
    ```

2. Where possible, replace the username with `elvis` (ie a clearly-arbitrary-but-fake user name)

3. If pasting a snippet of a long output, indicate cuts with `[snip]`:
    ```console
    nersc$ ls -l
    total 23
    drwxrwx--- 2 elvis elvis  512 Jan  5 13:56 accounts
    drwxrwx--- 3 elvis elvis  512 Jan  5 13:56 data
    drwxrwx--- 3 elvis elvis  512 Jan  9 15:35 demo
    drwxrwx--- 2 elvis elvis  512 Jan  5 13:56 img
    -rw-rw---- 1 elvis elvis  407 Jan  9 15:35 index.md
    [snip]
    ```
    
## Writing Style

* [Grammatical Person](https://en.wikiversity.org/wiki/Technical_writing_style#Grammatical_person)
* [Active Voice](https://en.wikiversity.org/wiki/Technical_writing_style#Use_active_voice)

## Shell code should be `bash` not `csh`

## Definitions

* I/O not IO
* Slurm allocation
* NERSC allocation

## Slurm options

* Show both long and short option when introducing an option in text
* Use the short version (where possible) in scripts

