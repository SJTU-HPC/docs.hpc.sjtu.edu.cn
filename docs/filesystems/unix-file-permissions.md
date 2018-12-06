# Unix File Permissions

## Brief Overview

Every file (and directory) has an owner, an associated Unix group, and a set of
permission flags that specify separate read, write, and execute permissions for
the "user" (owner), "group", and "other".  Group permissions apply to all users
who belong to the group associated with the file.  "Other" is also sometimes
known as "world" permissions, and applies to all users who can login to the
system.  The command `ls -l` displays the permissions and associated group for
any file.  Here is an example of the output of this command:

```
drwx------ 2 elvis elvis  2048 Jun 12 2012  private
-rw------- 2 elvis elvis  1327 Apr  9 2012  try.f90
-rwx------ 2 elvis elvis 12040 Apr  9 2012  a.out
drwxr-x--- 2 elvis bigsci 2048 Oct 17 2011  share
drwxr-xr-x 3 elvis bigsci 2048 Nov 13 2011  public
```

From left to right, the fields above represent:

1. set of ten permission flags
2. link count (irrelevant to this topic)
3. owner
4. associated group
5. size
6. date of last modification
7. name of file

The permission flags from left to right are:

| Position | Meaning                                                  |
|----------|----------------------------------------------------------|
| 1        | "d" if a directory, "-" if a normal file                 |
| 2, 3, 4  | read, write, execute permission for user (owner) of file |
| 5, 6, 7  | read, write, execute permission for group                |
| 8, 9, 10 | read, write, execute permission for other (world)        |

and have the following meanings:

| Value | Meaning                                                             |
|-------|---------------------------------------------------------------------|
| -     | Flag is not set.                                                    |
| r     | File is readable.                                                   |
| w     | File is writable. For directories, files may be created or removed. |
| x     | File is executable. For directories, files may be listed.           |
| s     | Set group ID (sgid). For directories, files created therein will be associated with the same group as the directory, rather than default group of the user.  Subdirectories created therein will not only have the same group, but will also inherit the sgid setting. |

These definitions can be used to interpret the example output of `ls -l`
presented above:

```
drwx------ 2 elvis elvis  2048 Jun 12 2012  private
```

This is a directory named "private", owned by user elvis and associated with
Unix group elvis.  The directory has read, write, and execute permissions for
the owner, and no permissions for any other user.

```
-rw------- 2 elvis elvis  1327 Apr  9 2012  try.f90
```

This is a normal file named "try.f90", owned by user elvis and associated with
group elvis.  It is readable and writable by the owner, but is not accessible
to any other user.

```
-rwx------ 2 elvis elvis 12040 Apr  9 2012  a.out
```

This is a normal file named "a.out", owned by user elvis and associated with
group elvis.  It is executable, as well as readable and writable, for the
owner only.

```
drwxr-x--- 2 elvis bigsci 2048 Oct 17 2011  share
```

This is a directory named "share", owned by user elvis and associated with
group bigsci.  The owner can read and write the directory; all members of the
file group bigsci can list the contents of the directory.  Presumably, this
directory would contain files that also have "group read" permissions.

```
drwxr-xr-x 3 elvis bigsci 2048 Nov 13 2011  public
```

This is a directory named "public", owned by user elvis and associated with
group bigsci.  The owner can read and write the directory; all other users can
only read the contents of the directory.  A directory such as this would most
likely contain files that have "world read" permissions.

## Useful File Permission Commands

### umask

When a file is created, the permission flags are set according to the file mode
creation mask, which can be set using the "umask" command. The file mode
creation mask (sometimes referred to as "the umask") is a three-digit octal
value whose nine bits correspond to fields 2-10 of the permission flags. The
resulting permissions are calculated via the bitwise AND of the unary
complement of the argument (using bitwise NOT) and the default permissions
specified by the shell (typically 666 for files and 777 for directories).
Common useful values are:

| umask value | File Permissions | Directory Permissions |
|-------------|------------------|-----------------------|
| 002         | -rw-rw-r--       | drwxrwxr-x            |
| 007         | -rw-rw----       | 	drwxrwx---           |
| 022         | -rw-r--r--       | drwxr-xr-x            |
| 027         | -rw-r-----       | drwxr-x---            |
| 077         | -rw-------       | drwx------            |

Note that at NERSC, a default umask of 007 is set in .bash_profile. This is
read *after* .bashrc, so setting umask in your .bashrc.ext won't work, you will
need to set it in your .bash_profile.ext.

### chmod

The chmod ("change mode") command is used to change the permission flags on
existing files. It can be applied recursively using the "-R" option. It can be
invoked with either octal values representing the permission flags, or with
symbolic representations of the flags. The octal values have the following
meaning:

| Octal Digit | Binary Representation (rwx) | Permission                                  |
|-------------|-----------------------------|---------------------------------------------|
| 0           | 000                         | none                                        |
| 1           | 001                         | execute only                                |
| 2           | 010                         | write only                                  |
| 3           | 011                         | write and execute                           |
| 4           | 100                         | read only                                   |
| 5           | 101                         | read and execute                            |
| 6           | 110                         | read and write                              |
| 7           | 111                         | read, write, and execute (full permissions) |

Here is an example of chmod using octal values:

```
nersc$ umask
0077
nersc$ touch foo
nersc$ ls -l foo
-rw------- 1 elvis elvis 0 Nov 19 14:49 foo
nersc$ chmod 755 foo
nersc$ ls -l foo
-rwxr-xr-x 1 elvis elvis 0 Nov 19 14:49 foo
```

In the above example, the umask for user elvis results in a file that is
read-write for the user, with no other permissions. The chmod command specifies
read-write-execute permissions for the user, and read-execute permissions for
group and other.

Here is the format of the chmod command when using symbolic values:

```
chmod [-R] [classes][operator][modes] file ...
```

The *classes* determine to which combination of user/group/other the operation
will apply, the *operator* specifies whether permissions are being added or
removed, and the *modes* specify the permissions to be added or removed.
Classes are formed by combining one or more of the following letters:

| Letter | Class | Description                                                            |
|--------|-------|------------------------------------------------------------------------|
| u      | user  | Owner of the file                                                      |
| g      | group | Users who are members of the file's group                              |
| o      | other | Users who are not the owner of the file or members of the file's group |
| a      | all   | All of the above (equivalent to "ugo")                                 |

The following *operators* are supported:

| Operator | Description                                                             |
|----------|-------------------------------------------------------------------------|
| +        | Add the specified modes to the specified classes.                       |
| -        | Remove the specified modes from the specified classes.                  |
| =        | The specified modes are made the exact modes for the specified classes. |

The modes specify which permissions are to be added to or removed from the
specified classes. There are three primary values which correspond to the basic
permissions, and two less frequently-used values that are useful in specific
circumstances:

| Mode | Name              | Description                                 |
|------|-------------------|---------------------------------------------|
| r    | read              | Read a file or list a directory's contents. |
| w    | write             | Write to a file or directory.               |
| x    | execute           | Execute a file or traverse a directory.     |
| X    | "special" execute | This is a slightly more restrictive version of "x".  It applies execute permissions to directories in all cases, and to files **only if** at least one execute permission bit is already set.  It is typically used with the "+" operator and the "-R" option, to give group and/or other access to a large directory tree, without setting execute permissions on normal (non-executable) files (e.g., text files).  For example, `chmod -R go+rx bigdir` would set read and execute permissions on every file (including text files) and directory in the bigdir directory, recursively, for group and other.  The command `chmod -R go+rX bigdir` would set read and execute permissions on every directory, and would set group and other read and execute permissions on files that were already executable by the owner. |
| s    | setgid or sgid    | This setting is typically applied to directories.  If set, any file created in that directory will be associated with the directory's group, rather than with the default file group of the owner. This is useful in setting up directories where many users share access.  This setting is sometimes referred to as the "sticky bit", although that phrase has a historical meaning unrelated to this context. |

Sets of class/operator/mode may separated by commas. Using the above
definitions, the previous (octal notation) example can be done symbolically:

```
nersc$ umask
0077
nersc$ touch foo
nersc$ ls -l foo
-rw------- 1 elvis elvis 0 Nov 19 14:49 foo
nersc$ chmod u+x,go+rx foo
nersc$ ls -l foo
-rwxr-xr-x 1 elvis elvis 0 Nov 19 14:49 foo
```
