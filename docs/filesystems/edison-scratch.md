Edison has three local scratch file systems named /scratch1,
/scratch2, and /scratch3. Users are assigned to either /scratch1 or
/scratch2 in a round-robin fashion, so a user will be able to use one
or the other but not both. The third file system is reserved for users
who need large IO bandwidth, and the access is granted
by
[request](https://www.nersc.gov/users/computational-systems/edison/file-storage-and-i-o/edison-scratch3-directory-request-form/).

| Filesystem | Total disk space | Bandwidth |
|------------|:----------------:|:---------:|
| /scratch1  | 2.1 PB           | 48 GB/s   |
| /scratch2  | 2.1 PB           | 48 GB/s   |
| /scratch3  | 3.2 PB           | 72 GB/s   |
