# CVMFS 

[CVMFS (CernVM-File System)](https://cvmfs.readthedocs.io/en/stable/)
is a software distribution service heavily used by High-Energy Physics (HEP) experiments to deliver software to multiple different computing systems worldwide. 
It presents remote software pulled over http as a POSIX read-only filesystem in user space (normally using FUSE). 
Access is performant and scalable due to multiple levels of caching such as a local file cache and squid web caches.

## NERSC Setup

At NERSC CVMFS is mounted on Cori and Edison compute nodes as an NFS mount via DVMFS (Cray's I/O forwarder for remote mounted filesystems that is also used for /project filesystems). 
It is also mounted on Cori and Edison login nodes. The NFS servers themselves mount cvmfs using FUSE in the normal way and we have dedicated local squid caches. 
We believe this setup to now be stable and perform well enough but users should please contact us if you see any problems, 
or if you need a new repository that is not currently available, via the [online help desk](https://help.nersc.gov/) or email to consult@nersc.gov. 
PDSF systems also provide CVMFS though via its 'normal' mechanism.

## Using CVMFS at NERSC

From a normal user perspective however this custom setup mentioned above doesn't change that all our mounted repositories appear under /cvmfs
    
    cori$ ls /cvmfs
    <various repositories>
    
CVMFS is also available on Cori and Edison compute nodes with or without shifter 

If using [shifter](/development/shifter/how-to-use.md) you should specify the `--modules=cvmfs` flag to shifter to make that mount appear inside your container (unless you want to use your own 
version of cvmfs inside the container).

    cori$ shifter --module=cvmfs

You can also load this module in your batch script and you should also add the cvmfs filesystem license `-L cvmfs` which would allow us to pause your jobs from running if the cvmfs mount was unavailable

    #!/bin/bash
    #SBATCH --module=cvmfs
    #SBATCH -L cvmfs

