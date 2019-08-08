## I/O Resources at NERSC
NERSC provides a range of online resources to assist users developing, deploying, understanding, 
and tuning their scientific I/O workloads, supplemented by direct support from the NERSC Consultants 
and the [Data Analytics Group](https://www.nersc.gov/about/nersc-staff/data-analytics-services/). Here, we provide a consolidated summary of these resources, 
along with pointers to relevant online documentation.

## Libraries and Tools available at NERSC
NERSC provides a number of I/O middleware libraries, as well as tools for profiling I/O performed 
by your jobs and monitoring system status. These resources include:

* High-level [I/O libraries](./library) available at NERSC, including the popular HDF5 and NetCDF libraries
* Tools for monitoring I/O activity:
 * The [Darshan](../../programming/performance-debugging-tools/darshan.md) job-level I/O profiling tool may be used to examine the I/O activity of your own jobs (available on Cori).
 * Real-time aggregate Cori scratch [I/O activity](https://my.nersc.gov/completedjobs.php) can give an estimate of overall aggregate-bandwidth utilization for Cori's scratch file system (click on any of your jobs, then click Lustre LMT).
* [Filesystem status](http://my.nersc.gov/filesystems-cs.php) reflected by timing of representative tasks (file creation, directory listing) can be found on MyNERSC.

Users should keep in mind the entire HPC I/O stack in order to design the best I/O strategy for the long term. 

|I/O Layer|I/O Libraries|
|:--------|:-------------|
|Productive Interface|H5py, Python, Spark, Tensorflow, PyTorch|
|High-Level I/O Library|HDF5, NetCDF, PnetCDF, Root|
|I/O Middleware|MPIIO, POSIX|
|Parallel File System|Lustre, Datawarp, GPFS, HPSS|

Please refer to the resources below which present more detailed introductions to some of these topics in tutorial form.

## Best Practices for Scientific I/O
While there is clearly a wide range of I/O workloads associated with the many scientific applications
deployed at NERSC, there are a number of general guidelines for achieving good performance
when accessing our filesystems from parallel codes. Some of the most important guidelines include:

* Use filesystems for their intended use-case; for example, don't use your home directory for production
 I/O (more details on intended use case may be found on our page providing general background on NERSC filesystems).
* Know what fraction of your wall-clock time is spent in I/O; for example, with estimates provided by [Darshan](../../programming/performance-debugging-tools/darshan.md),
profiling of critical I/O routines (such as with [Craypat](../../programming/performance-debugging-tools/craypat)'s trace groups), or explicit timing / instrumentation.
* When algorithmically possible:
    * Avoid workflows that produce large numbers of small files (e.g. a "file-per-process" access model at high levels of concurrency).
    * Avoid random-access I/O workloads in favor of contiguous access.
    * Prefer I/O workloads that perform large transfers that are similar in size or larger than,
and are also aligned with, the underlying filesystem storage granularity (e.g.
blocksize on GPFS-based filesystems, stripe width on Lustre-based filesystems).
    * Use [high-level libraries](./library) for data management and parallel I/O operations (as these will often
apply optimizations in line with the above suggestions, such as MPI-IO collective
buffering to improve aggregate transfer size, alignment, and contiguous access).

With these suggestions in mind, there are also filesystem-specific tuning parameters which may be used
to enable high-performance I/O. These can affect both the layout of your files on the underlying
filesystem (as is the case in Lustre), as well as the manner is which your I/O operations
will be routed to the storage system (as in the case of GPFS over DVS on the Crays).
These can broadly be classified by filesystem type:

* [Lustre filesystem tuning](./lustre) for Cori /scratch
* [Burst Buffer tuning](./bb) for using the Burst Buffer on Cori
* [Optimizing IO on Cori KNL](./knl) 

## Tutorials, Support, and Resource Allocation

Here, we list additional support resources for NERSC users, as well as pointers to previous
 and ongoing research projects associated with NERSC staff and LBL researchers to support high-performance scientific I/O.

### Online tutorials at NERSC
+ A detailed [Introduction to Scientific I/O](https://www.nersc.gov/users/training/online-tutorials/introduction-to-scientific-i-o/?show_all=1), discussing access patterns, Lustre, and examples using the (parallel) HDF5 library
+ A brief overview of [I/O Formats](./library) at in use NERSC (focused on HDF5 and NetCDF)
+ An introduction to Cori's Burst Buffer ([video](http://www.nersc.gov/assets/Uploads/DataDay-BurstBuffer-Tutorial.mp4), [slides](http://www.nersc.gov/assets/Uploads/DataDay-BurstBuffer.pdf))

### User support at NERSC
+ [Consulting services](../../help) provided by the NERSC User Services Group
### Resource requests
+ Quota increases (space or inodes) for NGF project and global scratch, as well as the Lustre local scratch filesystems, may be submitted [here](https://www.nersc.gov/users/storage-and-file-systems/file-systems/data-storage-quota-increase-request/)
+ New project directory requests may be submitted in NIM by PIs and PI proxies
### Previous and ongoing I/O research projects contributed to by NERSC and LBL researchers
+ The [ExaHDF5](https://sdm.lbl.gov/exahdf5/) group is working to develop next-generation I/O libraries and middleware to support scientific I/O (focused in particular on the HDF5 data format)



