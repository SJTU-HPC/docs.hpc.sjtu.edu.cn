Cori is a Cray XC40 with a peak performance of about 30 petaflops.
The system is named in honor of American biochemist Gerty Cori, the
first American woman to win a Nobel Prize and the first woman to be
awarded the prize in Physiology or Medicine. Cori is comprised of
2,388 Intel Xeon "Haswell" processor nodes, 9,688 Intel Xeon Phi
"Knight's Landing" nodes.

## Filesystems

### Cori scratch

Cori has one scratch file system named `/global/cscratch1` with 30 PB
disk space and >700 GB/sec IO bandwidth. Cori scratch is a Lustre
filesystem designed for high performance temporary storage of large
files. It contains 10000+ disks and 248 I/O servers (OSSs/OSTs).

* [Policy](data/policy)

### Burst Buffer

The 1.8 PB NERSC Burst Buffer is based on
Cray [DataWarp](http://www.cray.com/products/storage/datawarp) that
uses flash or SSD (solid-state drive) technology to significantly
increase the I/O performance on Cori for all file sizes and all access
patterns that sits within the High Speed Network (HSN) on
Cori. Accessible only from compute nodes, the Burst Buffer provides
per-job (or short-term) storage for I/O intensive codes.

The peak bandwidth performance is over 1.7 TB/s wtih each Burst Buffer
node contributing up to 6.5 GB/s. The number of Burst Buffer nodes
depends on the granularity and size of the Burst Buffer
allocation. Performance is also dependent on access pattern, transfer
size and access method (e.g. MPI I/O, shared files).

* [Examples](jobs/examples)
