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

* [Examples](/jobs/examples/index.md)
