# Burst Buffer

The NERSC Burst Buffer is based on Cray DataWarp that uses flash or
SSD (solid-state drive) technology to significantly increase the I/O
performance on Cori for all file sizes and all access patterns.

## Striping

Currently, the Burst Buffer granularity is 82GiB. If you request an
allocation smaller than this amount, your files will sit on one Burst
Buffer node. If you request more than this amount, then your files
will be striped over multiple Burst Buffer nodes. For example, if you
request 25GiB then your files all sit on the same Burst Buffer
server. This is important, because each Burst Buffer server has a
maximum possible bandwidth of roughly 6.5GB/s - so your aggregate
bandwidth is summed over the number of Burst Buffer servers. If other
people are accessing data on the same Burst Buffer server at the same
time, then you will share that bandwidth and will be unlikely to reach
the theoretical peak.

 * It is better to stripe your data over many Burst Buffer servers,
particularly if you have a large number of compute nodes trying to
access the data.

 * The number of Burst Buffer nodes used by an application should be
scaled up with the number of compute nodes, to keep the Burst Buffer
nodes busy but not over-subscribed. The exact ratio of compute to
Burst Buffer nodes will depend on the amount of I/O load produced by
the application.

## Use large transfer sizes

We have seen that using transfer sizes less than 512KiB results in
poor performance. In general, we recommend using as large a transfer
size as possible.

## Use more processes per Burst Buffer node

We have seen that the Burst Buffer cannot be kept busy with less than
4 processes writing to each Burst Buffer node - less than this will
not be able to achieve the peak potential performance of roughly 6.5
GB/s per node.
