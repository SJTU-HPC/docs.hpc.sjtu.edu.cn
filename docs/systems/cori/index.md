# Cori

Cori is a Cray XC40 with a peak performance of about 30 petaflops.
The system is named in honor of American
biochemist [Gerty Cori](https://en.wikipedia.org/wiki/Gerty_Cori), the
first American woman to win a Nobel Prize and the first woman to be
awarded the prize in Physiology or Medicine. Cori is comprised of
2,388 Intel Xeon "Haswell" processor nodes, 9,688 Intel Xeon Phi
"Knight's Landing" (KNL) nodes. The system also has a large Lustre
scratch file system and a first-of-its kind NVRAM "burst buffer"
storage device.

## Filesystems

* [Cori scratch](/filesystems/cori-scratch.md)
* [Burst Buffer](/filesystems/cori-burst-buffer.md)

[Filesystems at NERSC](/filesystems/index.md)

## Configuration

| Node type | # of cabinets | # of nodes |
|-----------|---------------|------------|
| haswell   | 14            | 2388       |
| KNL       | 54            | 9668       |

Each cabinet has 3 chassis; each chassis has 16 compute blades, each
compute blade has 4 nodes.

### Haswell nodes

* Each node has two sockets, each socket is populated with a
  16-core
  [Intel Xeon Processor E5-2698 v3](https://ark.intel.com/products/81060/Intel-Xeon-Processor-E5-2698-v3-40M-Cache-2_30-GHz).
* Each core supports 2 hyper-threads, and has 2 256-bit-wide vector
  units
* 36.8 Gflops/core (theoretical peak)
* 1.2 TFlops/node (theoretical peak)
* 2.81 PFlops total (theoretical peak)
* Each node has 128 GB DDR4 2133 MHz memory (four 16 GB DIMMs per
  socket)
* 298.5 TB total aggregate memory

### KNL nodes

* Each node is a single-socket
  [Intel Xeon Phi Processor 7250](http://ark.intel.com/products/94035/Intel-Xeon-Phi-Processor-7250-16GB-1_40-GHz-68-core) ("Knights Landing") processor with
  68 cores per node @ 1.4 GHz
* Each core has two 512-bit-wide vector processing units
* Each core has 4 hardware threads (272 threads total)
* 44.8 GFlops/core (theoretical peak)
* 3 TFlops/node (theoretical peak)
* 29.5 PFlops total (theoretical peak)
* Each node has 96 GB DDR4 2400 MHz memory, six 16 GB DIMMs (102 GiB/s
  peak bandwidth)
* Total aggregate memory (combined with MCDRAM) is 1.09 PB.
* Each node has 16 GB MCDRAM (multi-channel DRAM), > 460 GB/s peak
  bandwidth
* Each core has its own L1 caches, with 64 KB (32 KiB instruction
  cache, 32 KB data)
* Each tile (2 cores) shares a 1MB L2 cache

### Interconnect

Cray Aries with Dragonfly topology with >45 TB/s global peak bisection
bandwidth.
