# Future architecture readiness

This page contains recommendations for application developers to "hit
the ground running" with upcoming system architectures such
as [Perlmutter](https://www.nersc.gov/systems/perlmutter/).

Testing of performance on relevant hardware and compatability with the
software environment are both important.

## Proxy hardware platforms

Perlmutter will feature NVIDIA GPUs and AMD cpus.

### Cloud providers

* [AWS](https://aws.amazon.com/ec2/instance-types/p3/)
* [Google](https://cloud.google.com/compute/docs/gpus/)
* [Azure](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/sizes-gpu)

### HPC systems

Summit and Sierra feature NVIDIA GPUs and IBM CPUs. Piz Daint features
NVIDIA GPUs and Intel CPUs, but only has 1 GPU per node.

* [Summit](https://www.olcf.ornl.gov/summit/)
* [Sierra](https://computation.llnl.gov/computers/sierra)
* [Piz Daint](https://www.cscs.ch/computers/dismissed/piz-daint-piz-dora/)

### CPU

Current generation AMD cpus are a good place to start.

* [AWS](https://aws.amazon.com/ec2/amd/)

## Software environment

Compilers and programming models play key roles in the software
environment.

* `module load pgi` on Cori.
* [PGI Compilers on AWS](https://www.pgroup.com/blogs/posts/pgi-ami-on-aws.htm)

## Programming Models

The choice of programming model depends on multiple factors including:
number of performance critical kernels, source language, existing
programming model, and portability of algorithm. A 20K line C++ code
with 5 main kernels will have different priorities and choices vs a
10M line Fortran code.

### Fortran

* [OpenMP](https://www.openmp.org)
* [CUDA Fortran](https://developer.nvidia.com/cuda-fortran)
* [OpenACC](https://www.openacc.org)

### C

* [CUDA](https://developer.nvidia.com/cuda-zone)
* [OpenMP](https://www.openmp.org)
* [OpenACC](https://www.openacc.org)

### C++

* [CUDA](https://developer.nvidia.com/cuda-zone)
* [Kokkos](https://github.com/kokkos/kokkos)
* [RAJA](https://github.com/LLNL/RAJA)
* [Thrust](https://thrust.github.io)
* [CUDA](https://developer.nvidia.com/cuda-zone)
* [OpenMP](https://www.openmp.org)
* [OpenACC](https://www.openacc.org)

## Algorithms

The ability for applications to achieve both portability and high
performance across computer architectures remains an open
challenge.

However there are some general trends in current and emerging HPC
hardware: increased thread parallelism; wider vector units; and deep,
complex, memory hierarchies.

In some cases a [performance portable](portability.md) algorithm can
realized by considering generic "wide vectors" which could map to
either GPU SIMT threads or CPU SIMD lanes.

# References and Events

* P3HPC Workshop at SC
* [2019 DOE Performance Portability](https://doep3meeting2019.lbl.gov)
* https://performanceportability.org
