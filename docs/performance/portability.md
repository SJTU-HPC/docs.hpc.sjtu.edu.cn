# Portability Across DOE Office of Science HPC Facilities

As the HPC community prepares for exascale and the semiconductor industry approaches the end of Moore's Law in terms of transistor size, we have entered a 
period of time of increased diversity in computer architecture for High Performance Computing (HPC) with relatively new designs joining mature 
processor and memory 
technologies. These technologies include GPUs, Many-Core Processors, ARM, FPGAs, and ASICs, as well as new memory technologies in the form of High-Bandwidth 
Memory (HBM) often incorporated on the processor die as well as Non-Volatile memory (NVRAM) and Solid-State Disk (SSD) technology for accelerated IO. 

The DOE Office of Science operates three world-leading HPC facilities located at the Argonne Leadership Computing Facility (ALCF), the National Energy Research 
Scientific Computing Center (NERSC) at Lawrence Berkeley National Lab, and the Oak Ridge Leadership Computing Center (OLCF). These facilities field three of the most 
powerful supercomputers in world. These machines are used by scientists throughout the DOE Office of Science and the world for solving a 
number of important problems in domains including materials science and chemistry to nuclear, particle, and astrophysics. 

These facilities, with their latest systems, have begun the transition for DOE users to energy-efficient HPC architectures. The facilities are currently 
fielding systems with two-distinct "pre-exascale" architectures that we discuss in detail here. 


| System   | Titan    |
|----------|----------|
| Location | OLCF     |
| Architecture | CPU + GPU |
| Scale | 18,688 Nodes |


| System   | Cori    |
|----------|----------|
| Location | NERSC     |
| Architecture | Xeon-Phi |
| Scale | 9688 Nodes |
| Notes | SSD Burst-Buffer IO layer |

| System   | Theta    |
|----------|----------|
| Location | ALCF     |
| Architecture | Xeon-Phi |
| Scale | 3624 Nodes |

The two processor architectures deployed on these systems are the CPU+GPU hybrid architecture on Titan and the "self-hosted" Xeon-Phi processors 
(code named "Knights Landing"). These two architectures, while seemingly quite different at first, have a number of similarities that we believe 
represent general trends in exascale-like architectures:

* Increase parallelism (Cores, Threads, Warps/SMs/Blocks)
* Vectorization (AVX512 8 Wide Vector Units, 32 Wide Warps)
* Small Amount High-bandwidth Coupled with Large Amounts of Traditional DDR

While the details of the architectures are distinct, and vendor specific programming libraries/languages (CUDA, AVX512 Intrinsics etc.) exist to address 
specific architecture features; the commonalities are significant enough that a number of portable programming approaches are emerging for writing code that 
supports both architectures. 

This website is intended to be a living/growing documentation hub and guide for applications teams targeting systems at multiple DOE Office of Science 
facilities. In these pages, we 
discuss the differences between the systems, the software environment, and the job-submission process. We discuss how to define and measure performance 
portability and we provide recommendations based on case studies for the most promising performance-portable programming approaches. A 
[summary](http://performanceportability.org/perfport/summary/) of the high level findings and recommendations will be maintained.
