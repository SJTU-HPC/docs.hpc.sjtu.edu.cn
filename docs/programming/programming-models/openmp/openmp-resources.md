# OpenMP Resources

## What is OpenMP?


OpenMP is an industry standard API of C/C++ and Fortran for shared memory
parallel programming. The [OpenMP](http://openmp.org/) Architecture Review
Board (ARB) consists of major compiler vendors and many research institutions.
Common architectures include shared memory architecture (multiple CPUs sharing
global memory with uniform memory access [UMA] and a typical shared memory
programming model of OpenMP or pthreads), distributed memory architecture (each
CPU has its own memory with non-uniform memory access [NUMA] and the typical
message passing programming model of MPI), and hybrid architecture (UMA within
one node or socket, NUMA across nodes or sockets, and the typical hybrid
programming model of hybrid MPI/OpenMP). The current architecture trend needs a
hybrid programming model with three levels of parallelism: MPI between nodes or
sockets, shared memory (such as OpenMP) on the nodes/sockets, and increased
vectorization for lower-level loop structures.

OpenMP has three components: compiler directives and clauses, runtime
libraries, and environment variables. The compiler directives are only
interpreted when the OpenMP compiler option is turned on. OpenMP uses the "fork
and join" execution model: the master thread forks new threads at the beginning
of parallel regions, multiple threads share work in parallel; and threads join
at the end of parallel regions.

<img style="float: center;" alt="OpenMP fork and join model" src="../../../../img/OpenMPforkjoin.png">

In OpenMP, all threads have access to the same shared global memory. Each
thread has access to its own private local memory. Threads synchronize
implicitly by reading and writing shared variables. No explicit communication
is needed between threads.

<img style="float: center;" alt="OpenMP memory model" src="../../../../img/OpenMPmemorymodel.png">

Major features in OpenMP 3.1 include:

* Thread creation with shared and private memory
* Loop parallelism and work sharing constructs
* Dynamic work scheduling
* Explicit and implicit synchronizations
* Simple reductions
* Nested parallelism
* OpenMP tasking

New features in OpenMP 4.0 (released in July 2013) include:

* Device constructs for accelerators
* SIMD constructs for vectorization
* Task groups and dependencies
* Thread affinity control
* User defined reductions
* Cancellation construct
* Initial support for Fortran 2003
* `OMP_DISPLAY_ENV` for all internal variables

New features in OpenMP 4.5 (released in November 2015) include:

* Significantly improved support for devices
* Support for doacross loops
* New taskloop construct
* Reductions for C/C++ arrays
* New hint mechanisms
* Thread affinity support
* Improved support for Fortran 2003
* SIMD extensions

## OpenMP 4.0/4.5 Support in Compilers

* **GNU compiler**
    * From gcc/4.9.0 for C/C++ and OpenMP 4.0
    * From gcc/4.9.1 for Fortran with OpenMP 4.0
    * From gcc/6.0 and most OpenMP 4.5 features
    * From gcc/6.1 and full OpenMP 4.5 for C/C++ (not Fortran)
* **Intel compiler**
    * From intel/15.0 with most OpenMP 4.0 features
    * From intel/16.0 with full OpenMP 4.0
    * From intel/16.0 Update 2 and some OpenMP4.5 SIMD features
* **Cray compiler**
    * From cce/8.4.0 with full OpenMP 4.0

For more information on compiler support for OpenMP, click
[here](http://openmp.org/wp/openmp-compilers/).

More details of using OpenMP can be found in the OpenMP training and resources
sections below.

## Relevant NERSC Training Sessions on OpenMP

* [Advanced OpenMP and CESM Case Study](https://www.nersc.gov/assets/Uploads/Advanced-OpenMP-CESM-NUG2016-He.pdf)
  by Helen He, March 2016.
* [Nested OpenMP](http://www.nersc.gov/assets/Uploads/Nested-OpenMP-NUG-20151008.pdf)
  by Helen He, October 2015.
* [Tutorial: Getting up to Speed on OpenMP 4.0](https://www.youtube.com/playlist?list=PL20S5EeApOSshYrRnuY3S3BUw4IY3LYTt)
* [OpenMP Basics and MPI/OpenMP Scaling](http://www.nersc.gov/assets/pubs_presos/hybridMPIOpenMP20150323.pdf)
  Helen He. LBNL Computational Sciences Postdocs Training, March 2015.
* Intel OpenMP Training at NERSC ([part 1](http://www.nersc.gov/assets/For-Users/N8/0IntelThreadingIntroduction.pdf),
  [part 2](http://www.nersc.gov/assets/For-Users/N8/1IntelThreadingMIC-OpenMP.pdf),
  [part 3](http://www.nersc.gov/assets/For-Users/N8/2IntelMultiLevelOpenMP.pdf),
  [part 4](http://www.nersc.gov/assets/For-Users/N8/3IntelThreadingEMGeoPARSEC.pdf))
  by Jeongnim Kim, Intel.  March 2015.
* [Explore MPI/OpenMP Scaling on NERSC Systems](http://www.nersc.gov/assets/Training-Materials/NERSC-HybridMpiOpenmpOct2014.pdf)
  by Helen He, NERSC Training, October 2014.
* [OpenMP and Vectorization Training](http://www.nersc.gov/assets/Training-Materials/NERSC-VectorTrainingOct2014.pdf)
  by Jack Deslippe, Helen He, Harvey Wasserman, Woo-Sun Yang, October 2014.
* [Hybrid MPI/OpenMP Programming](http://www.nersc.gov/assets/Uploads/NUG2013hybridMPIOpenMP2.pdf)
  by Helen He,  NERSC User Group Training, February 2013.
* [Introduction to OpenMP](http://www.nersc.gov/assets/Uploads/IntroToOpenMP.pdf)
  by Matt Cordery, NERSC User Group Training, February 2013.

## Other Useful OpenMP Resources and Tutorials

* [Official OpenMP website](http://www.openmp.org/): OpenMP standards, API
  specifications, tutorials, forums, and a lot more other information and
  resources.
* [OpenMP Affinity on KNL](http://www.nersc.gov/assets/26-TACC-milfeld-OpenMP-Affinity-on-KNL.pdf)
  by Kent Milfield at IXPUG-ISC16 Workshop, June 2016.
* [ANL Training Program on Exascale Computing, August 2015](https://www.youtube.com/playlist?list=PLGj2a3KTwhRZR9yvRG2f3F7svgYYs2GSa)
    * [A "Hands On" Introduction to OpenMP: Part 1](https://www.youtube.com/watch?v=4MiXzs0d1eE&list=PLGj2a3KTwhRZR9yvRG2f3F7svgYYs2GSa&index=43)
      by Bronis de Supinski, LLNL; Tim Mattson, Intel.
    * [A "Hands On" Introduction to OpenMP: Part 2](https://www.youtube.com/watch?v=CzzFLj9P-hw&list=PLGj2a3KTwhRZR9yvRG2f3F7svgYYs2GSa&index=44)
      by Bronis de Supinski, LLNL; Tim Mattson, Intel.
    * [A "Hands On" Introduction to OpenMP: Part 3](https://www.youtube.com/watch?v=cAJ-JD8eef4&list=PLGj2a3KTwhRZR9yvRG2f3F7svgYYs2GSa&index=45)
      by Bronis de Supinski, LLNL; Tim Mattson, Intel.
 * UC Berkeley ParLab Boot Camp, 2014
    * [Hands-on](http://bebop.cs.berkeley.edu/bootcamp2014/index.html)
 * Tim Mattson's (Intel) "[Introduction to OpenMP](https://www.youtube.com/playlist?list=PLLX-Q6B8xqZ8n8bwjGdzBJ25X2utwnoEG)"
   (2013) on Youtube; 27 video segments, 4 hours total. Slides
   [here](http://openmp.org/mp-documents/Intro_To_OpenMP_Mattson.pdf) and
   exercises [here](http://openmp.org/mp-documents/Mattson_OMP_exercises.zip).
 * [LLNL OpenMP Tutorial](https://computing.llnl.gov/tutorials/openMP/)
   by Blaise Barney, LLNL.
 * UC Berkeley ParLab Boot Camp, 2013
    * [Shared Memory Programming with OpenMP - Basics](http://parlab.eecs.berkeley.edu/sites/all/parlab/files/openmp_basics_0.pdf)
      by Tim Mattson, Intel. Video
      [here](http://www.youtube.com/watch?v=fn2VAUSw6cI&list=PLYTiwx6hV33s0-gysyoIjTiovkIuzoyMg&index=3).
    * [More about OpenMP - New Features](http://parlab.eecs.berkeley.edu/sites/all/parlab/files/openmp_newer_features_0.pdf)
      by Tim Mattson, Intel. Video
      [here](https://www.youtube.com/watch?v=fn2VAUSw6cI&list=PLYTiwx6hV33s0-gysyoIjTiovkIuzoyMg&index=3#t=85m20s).

## Tools for OpenMP

Tools for tuning OpenMP codes to get better performance include:

* Use Intel Advisor for Threading Design and Vectorization
* Use Intel Inspector to Detect Threading and Memory Issues
* Use Intel VTune for Performance Tuning
* Use Cray Reveal to Insert OpenMP Directives

There are several [tools](openmp-tools.md) available at NERSC that are useful
for tuning OpenMP codes.

The [Performance and Debugging Tools](../../performance-debugging-tools/index.md)
page also shows other tools that can be useful for OpenMP codes.
