# LibSci

Cray LibSci is a collection of numerical routines tuned for
performance on Cray systems. Most users, on most codes, will find they
obtain better performance by using calls to Cray LibSci routines in
their applications instead of calls to public domain or user-written
versions.

Most LibSci components contain both single-processor and parallel
routines optimized specifically to make best use of Cray processors
and interconnect architectures. The general components of Cray LibSci
are:

*  BLAS (Basic Linear Algebra Subroutines)

*  CBLAS (C interface to the legacy BLAS)

*  BLACS (Basic Linear Algebra Communication Subprograms)

*  LAPACK (Linear Algebra routines)

*  ScaLAPACK (parallel Linear Algebra routines)

Two libraries unique to Cray are:

*  IRT (Iterative Refinement Toolkit) - a library of solvers and tools
   that provides solutions to linear systems using single-precision
   factorizations while preserving accuracy through mixed-precision
   iterative refinement.

*  CrayBLAS - a library of BLAS routines autotuned for Cray XC series
   systems through extensive optimization and runtime adaptation. For
   further information, see intro_blas3(3s).

!!! note
	Additional details are available in the `intro_libsci` man pages available on Cori and Edison with the command

	```shell
	man intro_libsci
	```
