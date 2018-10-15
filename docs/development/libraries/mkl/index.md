# MKL

The [Intel Math Kernel Library](https://software.intel.com/en-us/mkl)
(Intel MKL) contains highly optimized, extensively threaded math
routines for science, engineering, and financial applications. Core
math functions include BLAS, LAPACK, ScaLAPACK, Sparse Solvers, Fast
Fourier Transforms, Vector Math, and more.

## Usage

### Intel compilers

When using the Intel compilers, MKL can be conveniently
used by adding the `-mkl` flag to the compiler line.

```bash
nersc$ ftn -mkl	test.f90
nersc$ cc -mkl test.c
```

If you need the sequential MKL routines only then use

```bash
nersc$ ftn -mkl="sequential" test.f90
nersc$ cc -mkl="sequential" test.c
```

If ScaLapack routines are needed then

```bash
nersc$ ftn -mkl="cluster" test.f90
nersc$ cc -mkl="cluster" test.c
```

### other compilers

Use the
[Intel MKL Link Line Advisor](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor) to
determine the appropriate link lines.
