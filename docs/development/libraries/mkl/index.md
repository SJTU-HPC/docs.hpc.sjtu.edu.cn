# MKL

The [Intel Math Kernel Library](https://software.intel.com/en-us/mkl)
(Intel MKL) contains highly optimized, extensively threaded math
routines for science, engineering, and financial applications. Core
math functions include BLAS, LAPACK, ScaLAPACK, Sparse Solvers, Fast
Fourier Transforms, Vector Math, and more.

MKL is available on NERSC computer platforms where the Intel compilers
are available. For instances, on Cori and Edison. If you use intel
compilers to compile your code, you can conveniently use the Intel
compiler flag, -mkl, to link your code to MKL. If you use other
compilers, for example, GNU or Cray Compilers, we recommend you to use
the Intel MKL Link Line Advisor to get the advice about the compiler
flags and the link lines to use MKL.

In this page we provide the instructions about how to use MKL on Cori
and Edison. Please refer to the Intel MKL Link Line Advisor page for
any use cases that are not covered in this page.

## Usage

On Cori and Edison, the Intel programming environment is the
default. So if you compile under the default programming environment,
you just use the "-mkl" flag with the compiler wrappers under the
Intel programming environment. The default is to use the threaded
libraries ("-mkl" is equivalent to "-mkl=parallel").  If sequential
libraries are needed, use the "-mkl=sequential" flag instead. If
ScaLapack routines are needed, use the -mkl=cluster option. Note, the
-mkl=cluster flag links to the sequential routines of the libraries,
if the ScaLapack with threaded libraries are needed, you need to
provide the correct combinations of the libraries to the link
line. Please refer to
the
[Intel MKL Link Line Advisor page](https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor).

```shell
ftn my_code.f -mkl
cc test.c -mkl
```
