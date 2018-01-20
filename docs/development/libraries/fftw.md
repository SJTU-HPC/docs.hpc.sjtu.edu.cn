# FFTW3

FFTW3 is a C subroutine library with Fortran interfaces for computing
complex-to-complex and real-to-complex/complex_to_real single and
multidimensional discrete Fourier transforms (DFTs). The library also
includes routines to compute discrete cosine and sine transforms
(DCTs/DSTs) on even and odd data, respectively.

To use FFTW 3.x, load the module cray-fftw or cray-fftw/<version>

In FFTW 3.x, the single- and double-precision routine names are unique,
and therefore both libraries automatically appear on the link line when
the fftw module is loaded, so that a user's program can call single- or
double-precision routines. The single- and double-precision libraries
are libfftw3f.a and libfftw3.a, respectively.

Additional details are available in the `intro_fftw3` man pages on Cori and Edison.
