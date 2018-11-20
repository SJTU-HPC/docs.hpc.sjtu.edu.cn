# Native compilers on NERSC systems

Cori and Edison provide three compiler suites:

  * Intel
  * GNU
  * Cray

Each suite provides compilers for C, C++, and Fortran. Each compiler has
different characteristics - different compilers generate faster code under
different circumstances. All three compilers provide support for OpenMP.

All compilers on Cori (Intel, Cray, and GNU), are provided via three
"programming environments" that are accessed via the `module` utility. Each
programming environment contains the full set of compatible compilers and
libraries. To change from one compiler to the other you change the programming
environment via the `module swap` command.

Additionally, NERSC provides the LLVM compilers `clang`, `clang++`, and `flang`
for C, C++, and Fortran, respectively. These are not supported by Cray and
therefore are not compatible with all of the same software and libraries that
the Cray-provided compiler suites are, but are nevertheless useful for users
who require an open-source LLVM-based compiler toolchain.

## Intel

The Intel compiler suite is available via the `PrgEnv-intel` module, and is loaded by
default on Cori and Edison. The native compilers in this suite are:

  * C: `icc`
  * C++: `icpc`
  * Fortran: `ifort`

Full documentation of the Intel compilers is provided
[here](https://software.intel.com/en-us/intel-compilers). Additionally,
compiler documentation is provided through `man` pages (e.g., `man icpc`) and
through the `-help` flag to each compiler (e.g., `ifort -help`).

## GNU

The GCC compiler suite is available via the `PrgEnv-gnu` module. The native
compilers in this suite are:

 * C: `gcc`
 * C++: `g++`
 * Fortran: `gfortran`

Full documentation of the GCC compilers is provided
[here](https://gcc.gnu.org/onlinedocs). Additionally, compiler documentation is
provided through `man` pages (e.g., `man g++`) and through the `--help` flag to
each compiler (e.g., `gfortran --help`).

## Cray

The Cray compiler suite is available via the `PrgEnv-cray` module. The native
compilers in this suite are:

 * C: `cc`
 * C++: `CC`
 * Fortran: `ftn`

 Full documentation of the Cray compilers is provided
 [here](https://pubs.cray.com/content/S-2179/8.7/cray-c-and-c++-reference-manual/the-cray-compiling-environment)
 for the C/C++ compilers, and
 [here](https://pubs.cray.com/content/S-3901/8.7/cray-fortran-reference-manual/fortran-compiler-introduction)
 for the Fortran compiler. Additionally, compiler documentation is provided
 through `man` pages (e.g., `man CC`).


## LLVM

The LLVM core libraries along with the compilers are available only on Cori. It
is compiled against the GCC compiler suite and thus cannot be used with the
Intel or Cray programming environments.

In order to enable clang compiler, first make sure to load the gnu programming
environment

```Shell
module load PrgEnv-gnu
module load gcc
module load llvm/<version>
```

where `module avail llvm` displays which versions are currently installed.

The module `llvm/5.0.0-gnu-flang` contains the LLVM-based Fortran compiler
`flang`. However, this compiler is highly experimental and should not be used
for production applications. Furthermore, it does not find the standard headers
and Fortran modules by default. Therefore, those need to be added manually to
the compilation flags using `-I`. Please use `module show llvm/5.0.0-gnu-flang`
to find the corresponding include paths for these headers and modules.

For more information, see [LLVM](https://llvm.org/),
[Clang](https://clang.llvm.org/), and
[Flang](https://github.com/flang-compiler/flang/wiki) websites.

## Common compiler options

Below is a table documenting common flags for each of the three compilers.

|                      | Intel  | GCC  | Cray  | comment |
|----------------------|--------|------|-------|---------|
| Overall optimization | `-O<n>`    | `-O<n>`        | `-O<n>`    | Replace `<n>` with `1`, `2`, `3`, etc.                    |
| Enable OpenMP        | `-qopenmp` | `-fopenmp`     | `-h omp`   | OpenMP enabled by default in Cray.                        |
| Free-form Fortran    | `-free`    | `-ffree-form`  | `-f free`  | Also determined by file suffix (`.f`, `.F`, `.f90`, etc.) |
| Fixed-form Fortran   | `-fixed`   | `-ffixed-form` | `-f fixed` | Also determined by file suffix (`.f`, `.F`, `.f90`, etc.) |
| Debug symbols        | `-g`       | `-g`           | N/A        | Debug symbols enabled by default in Cray.                 |
