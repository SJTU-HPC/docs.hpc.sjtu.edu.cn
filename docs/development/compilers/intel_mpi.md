# Intel Compiler Wrappers

## Introduction

Although the [Cray compiler wrappers](index.md) `cc`, `CC`, and `ftn`, are the
default (and recommended) compiler wrappers on the Cori and Edison systems,
wrappers for Intel MPI are provided as well via the the `impi` module.

The Intel MPI wrapper commands are `mpiicc`, `mpiicpc`, and `mpiifort`, which
are analogous to `cc`, `CC`, and `ftn` from the Cray wrappers, respetively. In
contrast to the Cray wrappers, the default link type for the Intel wrappers is
dynamic, not static.

!!! warning
	Although Intel MPI is available on the Cray systems at NERSC, it is not
	tuned for high performance on the high speed network on these systems.
	Consequently, it is possible that MPI application performance will be lower
	if compiled with Intel MPI than with Cray MPI.

!!! note
	If one chooses to use the Intel MPI compiler wrappers, they are compatible
	only with the Intel compilers `icc`, `icpc`, and `ifort`. They are
	incompatible with the Cray and GCC compilers.

!!! note
	While the Cray compiler wrappers cross-compile source code for the appropriate
	architecture based on the `craype-<arch>` modules (e.g., `craype-haswell` for
	Haswell code and `craype-mic-knl` for KNL code), the Intel wrappers do not. The
	user must apply the appropriate architecture flags to the wrappers manually,
	e.g., adding the `-xMIC-AVX512` flag to compile for KNL.

## Compiling

The Intel compiler wrappers function similarly to the Cray wrappers `cc`, `CC`,
and `ftn`. However a few extra steps are required. To compile with the Intel
MPI wrappers, one must first load the `impi` module. Then, whereas the Cray
compiler may look like

```shell
user@nersc> module swap craype-haswell craype-mic-knl
user@nersc> ftn -o example.x example.f90 # compile a code for KNL
```

the Intel wrapper would look like:

```shell
user@nersc> module load impi
user@nersc> mpiifort -xMIC-AVX512 -o example.x example.f90 # Intel wrappers ignore craype module
```

## Running

To run an application compiled with Intel MPI, one must set a few extra
environment variables:

```shell
export I_MPI_FABRICS=ofi
export I_MPI_OFI_PROVIDER=gni
export I_MPI_OFI_LIBRARY=/global/common/cori/software/libfabric/1.6.1/gnu/lib/libfabric.so
export I_MPI_PMI_LIBRARY=/usr/lib64/slurmpmi/libpmi.so
```

After setting these, one may issue the same `srun` commands as typical for an
application compiled with the Cray wrappers (see the [example batch
scripts](../../jobs/examples/index.md) page).
