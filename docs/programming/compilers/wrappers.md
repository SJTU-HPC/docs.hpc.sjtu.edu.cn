# Compiler Wrappers

NERSC provides compiler wrappers on Cori which combine the [native
compilers](native.md) (Intel, GNU, and Cray) with MPI and various other
libraries, to enable streamlined compilation of scientific applications.

## Cray Compiler Wrappers

Cray provides a convenient set of wrapper commands that should be used in
almost all cases for compiling and linking parallel programs. Invoking the
wrappers will automatically link codes with MPI libraries and other Cray system
software. All MPI and Cray system directories are also transparently imported.
In addition, the wrappers cross-compile for the appropriate compute node
architecture, based on which `craype-<arch>` module is loaded when the compiler
is invoked, where the possible values of `<arch>` are discussed below.

!!! note
	The intention is that programs are compiled on the login nodes and executed
	on the compute nodes. Because the compute nodes and login nodes have
	different hardware and software, executables cross-compiled for compute
	nodes may fail if run on login nodes. The wrappers mentioned above
	guarantee that codes compiled using the wrappers are prepared for running
	on the compute nodes.

!!! warning
	On Cori there are two types of compute nodes: Haswell and KNL. While
	applications cross-compiled for Haswell do run on KNL compute nodes, the
	converse is not true (applications compiled for KNL will fail if run on
	Haswell compute nodes). Additionally, even though a code compiled for
	Haswell will run on a KNL node, it will not be able to take advantage of
	the wide vector processing units available on KNL. Consequently, one should
	specifically target KNL nodes during compilation in order to achieve the
	highest possible code performance. Please see below for more information on
	how to compile for KNL compute nodes.

### Basic Example

The Cray compiler wrappers replace other compiler wrappers commonly found on
computer clusters, such as `mpif90`, `mpicc`, and `mpic++`. By default, the
Cray wrappers include MPI libraries and header files, as well as the many
scientific libraries included in Cray [LibSci](../libraries/libsci/index.md).

#### Fortran

```shell
ftn -o example.x example.f90
```

#### C

```shell
cc -o example.x example.c
```

#### C++

```shell
CC -o example.x example.cpp
```

!!! tip "Using compiler wrappers in `./configure`"
	When compiling an application which uses the standard series of `./configure`,
	`make`, and `make install`, often specifying the compiler wrappers in the
	appropriate environment variables is sufficient for a configure step to
	succeed, e.g.:
	```shell
	./configure CC=CC CXX=cc FC=ftn
	```

## Intel Compiler Wrappers

Although the Cray compiler wrappers `cc`, `CC`, and `ftn`, are the default (and
recommended) compiler wrappers on the Cori system, wrappers for
Intel MPI are provided as well via the the `impi` module.

The Intel MPI wrapper commands are `mpiicc`, `mpiicpc`, and `mpiifort`, which
are analogous to `cc`, `CC`, and `ftn` from the Cray wrappers, respetively. In
contrast to the Cray wrappers, the default link type for the Intel wrappers is
dynamic, not static.

!!! warning
	Although Intel MPI is available on the Cray systems at NERSC, it is not
	tuned for high performance on the high speed network on these systems.
	Consequently, it is possible, even likely, that MPI application performance
	will be lower if compiled with Intel MPI than with Cray MPI.

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

!!! warning
	Unlike the Cray compiler wrappers, the Intel compiler wrappers do not
	automatically include and link to scientific libraries such as LibSci.
	These libraries must be included and linked manually if using the Intel MPI
	wrappers.

### Compiling

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

### Running

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
