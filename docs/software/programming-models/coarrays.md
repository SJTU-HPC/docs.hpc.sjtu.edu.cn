# Fortran Coarrays

The Fortran 2008 standard introduced "coarrays," which provide a new model for
parallel programming integrated directly into the Fortran language. Coarrays
allow an application developer can write parallel programs using only a Fortran
compiler - no external APIs such as MPI or OpenMP are necessary. The Fortran
2018 standard (formerly called "Fortran 2015") extends coarrays further to
include "teams" and other coarray collective procedures. Prior to becoming a
part of the Fortran standard, coarrays were widely implemented in a language
extension called "Co-Array Fortran" ("CAF"). More information about the
motivation, design, and functionality of coarrays in Fortran 2008 can be found
in [this paper](https://wg5-fortran.org/N1801-N1850/N1824.pdf).

## Overview

Corrays were designed with both performance and ease of use in mind. For
example, consider a finite-difference code which expresses parallelism by
decomposing a domain among MPI processes. The MPI processes must exchange
boundary condition information with each other in order to compute the
derivative of a field on the decomposed domain. Such boundary exchange may take
the following form:

```Fortran
! send 2 ghost zones on left and right sides to my left and right neighbors

bc_send_left = field_data(-1:0)
bc_send_right = field_data(sz+1:sz+2)

! send BCs to my left neighbor
call MPI_Isend(bc_send_left, 2, MPI_DOUBLE_PRECISION, proc_left, tag1, &
               MPI_COMM_WORLD, send1_rq, ierr)
! send BCs to my right neighbor
call MPI_Isend(bc_send_right, 2, MPI_DOUBLE_PRECISION, proc_right, tag2, &
               MPI_COMM_WORLD, send2_rq, ierr)
! receive BCs from my left neighbor
call MPI_Irecv(bc_recv_left, 2, MPI_DOUBLE_PRECISION, proc_left, tag3, &
               MPI_COMM_WORLD, recv1_rq, ierr)
! receive BCs from my right neighbor
call MPI_Irecv(bc_recv_right, 2, MPI_DOUBLE_PRECISION, proc_right, tag4, &
               MPI_COMM_WORLD, recv2_rq, ierr)

field_data(sz+1:sz+2) = bc_recv_right
field_data(-1:0) = bc_recv_left
```

A similar procedure using Fortran coarrays may look like:

```Fortran
bc_left = field_data(-1:0)
bc_right = field_data(sz+1:sz+2)

field_data(sz+1:sz+2)[this_image()-1] = bc_left  ! write BCs on my left neighbor
field_data(  -1:   0)[this_image()+1] = bc_right ! write BCs on my right neighbor
```

As illustrated in these examples, the PGAS nature of coarrays allows the
application developer to break the symmetry inherent to two-sided message
passing in MPI, in which every "send" must match with a "receive". By contrast,
with coarrays one can write data directly from one coarray image onto another,
or, conversely, one can retrieve data directly from another coarray image;
these procedures are analogous to the `MPI_Put` and `MPI_Get` procedures in
one-sided MPI communication. This asymmetry often leads to fewer lines of code,
as shown above, as well as higher performance than using explicit two-sided
message passing.

In addition to the ease of use, due to the PGAS-style semantics of coarrays in
Fortran, applications can achieve high performance at extremely high
concurrency; tests at NERSC have shown that coarray program scale efficiently
to >100,000 cores.

## Compiler support on NERSC systems

Coarrays are a large addition to the Fortran language, and consequently
compiler support of coarray codes varies. Below is a description of coarray
support on each compiler supported on the NERSC systems.

### Cray

The Cray Fortran compiler supports coarray code compilation by default. No
extra flags are required, e.g.,

```bash
module load PrgEnv-cray
ftn -o my_coarray_code.ex my_coarray_code.f90
```

### Intel

Intel supports coarray compliation in two different forms: shared memory and
distributed memory. By default, neither mode is active, and the compiler will
throw an error if it encounters coarray syntax.

To enable coarrays for shared memory (parallelism within a single node, similar
to OpenMP), one should add the compiler flag `-coarray` or `-coarray=shared`:

```bash
module load PrgEnv-intel
ftn -coarray=shared my_coarray_code.ex my_coarray_code.f90
```

To enable coarrays for distributed memory (parallelism across multiple nodes,
similar to MPI), one should add the flag `-coarray=distributed`:

```bash
module load PrgEnv-intel
ftn -coarray=distributed my_coarray_code.ex my_coarray_code.f90
```

!!!warning
    Intel's implementations of both shared and distributed memory coarrays is
    currently incompatible with Cray MPI, and consequently this mode does not
    work on Cori or Edison.

### GCC

GCC supports single-image coarray compliation, and also distributed-memory
coarrays via [OpenCoarrays](http://www.opencoarrays.org/). By default, coarray
support is disabled, and GCC will throw an error if it encounters coarray
syntax.

GCC supports single-image coarrays via the `-fcoarray=single` compiler flag:
```bash
module load PrgEnv-gnu
ftn -fcoarray=single my_coarray_code.ex my_coarray_code.f90
```

For multi-image support, one must link the coarray code to the OpenCoarrays
library via the `-fcoarray=lib` flag, along with the appropriate linker flags
to the OpenCoarrays libraries, e.g.,
```bash
module load PrgEnv-gnu
ftn -fcoarray=lib my_coarray_code.ex my_coarray_code.f90 \
    -L/path/to/OpenCoarrays/installation/lib -lcaf_mpi
```
where the above example assumes that OpenCoarrays has been compiled using MPI.
OpenCoarrays also supports a GASNet backend.

NERSC provides OpenCoarrays as a module (called `opencoarrays`), built on top
of Cray MPI. One can link a GCC-compiled coarray code to this module via the
`OPENCOARRAYS_LIB` environment variable:
```bash
module load PrgEnv-gnu
ftn -fcoarray=lib my_coarray_code.ex my_coarray_code.f90 \
    -L${OPENCOARRAYS_LIB} -lcaf_mpi
```
