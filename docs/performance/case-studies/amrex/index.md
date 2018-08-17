# AMReX

## Background

AMReX is a publicly available software framework for building massively
parallel block-structured AMR applications. It combines elements of both the
BoxLib and Chombo AMR frameworks. Key features of AMReX include

- Support for block-structured AMR with optional subcycling in time
- Support for cell-centered, face-centered and node-centered data
- Support for hyperbolic, parabolic, and elliptic solves on hierarchical grid
  structure
- Support for hybrid parallelism model with MPI and OpenMP
- Basis of mature applications in combustion, astrophysics, cosmology, and
  porous media
- Demonstrated scaling to over 200,000 processors
- Source code [freely available](https://amrex-codes.github.io/amrex/index.html)

We have approached the problem of optimization for manycore architectures from
several different perspectives. Listed below are a few such efforts.

## Loop tiling with OpenMP

In most AMReX codes, the majority of the computational expense lies in either
"vertical" (point-wise) evaluations, e.g.,
```
do k = lo(3), hi(3)
  do j = lo(2), hi(2)
    do i = lo(1), hi(1)
      phi_new(i, j, k) = phi_old(i, j, k) + &
                         dt * (flux_x(i, j, k) + flux_y(i, j, k) + flux_z(i, j, k))
    end do
  end do
end do
```

or "horizontal" (stencil) evaluations:

```
do k = lo(3), hi(3)
  do j = lo(2), hi(2)
    do i = lo(1), hi(1)
      val(i, j, k) = (p_old(i-1, j, k) - p_old(i+1, j, k)) + &
                     (p_old(i, j-1, k) - p_old(i, j+1, k)) + &
                     (p_old(i, j, k-1) - p_old(i, j, k+1))
    end do
  end do
end do
```

where `lo` and `hi` represent the lower and upper coordinates of the
computational grid. To parallelize this calculation, AMReX divides the domain
into smaller "Boxes" and distributes these Boxes across MPI processes. Each MPI
process then iterates through each Box that it "owns," applying the same
operator to each Box and exchanging ghost cells between Boxes as necessary.
Within a Box, `lo` and `hi` represent the lower and upper coordinates of the
Box, not the entire domain.

Traditionally, iterating over distributed data sets contained in Boxes has been
the task of flat MPI code. However, emerging HPC architectures require more
explicit on-node parallelism, for example using threads via OpenMP, in order to
use the machine resources effectively. There are several possible ways to
express on-node parallelism in AMReX with OpenMP:

- Thread over Boxes, with each entire Box being assigned to a thread. This is
straightforward to implement and is noninvasive to the existing code base.
However it is highly susceptible to load imbalance, e.g., if an MPI process
owns 5 boxes but has 8 threads, 3 of those threads will be idle. This kind of
imbalance occurs frequently in AMR simulations, so we did not pursue this
option.

- Thread within each Box. Rather than assign an entire Box to a thread, we
stripe the data within each Box among different threads. This approach is less
susceptible to load imbalance than option #1. However, it requires spawning and
collapsing teams of threads each time an MPI process iterates over a new Box,
which can lead to significant overhead. For example, updating the 3-D heat
equation at a new time step might look like:

```
!$omp parallel private (i, j, k)

!$omp do collapse(2)
do k = lo(3), hi(3)
  do j = lo(2), hi(2)
    do i = lo(1), hi(1)
      f_x(i, j, k) = (p_old(i-1, j, k) - p_old(i, j, k)) * dx_inv
    end do
  end do
end do

!$omp do collapse(2)
do k = lo(3), hi(3)
  do j = lo(2), hi(2)
    do i = lo(1), hi(1)
      f_y(i, j, k) = (p_old(i, j-1, k) - p_old(i, j, k)) * dx_inv
    end do
  end do
end do

!$omp do collapse(2)
do k = lo(3), hi(3)
  do j = lo(2), hi(2)
    do i = lo(1), hi(1)
      f_z(i, j, k) = (p_old(i, j, k-1) - p_old(i, j, k)) * dx_inv
    end do
  end do
end do

!$omp end parallel
```

- Tile the iteration space over all Boxes owned by a process. Rather than a
group of threads operating on each Box, divide each Box owned by a given MPI
process into smaller tiles, and assign a complete tile to a single thread. For
example, if an MPI process owns 2 Boxes, each of size 643, and if we use tiles
which are of size 64x4x4, then each Box yields 256 tiles, and that process will
operate on a pool of 256x2 = 512 tiles in total. If each MPI process can spawn
16 threads, then each thread works on 32 tiles.

Code featuring loop tiling in AMReX would look something like the following:

```
!$omp parallel private(i,mfi,tilebox,tlo,thi,pp_old,pp_new,lo,hi)

call mfiter_build(mfi, phi_old, tiling= .true., tilesize= tsize)

do while(next_tile(mfi,i))

   tilebox = get_tilebox(mfi)
   tlo = lwb(tilebox)
   thi = upb(tilebox)

   pp_old => dataptr(phi_old,i)
   pp_new => dataptr(phi_new,i)
   lo = lwb(get_box(phi_old,i))
   hi = upb(get_box(phi_old,i))

   call advance_phi(pp_old(:,:,:,1), pp_new(:,:,:,1), &
        ng_p, lo, hi, dx, dt, tlo, thi)

end do
!$omp end parallel


subroutine advance_phi(phi_old, phi_new, ng_p, glo, ghi, dx, dt, tlo, thi)

! variable declarations ...

  do k = tlo(3), thi(3)
    do j = tlo(2), thi(2)
      do i = tlo(1), thi(1)
        phi_new(i, j, k) = phi_old(i, j, k) + dt * (flux_x + flux_y + flux_z)
      end do
    end do
  end do
end subroutine do work
```

We tested outer-loop-level threading as well as tiling in AMReX codes and found
the latter to be significantly faster, especially for large numbers of threads.
As an example, we implemented the threading strategies discussed in options #2
and #3 above in a simple 3-D heat equation solver with explicit time-stepping
in AMReX. The domain was a 1283 grid spanned by a single Box (for flat MPI
codes we would generally use smaller and more numerous Boxes, typically of size
323). We then ran the heat solver for 1000 time steps using a single MPI task
with 16 threads spanning one 16-core Haswell CPU on a dual-socket system node.

Before analyzing the code with VTune, we executed both threaded codes 5 times
consecutively and measured the wall time for each run. (OS noise and other
"ambient" factors in a computational system may cause a single run of a
threaded code may run abnormally quickly or slowly; a statistical sample of run
times is therefore crucial for obtaining a reliable timing measurement.) The
timings reported were as follows:

| Iteration # | Outer-loop-level threading | Loop tiling |
|:-----------:|:--------------------------:|:-----------:|
|      1      |           3.76             |    0.58     |
|      2      |           3.87             |    0.69     |
|      3      |           3.86             |    0.70     |
|      4      |           3.86             |    0.69     |
|      5      |           3.84             |    0.68     |

We see that in all 5 iterations, the tiled code ran about 5.6x faster than the
outer-loop-level threaded code. Now we turn to VTune to find out exactly what
lead to such an enormous speedup. We will focus first on the outer-loop-level
threaded code, and next on the tiled code.

### Outer-loop-level threading: General exploration

We first turn to the "general exploration" analysis, which provides a
high-level overview of the code's characteristics (open the image in a new
window to view in full size):

![No tiling: VTune General Exploration Summary][no-tiling-vtune-ge-summary]
[no-tiling-vtune-ge-summary]: no-tiling-ge-summary.png "No tiling: VTune General Exploration Summary"

We see that the code is almost entirely (93%) back-end bound. This is typical
for HPC codes using compiled languages (as opposed to JIT codes). Within the
"back end", we see further that 78% of the code instructions are memory bound,
and only 15% core bound. This is typical of stencil-based codes. Since memory
access seems to be the bottleneck in this code, we can run a "memory access"
analysis within VTune to find more information about how exactly the code
accesses data in memory.

### Outer-loop-level threading: Memory access

Below is shown the summary page from the memory access analysis in VTune:

![No tiling: VTune Memory Access Summary][no-tiling-vtune-ma-summary]
[no-tiling-vtune-ma-summary]: no-tiling-ge-summary.png "No tiling: VTune Memory Access Summary"

We see first that the "elapsed time" according to this analysis is
significantly higher than both the raw wall clock time we measured ourselves
(see the above table) and the wall time according to the general exploration
analysis. It is likely that VTune's detailed sampling and tracing interfere
with its wall clock measurements. This is not a problem for us since we are
much more interested in VTune's sampling and tracing data than in its wall
clock measurements.

The summary page shows that a large fraction of memory loads come all the way
from DRAM, as opposed to cache. The latency to access DRAM is slow, so we want
to minimize this as much as possible. In particular, the bottom-up view (see
figure below) shows that in the `advance_3d` routine (which comprises the bulk
of the code's run time), every micro-operation instruction stalls for an
average of 144 CPU cycles while it waits for the memory controller to fetch
data from DRAM. This highlights the fundamental flaw of outer-loop-level
threading in our heat equation solver: it makes extremely poor use of local
caches; only about 1/3 of the code is bound by latency to access L1 (private)
or L3 (shared) cache, while most of the remainder is spent waiting for DRAM
accesses.

The reason that DRAM latency is the fundamental bottleneck of this code is
because the 3-D heat stencil loads data from non-contiguous locations in
memory, i.e., it reads elements `i` and `i-1`, as well as `j` and `j-1` and `k`
and `k-1`. When we collapse the loop iteration space, each thread will load
element `i` and several of its neighbors (e.g., `i-2`, `i-1`, `i+1`, `i+2`) in
a single cache line, since the data are contiguous along the x-direction.
However, the `j` and `j-1` and `k` and `k-1` elements will likely not be
located in the same cache line as well (unless the total problem domain is
extremely small). So every evaluation of elements `j-1` and `k-1` requires
loading a new cache line from DRAM. This leads to extremely large DRAM
bandwidth and, more importantly, the huge latency we see in the memory access
analysis. We can see the huge bandwidth usage in the "bottom-up" view:

![No tiling: VTune Memory Access Bottom-Up][no-tiling-vtune-ma-bottom-up]
[no-tiling-vtune-ma-bottom-up]: no-tiling-ma-bottom-up.png "No tiling: VTune Memory Access Bottom-Up"

The memory bandwidth throughout the code execution is about 56 GB/sec, which is
nearly the limit for the entire socket. This is due to the frequent cache line
loads to access non-contiguous data elements for the heat stencil evaluation,
as discussed earlier.

We now turn to the tiled code to explore how it solves these problems.

### Tiling: General exploration

Shown below is the summary page in the general exploration analysis of the
tiled code:

![With tiling: VTune General Exploration Summary][with-tiling-vtune-ge-summary]
[with-tiling-vtune-ge-summary]: with-tiling-ge-summary.png "With tiling: VTune General Exploration Summary"

As with the outer-loop-level threaded code, the tiled code is still primarily
back-end bound, although this component is reduced from 93% to 76%.
Interesting, within the "back end," whereas the previous code was almost
entirely memory bound, the tiled code is split between being core-bound and
memory-bound. This indicates that we have relieved much of the pressure that
the previous code placed on the memory controller, presumably by requesting
fewer accesses to DRAM. We turn to the memory access analysis for more details.

### Tiling: Memory access

The summary page of the memory access analysis of the tiled code is as follows:

![With tiling: VTune Memory Access Summary][with-tiling-vtune-ma-summary]
[with-tiling-vtune-ma-summary]: with-tiling-ma-summary.png "With tiling: VTune Memory Access Summary"

From this analysis we see that the the L1 cache usage is roughly the same as
with the previous code. The L3 cache behavior is somewhat improved. However,
the crucial difference is that the number of DRAM accesses has been reduced by
a factor of almost 4. This in turn has reduced the average latency per
instruction in the `advance_3d` routine from 144 cycles in the previous example
to just 29 in the tiled code.

The bottom-up view sheds additional light on the memory access patterns of the
tiled code:

![With tiling: VTune Memory Access Bottom-Up][with-tiling-vtune-ma-bottom-up]
[with-tiling-vtune-ma-bottom-up]: with-tiling-ma-bottom-up.png "With tiling: VTune Memory Access Bottom-Up"

As expected, the DRAM bandwidth is much lower, averaging around 10 GB/sec. This
is because far fewer cache lines need to be loaded to evaluate each stencil in
the tiled iteration space.

We see now that the overwhelming benefit of loop tiling over outer-loop-level
threading is the reduction in memory bandwidth and latency by reducing accesses
to DRAM. The reason for this reduction is that each tile of the iteration space
contains nearby neighbors in all 3 directions from point `(i, j, k)`. So each
load of elements `j-1` and `k-1` is likely now to come from a layer of cache,
rather than all the way from DRAM.

We emphasize that the tiling technique is most useful for horizontal (stencil)
calculations, in that it gathers non-contiguous data elements into cache. For
vertical (point-wise) calculations which depend only on point `(i, j, k)` and
not on any of its neighbors, tiling has little effect on performance. For these
types of calculations, vectorization is a much more fruitful endeavor.

## Vectorization of compressible gasdynamics algorithms

AMReX-powered codes spend the majority of their execution time applying a
series of operators to a hierarchy of structured Cartesian grids. Such
operators generally take one of two forms: "horizontal" operations on several
neighboring grid points to produce a result at a single point (e.g., stencils);
and "vertical" or "point-wise" operators, which require only data at a single
grid point in order to produce a new result at that point (e.g., reaction
networks, equations of state, etc.). As we have seen above, horizontal
operators benefit significantly from loop tiling, because it gathers
neighboring points of grids (which are non-contiguous in memory) into cache,
requiring fewer DRAM accesses during each grid point update. However, vertical
operators see less benefit from loop tiling, because they do not require access
to data at neighboring points.

The performance of point-wise operators is of critical importance in
compressible gasdynamics codes such as
[Nyx](https://amrex-astro.github.io/Nyx/) and
[Castro](https://github.com/AMReX-Astro/Castro). The reason for this is that
these two codes use computationally expensive advection schemes (based on the
piecewise-parabolic method, or "PPM"), and Nyx also uses an exact Riemann
solver, which far more expensive than the approximate Riemann solvers used in
many other gasdynamics codes. In fact, the PPM and Riemann solver are the two
most expensive kernels in the entire Nyx code; we can see this in VTune using
its "general exploration" feature, shown below:

![Nyx without vectorization: VTune General Exploration Bottom-Up][nyx-novec-ge-bottom-up]
[nyx-novec-ge-bottom-up]: Nyx-VTune-general-exploration-bottom-up-novec.png "Nyx without vectorization: VTune General Exploration Bottom-Up"

We see in the "bottom-up" view that the routines `analriem` and `riemannus`
(both components of the analytic Riemann solver), and `ppm_type1` are the
hottest kernels in the code, and together comprise 1/3 of the total
instructions retired, as well as more than 1/3 of the entire code run time. All
of these routines contain point-wise operations on each `(x, y, z)` grid point
on the problem domain. We see that the routine `analriem` is heavily L1
cache-bound, makes frequent use of divider instructions, and also rarely
exploits more than 1 of the 8 available execution ports on the Haswell
processor. (A description of the different execution ports is provided in
Chapter 2 of the [Intel 64 and IA-32 Architectures Optimization Reference
Manual](https://www-ssl.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf).
See also [this Intel page](https://software.intel.com/en-us/node/544476) for a
description of the functions mapped to each port.) All of these characteristics
indicate that the Riemann solver is making poor use of vector instructions, and
also that its performance would increase significantly via loop vectorization.

Now that we have confirmed that a lack of vectorization is the cause of this
code bottleneck, we can turn to the code itself to see what options are
available to remedy this problem. The pseudo-code for the Riemann solver in Nyx
takes the following form:

```
subroutine riemannus()

  ! declare lots of scalar temporary variables to save temporary values for
  ! each complete iteration of the i loop

  do k = klo, khi
    do j = jlo, jhi
      do i = ilo, ihi

        ! calculate left and right gas velocities

        ! lots of sanity checks for gas pressure and internal energy

        ! lots of if/then/else's to make sure we use sane values

        call analriem(gas_velocities, misc_gas_data) ! returns new pressure and velocity from analytic Riemann solver

        ! lots of post-processing if/then/else's

        ! calculate shock velocity

        ! calculate fluxes

      end do
    end do
  end do

end subroutine riemannus
```

This loop is hundreds of line long, and contains a great deal of branching due
to if/then/else's which are used to ensure the Riemann solver uses physically
consistent values. The `analriem` routine is also complicated, although it is
smaller than `riemannus`:

```
subroutine analriem(gas_velocities, misc_gas_data)

  ! set up lots of auxiliary variables derived from gas velocity, pressure, etc.

  do i = 1, 3,
    ! iterate on the new values of pressure and density 3 times to ensure they've converged
  end do

end subroutine analriem
```

As a result of this code complexity, compilers are unable to vectorize any
layer of the outer `(i, j, k)` loop in `riemannus`, even though the Riemann
solver does not require values from any grid points other than `(i, j, k)`
itself. The entire Riemann solver, then, is executed serially, leading to a
severe bottleneck in the overall execution of the Nyx code. Additionally, even
if a compiler was smart enough to generate vectorized versions of this loop,
the heavy use of if/then/else inside the loop would inhibit performance due to
a high degree of branch mispredicts. Furthermore, many compilers are extremely
cautious or unwilling entirely to vectorize function calls inside loops, such
as the call to `analriem` inside the loop in `riemannus`.

One way to vectorize the outer loop of the Riemann solver is to use the `omp
simd` directive [introduced in OpenMP
4.0](https://software.intel.com/en-us/articles/enabling-simd-in-program-using-openmp40).
However, this approach generates vector instructions only if we compile the
code with OpenMP support; if instead we use pure MPI or another framework to
express intra-node parallelism, the compiler ignores the `omp` directives and
the generated instructions will remain serial. A more portable solution is to
write the loops in such a way that compilers' static analysis can easily detect
that the loops have no dependencies and are vectorizable. In the case of this
Riemann solver, we can combine the introduction of some temporary arrays with
loop fissioning techniques to generate a series of much simpler loops. We
caution that, although we have had great success with these techniques, they
can result is more obfuscated code which, although easy for a compiler to
analyze, may be more difficult for a (human) developer to interpret.

The pseudo-code for the vectorized version of the Riemann solver looks like the
following:

```
subroutine riemannus()

  ! replace scalar temporary variables which are overwritten after each
  ! iteration of the i loop with arrays which save temporary values for all
  ! values of (i, j, k)

  do k = klo, khi
    do j = jlo, jhi
      do i = ilo, ihi

        ! calculate left and right gas velocities

      end do
    end do
  end do

  do k = klo, khi
    do j = jlo, jhi
      do i = ilo, ihi

        ! lots of sanity checks for gas pressure and internal energy

        ! lots of if/then/else's to make sure we use sane values
      end do
    end do
  end do

  call analriem(gas_velocities, misc_gas_data) ! returns new pressure and velocity from analytic Riemann solver

  do k = klo, khi
    do j = jlo, jhi
      do i = ilo, ihi

        ! lots of post-processing if/then/else's

      end do
    end do
  end do

  do k = klo, khi
    do j = jlo, jhi
      do i = ilo, ihi

        ! calculate shock velocity

        ! calculate fluxes

      end do
    end do
  end do

end subroutine riemannus
```

We have replaced the scalar temporary variables in `riemannus` with arrays of
temporaries which endure throughout the multiple `(i, j, k)` loops in
`riemannus`. This increases the memory footprint of the algorithm, but also
yields shorter loops which are easier for compilers to vectorize. We note that,
if the compiler generates purely scalar instructions for these refactored
loops, the overall execution time will be the same as the original loop,
although the memory footprint will still be larger. However, since we have had
success with several different compilers vectorizing these loops, the
performance of `riemannus` has increased significantly. We note a few features
of these new, shorter loops:

Some loops still do not easily vectorize, or if they do, their vector
performance is not much faster than the original scalar code. These loops are
chiefly the ones which contain all the if/else logic. Happily, these are not
particularly expensive loops, and so it does not hurt performance a great deal
if they remain scalar. The loop which we do want to vectorize is `analriem`,
which is no longer called inside an `(i,j,k)` loop, but rather has absorbed the
loop inside itself:

```
subroutine analriem(gas_velocities, misc_gas_data)

  do k = klo, khi
    do j = jlo, jhi
      do i = ilo, ihi

        ! set up lots of auxiliary variables derived from gas velocity, pressure, etc.

        do ii = 1, 3,
          ! iterate on the new values of pressure and density 3 times to ensure they've converged
        end do

      end do
    end do
  end do

end subroutine analriem
```

In this form, compilers are much more willing to vectorize the analytic Riemann
solver (and unroll the small innermost loop as well), since it is no longer
called inside an outer `(i, j, k)` loop.

We applied similar loop fissioning techniques to the PPM advection algorithm.
The original pseudo-code had the following form:

```
subroutine ppm_type1()

  ! x-direction

  do k = klo, khi
    do j = jlo, jhi
      do i = ilo, ihi

        ! lots of if/else to calculate temporary variables

        ! use temporary value to calculate characteristic wave speeds

      end do
    end do
  end do

  ! similar triply-nested loops for y- and z-directions

end subroutine ppm_type1
```

After replacing several temporary scalar variables with temporary arrays, we
were able to fission these loops as follows:

```
subroutine ppm_type1()

  ! x-direction

  do k = klo, khi
    do j = jlo, jhi
      do i = ilo, ihi

        ! lots of if/else to calculate temporary variables

      end do
    end do
  end do

  do k = klo, khi
    do j = jlo, jhi
      do i = ilo, ihi

        ! use temporary value to calculate characteristic wave speeds
      end do
    end do
  end do

  ! similar triply-nested loops for y- and z-directions

end subroutine ppm_type1
```

The first of these new, shorter loops absorbs all of the myriad boolean logic
encountered in the PPM algorithm, and although several compilers do vectorize
it, its performance suffers from masking and branch mispredicts. The second
short loop, however, has no logic, and it generates highly vectorizable code.

The results of these loop fissioning techniques is a significant increase in
performance of the entire code. We can see the effects in both Intel VTune and
in Intel Advisor. Starting with VTune, we see that the memory-bound portions of
the code (and especially the L1 cache-bound portions) have been reduced by
almost 10%. In addition, the code has become almost 10% more core-bound, which
indicates that these changes have moved the Nyx code higher toward the roofline
performance of the Haswell architecture on Cori Phase 1.

Turning to the "bottom-up" view in VTune, we see that, whereas the original
Riemann solver was 62% L1 cache-bound, the new solver is only 22% bound. Also,
the number of cycles with multiple ports used has increased significantly in
the vector code, indicating that the vector instructions are making more
efficient use of all of the execution ports available on Haswell.

![Nyx with vectorization: VTune General Exploration Bottom-Up][nyx-vec-ge-bottom-up]
[nyx-vec-ge-bottom-up]: Nyx-VTune-general-exploration-bottom-up-vec.png "Nyx with vectorization: VTune General Exploration Bottom-Up"

Intel Advisor shows similar results to the VTune data. The vector efficiency of
many of the newly fissioned loops in the PPM and Riemann solver routines is
nearly optimal, although a few do suffer (not surprisingly, the ones which
absorbed all of the complex boolean logic).

Because the Nyx and Castro codes use nearly identical advection schemes and
Riemann solvers, porting these changes to Castro will be straightforward, and
should yield a similar performance boost. We anticipate that the performance of
the vectorized version of this code will increase even further on Knights
Landing since it has even wider vector units than Haswell.

## Vectorization of ODE integration in each cell

When simulating problems with radiation in Nyx, the most expensive kernel in
the code is the one which integrates an ordinary differential equation (ODE) in
each cell, which computes the effects of the radiation over the course of the
time step being taken. This ODE is sensitive to the state of the cell -
density, temperature, etc. - and can exhibit sharp, nearly discontinuous
behavior when crossing certain physical thresholds. Consequently, one must
integrate the ODE with an explicit method using many narrow time steps to
resolve the sudden changes, or with an implicit method, which can take larger
time steps. Nyx uses the latter approach, specifically using the
[CVODE](https://computation.llnl.gov/projects/sundials/cvode) solver from the
[SUNDIALS](https://computation.llnl.gov/projects/sundials) software suite.

This integration is expensive to compute because it requires several
evaluations of the right-hand side (RHS) of the ODE; each evaluation of the RHS
requires solving a non-linear equation via Newton-Raphson; and each
Newton-Raphson iteration requires evaluating multiple transcendental functions,
which are highly latency-bound operations. Each of these ODE integrations is
done per cell, and in its original form in Nyx, this was an entirely scalar
evaluation, leading to poor performance, especially compared with older Xeon
processors which exhibit shorter latency for these types of function
evaluations.

To improve the performance of this kernel, we rewrote this integration step
such that CVODE treated multiple cells as part of a single system of ODEs.
Typically implicit solvers such as CVODE require computing a Jacobian matrix of
the ODE system during the integration; however, because these cells are in this
case independent, the Jacobian matrix is diagonal. This allows us to use a
diagonal solver for the Jacobian matrix, resulting in far fewer RHS evaluations
than would be required if the Jacobian matrix was full. Finally, by grouping
cells together as a single system, we rewrote the RHS evaluation as a SIMD
function, computing the RHS for all cells in the system simultaneously.

SIMD evaluation of multiple Newton-Raphsons is challenging due to the differing
number of iterations which are required to converge in the RHS evaluation in
each cell. To address this, we require that, within a group of cells, the
Newton-Raphson iterations for their RHSs continues until all cells achieve an
error which is within the specified tolerance. This generally results in a few
cells within the group iterating more than the minimum number of times required
for convergence, which they wait for the slowest-converging cell to converge.
However, while they may perform some unnecessary work, the overall speedup due
to SIMD evaluation of the transcendental functions more than compensates for
the extra work, and the result is significantly faster code, on both Xeon and
Xeon Phi.

## AMReX code performance on Intel Xeon Phi

In the figure below is a high-level summary of the performance improvement of
vectorization and tiling on Nyx, a hybrid compressible gasdynamics/N-body code
for cosmological simulations, which is based on the AMReX framework. The code
was run on a pre-production version of the Intel Xeon Phi CPU 7210 ("Knights
Landing"), which has 64 cores running at 1.30 GHz. The node was configured in
"quadrant" mode (a single NUMA domain for the entire socket) and with the
high-bandwidth memory in "cache" mode. The problem was a 128^3 domain spanned
by a single 128^3 Box. We used a single MPI process and strong scaled from 1 to
64 OpenMP threads, using a single hardware thread per core (our codes see
relatively little benefit from using multiple threads per core on Xeon Phi). We
see that when we use all 64 cores, Nyx runs nearly an order of magnitude faster
now than it did prior to the start of the NESAP program.

![Nyx performance before and after NESAP][nyx-before-after]
[nyx-before-after]: ResizedImage600370-Nyx-LyA-OpenMP-strong-scaling-before-vs-after-NESAP.png "Nyx performance before and after NESAP"

## Tuning geometric multigrid solvers

One of the major challenges in applied mathematics for HPC is the [scalability
of linear
solvers](http://crd.lbl.gov/departments/applied-mathematics/scalable-solvers-group/).
This challenge affects many AMReX codes, since they encounter a variety of
elliptic problems which must be solved at each time step. AMReX itself provides
two elliptic solvers (one in C++, the other in Fortran), both of which use
geometric multigrid methods.

In order to evaluate the performance of various algorithmic choices made in the
two AMReX multigrid solvers, we recently ported the emergent geometric
multigrid benchmark code [HPGMG](https://hpgmg.org/) to AMReX. HPGMG provides
compile-time parameters for choosing many different algorithms for the
different steps of the multigrid method, including

- cycle types (pure F-cycles, pure V-cycles, or one F-cycle followed by V-cycles)
- smoothers (Gauss-Seidel red-black, Chebyshev, Jacobi, etc.)
- successive over-relaxation (SOR) parameters
- Helmholtz/Poisson operator discretizations (7-point and 27-point cell-centered; 2nd-order and 4th-order finite-volume; constant- or variable-coefficient)
- prolongation operators (piecewise-constant, piecewise-linear, piecewise-parabolic, 2nd-order and 4th-order finite-volume)
- bottom solvers (conjugate-gradient, BiCGSTAB, etc.)
- box agglomeration strategies during the restriction phase

By comparing these different algorithms with those in the AMReX multigrid
solvers, we are able to see which are the most performant for the various types
of elliptic problems encountered in AMReX codes. The results of this study are
forthcoming.

## In situ and in-transit data analysis

Although the majority of application readiness efforts for manycore
architectures are focused on floating-point performance, node-level
parallelism, and tireless pursuit of the
[roofline](http://crd.lbl.gov/departments/computer-science/PAR/research/roofline/),
data-centric challenges in HPC are emerging as well. In particular, simulations
with AMReX codes can generate terabytes to petabytes of data in a single run,
rendering parameter studies unfeasible. One may address this problem in several
ways:

1. Assume that the generation of huge data sets is unavoidable. The problem
then becomes one first of storage, and second of making sense of the data after
the simulation is complete. This is the charge of [data
analytics](http://www.nersc.gov/users/data-analytics/).

2. Preempt the generation of huge data sets by performing the data analysis
(and hopefully reduction) while the simulation is running.

We have focused on the 2nd approach. This option is sensible if one knows
beforehand which features of the simulation data set are "interesting." Such a
luxury is not always the case; exploratory calculations are a critical
component of simulations, and in those cases this option is not useful.
However, many data stewardship problems arise during parameter studies, where
the same simulation is run with varying sets of parameters.

This type of on-the-fly data post-processing can be implemented in two
different ways:

1. In situ: All compute processes pause the simulation and execute
post-processing tasks on the data that they own.

2. In-transit: A separate partition of processes is tasked purely with
performing post-processing, while the remaining processes continue with the
simulation. This is a heterogeneous workflow and is therefore more complicated
to implement, as well as more complicated optimize.

We recently implemented both of these data post-processing techniques in Nyx, a
compressible gasdynamics/N-body code for cosmology based on the AMReX
framework. Specifically, we focused on two sets of commonly used
post-processing algorithms:

1. Finding dark matter "halos" by calculating topological merge trees based on
level sets of the density field.

2. Calculating probability distribution functions (PDFs) and power spectra of
various state variables.

The results of this work are
[published](https://doi.org/10.1186/s40668-016-0017-2) in the open-access
journal Computational Astrophysics and Cosmology.
