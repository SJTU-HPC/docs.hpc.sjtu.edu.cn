# Vectorization

Modern CPUs have Vector Processing Units (VPUs) that allow the
processor to do the same instruction on multiple
data, [SIMD](https://en.wikipedia.org/wiki/SIMD) per cycle.

| System | microarchitecture | Instruction Set | SIMD width |
|--|--|--|--|
| Cori   | Haswell   | AVX2 | 256 bits |
| Cori | KNL | AVX-512 | 512 bits |

!!! tip
	On KNL with 512 bit vector operations 8 double precision
	operations can be done with each instruction. A code which takes
	advantage of that can potentially achieve an 8x speedup!

## Auto-vectorization

In many cases a compiler is able to transform sequential code into
vector operations automatically - a process known
as
[automatic vectorization](https://en.wikipedia.org/wiki/Automatic_vectorization).

!!! example
	```fortran
	do i = 1, n
		c(i) = a(i) + b(i)
	end do
	```

	Could be transformed by the compiler such that blocks of 4
	elements are processed at a time:

	```fortran
	do i = 1, n, 4
		c(i) = a(i) + b(i)
		c(i+1) = a(i+1) + b(i+1)
		c(i+2) = a(i+2) + b(i+2)
		c(i+3) = a(i+3) + b(i+3)
	end do
	```

## Vectorization requirements

1. The loop trip count must be known at entry to the loop at
   runtime. Statements that can change the trip count dynamically at
   runtime (such as Fortran's `EXIT`, computed `IF`, etc. or C/C++'s
   `break`) must not be present inside the loop.

1. Branching in the loop inhibits vectorization. Thus, C/C++'s
   `switch` statements are not allowed. However, `if` statements are
   allowed as long as they can be implemented as masked
   assignments. The calculation is done for all `if` branches but the
   results is stored only for those elements for which the mask
   evaluates to true.

1. Only the innermost loop is eligible for vectorization. If the
   compiler transforms an outer loop into an inner loop as a result of
   optimization, then the loop may be vectorized.

1. A function call or I/O inside a loop prohibits
   vectorization. Intrinsic math functions such as `cos`, `sin`,
   etc. are allowed because such library functions are usually
   vectorized versions. A loop containing a function that is inlined
   by the compiler can be vectorized because there will be no more
   function call.

1. Data dependencies in the loop could prevent vectorization.

1. Non-contiguous memory access hampers vectorization
   efficiency. Eight consecutive ints or floats, or four consecutive
   doubles, may be loaded directly from memory in a single AVX
   instruction. But if they are not adjacent, they must be loaded
   separately using multiple instructions, which is considerably less
   efficient.

### Data dependency

#### read-after-write

Also known as "flow dependency". Vectorization creates wrong results.

```fortran
do i=2,n
	a(i) = a(i-1) + 1
end do
```

#### write-after-read

Also known as "anti-dependency" and can be vectorized.

```fortran
do i=2,n
	a(i-1) = a(i) + 1
end do
```

#### write-after-write

Also known as "output dependency" and cannot be vectorized.

```fortran
do i=2,n
	a(i-1) = x(i)
	a(i)   = 2.0 * i
end do
```

#### Reduction operations

Reduction operations can be vectorized.

```fortran
s=0.0
do i=1,n
	s = s + a(i) * b(i)
end do
```

## Memory alignment

Data movement instructions are more efficient when operating on data
objects that are aligned.

* [Data structure alignment](https://en.wikipedia.org/wiki/Data_structure_alignment)
* [aligned_alloc (C)](https://en.cppreference.com/w/c/memory/aligned_alloc)
* [std::aligned_alloc (C++)](https://en.cppreference.com/w/cpp/memory/c/aligned_alloc)

!!! note
	While Fortran does not have extensions in the language itself for
	data alignment some compilers provide non-portable directives or
	command line flags: the Cray compiler has the directive `!DIR$
	VECTOR ALIGNED` and the Intel compiler has the compiler flag
	`-align array64byte`.

### Fortran alignment example

The following test code examines the effect of memory alignment in a
simple-minded matrix-matrix multiplication case. We pad the matrices
with extra rows to make them aligned at certain boundaries.

```fortran
      program matmat

      implicit none
      integer :: n = 31
      integer :: itmax = 200000
#ifdef REAL4
      real, allocatable :: a(:,:), b(:,:), c(:,:)
#else
      real*8, allocatable :: a(:,:), b(:,:), c(:,:)
#endif
#ifdef ALIGN16
!dir$ attributes align : 16 :: a,b,c
#elif defined(ALIGN32)
!dir$ attributes align : 32 :: a,b,c
#elif defined(ALIGN64)
!dir$ attributes align : 64 :: a,b,c
#endif
      integer i, j, k, it
      integer :: vl, nr
      integer*8 c1, c2, cr, cm
      real*8 dt

!...  Vector length

#ifdef ALIGN16
      vl = 16 / (storage_size(a) / 8)
#elif defined(ALIGN32)
      vl = 32 / (storage_size(a) / 8)
#elif defined(ALIGN64)
      vl = 64 / (storage_size(a) / 8)
#else
      vl = 1
#endif

      nr = ((n + (vl - 1)) / vl) * vl      ! padded row dimension
      allocate (a(nr,n), b(nr,n), c(nr,n))

!...  Initialization

      do j=1,n
#if defined(ALIGN16) || defined(ALIGN32) || defined(ALIGN64)
!dir$ vector aligned
#endif
         do i=1,nr
            a(i,j) = cos(i * 0.1 + j * 0.2)
            b(i,j) = sin(i * 0.1 + j * 0.2)
            c(i,j) = 0.
         end do
      end do

!...  Main loop

      call system_clock(c1, cr, cm)
      do it=1,itmax
         do j=1,n
            do k=1,n
#if defined(ALIGN16) || defined(ALIGN32) || defined(ALIGN64)
!dir$ vector aligned
#endif
               do i=1,nr
                  c(i,j) = c(i,j) + a(i,k) * b(k,j)
               end do
            end do
         end do
      end do
      call system_clock(c2, cr, cm)

      print *, c(1,1)+c(n,n), dble(c2-c1)/dble(cr)
      deallocate(a, b, c)
      end
```

## AoS vs SoA

[Array of Structures vs Structures of Arrays](https://en.wikipedia.org/wiki/AOS_and_SOA)

A data object can become complex with multiple component elements or
attributes. Programmers often represent a group of such data objects
using an array of Fortran's derived data type or C's struct objects
(i.e., an array of structures or AoS). Although an AoS provides a
natural way to represent such data, memory reference of any component
requires non-unit stride access. Such a situation is illustrated in
the following example code. When the main loop is transformed into a
vector loop, three components of a 'coords' object will be stored into
three separate vector registers, one for each component. With the AoS
data layout, loading into such a register will require stride 3 (or
more) access, reducing efficiency of the vector load.

A better data structure for vectorization is to separate each
component of the objects into its own array, and then form a data
object composed of three arrays (i.e., a structure of arrays or
SoA). When the main loop is vectorized, each component will be loaded
into a separate register but this will be done with unit-stride
access. Therefore, vectorization will be more efficient.

```fortran
      program aossoa

      implicit none
      integer :: n = 1000
      integer :: itmax = 10000000
#ifdef SOA
      type coords
         real, pointer :: x(:), y(:), z(:)
      end type
      type (coords) :: p
#else
      type coords
         real :: x, y, z
      end type
      type (coords), allocatable :: p(:)
#endif
      real, allocatable :: dsquared(:)
      integer i, it
      integer*8 c1, c2, cr, cm
      real*8 dt

!...  Initialization

#ifdef SOA
      allocate(p%x(n), p%y(n), p%z(n), dsquared(n))
      do i=1,n
         p%x(i) = cos(i + 0.1)
         p%y(i) = cos(i + 0.2)
         p%z(i) = cos(i + 0.3)
      end do
#else
      allocate(p(n), dsquared(n))
      do i=1,n
         p(i)%x = cos(i + 0.1)
         p(i)%y = cos(i + 0.2)
         p(i)%z = cos(i + 0.3)
      end do
#endif

!...  Main loop

      call system_clock(c1, cr, cm)
      do it=1,itmax
#ifdef SOA
         do i=1,n
            dsquared(i) = p%x(i)**2 + p%y(i)**2 + p%z(i)**2
         end do
#else
         do i=1,n
            dsquared(i) = p(i)%x**2 + p(i)%y**2 + p(i)%z**2
         end do
#endif
      end do
      call system_clock(c2, cr, cm)

      dt = dble(c2-c1)/dble(cr)
      print *, dsquared(1)+dsquared(n/2)+dsquared(n), dt
#ifdef SOA
      deallocate(p%x, p%y, p%z, dsquared)
#else
      deallocate(p, dsquared)
#endif
      end
```

## Elemental functions

Elemental functions are functions that can be also invoked with an
array actual argument and return array results of the same shape as
the argument array. This convenient feature is quite common in Fortran
as it is widely used in many intrinsic functions.

A function call inside a loop generally inhibits
vectorization. However, if an elemental function is called within a
loop, the loop can be executed in vector mode. In vector mode, the
function is called with multiple data packed in a vector register and
returns packed data.

### Fortran example

```
module fofx
  implicit none
contains
  elemental function f(x)
    real(8) :: f
    real(8), intent(in) :: x
    f = cos(x * x + 1.0_8) / (x * x + 1.0_8)
  end function f
end module fofx

program main
  use fofx
  implicit none
  integer :: n = 1024
  integer :: itmax = 1000000
  real(8), allocatable :: a(:), x(:)
  integer :: i, it
  integer(8) :: c1, c2, cr, cm
  real(8) ::  dt

  allocate (a(n), x(n))

  !... Initialization

  do i=1,n
     x(i) = cos(i * 0.1_8) + 0.2_8
  end do

  !... Main loop

  call system_clock(c1, cr, cm)
  do it=1,itmax
     do i=1,n
        a(i) = f(x(i))
     end do
     x(n) = x(n) + 1.0_8
  end do
  call system_clock(c2, cr, cm)

  dt = real(c2-c1, 8)/real(cr, 8)
  write(*,*) n, a(1)+a(n/2)+a(n), dt

  deallocate(a, x)
end program main
```

## Pointer aliasing

* [restrict keyword (C)](https://en.wikipedia.org/wiki/Restrict)

## OpenMP

The OpenMP standard has the SIMD construct since 4.0 to specify the
execution of a loop in vectorization mode (i.e., SIMD operations).

```C
#pragma omp simd [clause...]
```

```fortran
!$omp simd [clause...]
```

where the optional clause could be:

* `safelen(length)` - maximum length for safe vectorization without
  incurring data dependency
* `aligned(list[:alignment])` - list of the variables are aligned to
  the number of bytes expressed in the optional parameter.
* `reduction(reduction-identifier:list)` - list the variables where a
  reduction operation (i.e., `+` for summation, `min` for minimum, `max` for
  maximum, etc.) result is stored
* `collapse(n)` - how many levels of the nested loops that immediately
  follow the OpenMP directive should be collapsed into a single
  aggregate loop with larger iteration space.

### Memory alignment

```fortran
      do it=1,itmax
         do j=1,n
            do k=1,n
#if   defined(ALIGN16)
!$omp simd aligned(a,b,c:16)
#elif defined(ALIGN32)
!$omp simd aligned(a,b,c:32)
#elif defined(ALIGN64)
!$omp simd aligned(a,b,c:64)
#endif
               do i=1,nr
                  c(i,j) = c(i,j) + a(i,k) * b(k,j)
               end do
            end do
         end do
      end do
```

### Elemental functions

It is also possible  to declare that a function can be vectorized with
OpenMP.

```fortran
     module fofx
       implicit none
     contains
#ifdef ELEMENTAL
!$omp declare simd (f)
#endif
       function f(x)
#ifdef REAL4
         real f, x
#else
         real*8 f, x
#endif
#ifdef REAL4
         f = cos(x * x + 1.e0) / (x * x + 1.e0)
#else
         f = cos(x * x + 1.d0) / (x * x + 1.d0)
#endif
       end function f
     end module fofx
...
```

## Additional resources

* [Vectorization with Intel C++ compiler](https://software.intel.com/sites/default/files/8c/a9/CompilerAutovectorizationGuide.pdf)
* [Cornell Vectorization Workshop](https://cvw.cac.cornell.edu/vector/default)
* [Effective Vectorization with OpenMP 4.5](https://info.ornl.gov/sites/publications/files/Pub69214.pdf)
* `man icpc`
* `man icc`
* `man ifort`
