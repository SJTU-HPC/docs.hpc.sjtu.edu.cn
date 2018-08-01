# Vectorization

Modern CPUs have Vector Processing Units (VPUs) that allow the
processor to do the same instruction on multiple
data, [SIMD](https://en.wikipedia.org/wiki/SIMD) per cycle.

| System | microarchitecture | Instruction Set | SIMD width |
|--|--|--|--|
| Edison | IvyBridge | AVX | 128 bits |
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
