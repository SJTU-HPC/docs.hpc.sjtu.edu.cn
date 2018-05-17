# Vectorization

Vectorization is actually another form of on-node parallelism. Modern
CPUs have Vector Processing Units (VPUs) that allow the processor to
do the same instruction on multiple data (SIMD) per cycle. Consider
the figure below. The loop can be written in "vector" notation:

```fortran
do i=1, n
	a(i) = b(i) + c(i)
end do
```

$$
\vec{a} = \vec{b} + \vec{c}
$$

On KNL, the VPU will be capable of computing the operation on 8 rows
of the vector concurrently. This is equivalent to computing 8
iterations of the loop at a time.

The compilers on Cori want to give you this 8x speedup whenever
possible. However some things commonly found in codes stump the
compiler and prevent it from vectorizing. The following figure shows
examples of code the compiler won't generally choose to vectorize

## Loop Dependency

Compilers are unable to give you vector code because the $i^{th}$
iteration of the loop depends on the $(i-1)^{th}$ iteration.  In other
words, it is not possible to compute both these iterations at once.

```fortran
do i = 1, n
	a(i) = a(i-1) + b(i)
end do
```

## Task Forking

An example where the execution of the loop forks based on an if
statement. Depending on the exact situation, the compiler may or may
not vectorize the loop. Compilers typically use heuristics to decide
if they can execute both paths and achieve a speedup over the
sequential code.

```fortran
do i = 1, n
	if (a(i) < x) cycle
	if (a(i) > x) a(i) = a(i) + 1
end do
```
