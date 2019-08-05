# Parallelism

## On-node parallelism

On Cori, it will be necessary to specifically think both about
inter-node parallelism as well as on-node parallelism. The baseline
programming model for Cori is MPI+X where X represents some conscious
level of on-node parallelism, which could also be expressed as MPI or
a shared memory programming model like OpenMP, pthreads etc. For a lot
of codes, running without changes on 72 (or up to 288) MPI tasks per
node could be troublesome. Examples are codes that are MPI latency
sensitive like 3D FFTs and codes that duplicate data structures on MPI
ranks (often MPI codes don't perfectly distribute every data structure
across ranks) which could more quickly exhaust the HBM if running in
pure MPI mode.

### Threading

- Thread libraries per language
    - C: pthreads, OpenMP
    - C++: pthreads, OpenMP, STL threads, TBB
    - Fortran: OpenMP

- Library Overview
    - `pthreads`
        - Usage style: functional C
        - Ease-of-use: low
        - Portability: moderate
            - UNIX-only
        - Features
            - Generally provide best performance due to low-level nature
        - Relevant C/C++ compiler flags
            - `-pthread` (GCC, LLVM, Intel)
    - `C++11 threads`
        - Object-oriented C++ wrappers around OS-defined threading library
        - Ease-of-use: medium
        - Portability: high
            - Any compiler conforming to C++11 standard
        - Features
            - Easiest to use the `pthreads`, more difficult to use the `OpenMP`
            - Lightweight wrapper around `pthreads` on UNIX with similar performance
    - `OpenMP`
        - Usage style: directives (`#pragma omp`)
        - Ease-of-use: high
        - Portability: moderate
            - Compiler-dependent but supported by most major compilers
            - Performance can vary between compiler versions
        - Features
            - Built-in [thread-pool](https://en.wikipedia.org/wiki/Thread_pool)
            - Serial (compiled without OpenMP) and MT (compiled with OpenMP) provided by same block(s) of code
            - SIMD, reductions, GPU offloading (currently limited), tasking
        - Relevant C/C++ compiler flags
            - `-fopenmp` (GCC/Intel)
            - `-fopenmp=libomp` (LLVM)
    - `TBB` (Intel Threading Building Blocks)
        - Task-based parallelism library
        - Ease-of-use
            - C++98: medium
            - C++11: high
        - Portability: highest
            - Any C++ compiler
        - Features
            - Built-in [thread-pool](https://en.wikipedia.org/wiki/Thread_pool)
            - Highly efficient nested and recursive parallelism
            - Constructs for parallel: loops, reductions, function calls, etc.
            - Flow-graphs and pipelines

#### Performance

The best performance occurs when threads are persistent (i.e. not continuously created and deleted, may require
[Thread Pool](https://en.wikipedia.org/wiki/Thread_pool)), at the top of the call-stack, and with minimal synchronization.

##### Thread Pool

Both OpenMP and TBB provide thread-pools by default.

##### Call-Stack

Take the following example for the case of nested loops:

```c++
// loop "A"
for(unsigned i = 0; i < N; ++i)
{
    // loop "B"
    for(unsigned j = 0; j < M; ++j)
    {
        // some work function
        work(i, j);
    }
}
```

Given the choice of using threads to operate on either loop "A" or loop "B", the most speed-up is provided by having the threads operate on a block/iteration from loop "A".

When using the OpenMP library, unless otherwise specified to enable nesting, inserting a `#pragma omp parallel for` before both loops:

```c++
// loop "A"
#pragma omp parallel for
for(unsigned i = 0; i < N; ++i)
{
    // loop "B"
    #pragma omp parallel for
    for(unsigned j = 0; j < M; ++j)
    {
        // some work function
        work(i, j);
    }
}
```
 will result in OpenMP ignoring the directive on loop "B". Enabling nesting with OpenMP is not desired because it can quickly cause an excessive number of threads to be created. In the above example, if `OMP_NUM_THREADS=16` and nesting enabled, 16 threads are created to parallelize loop "A" and then each of those 16 threads will create and additional 16 threads to parallelize loop "B" -- resulting in 256 total threads. Application failures will arise (due to exceeding the max threads for OS) when the work inside loop "B" makes calls to external libraries that also use OpenMP.

 When using the tasking model such as when using TBB or OpenMP tasks, threads are abstracted and the work is bundled into "tasks" -- which can be thought as function calls. Threads do not operate directly within the loop but instead sit idle in a thread-pool until it has been given work to do in the form of a "task". The main thread divides the loop iterations into tasks and adds them to the queue. When this occurs, any idle threads are activated to begin popping tasks out of the queue and executing them. When the thread is finished executing one task, it either proceeds to the next task or, if the queue is empty, goes back to "sleep". The end result is that the tasking model provides better fine-grained parallelism and work-load balancing between the threads.

```c++
// lambda function for executing an iteration of loop "A"
auto do_loop_b = [] (int i)
{
    // lambda function wrapping "work(i, j)" call
    // [&] captures "i"
    auto do_work = [&] (int j) { work(i, j); };

    // loop "B"
    tbb::parallel_for(0, M, do_work);
};

// loop "A"
tbb::parallel_for(0, N, do_loop_b);
```

##### Synchronization

Synchronizing the access of shared memory in multithreading creates a serial bottleneck and poor synchronization implementations
can all but eliminate any potential speed-up. One of the most common synchronization issues involves updating shared containers:

```c++
// a shared container
static std::vector<double> my_container = std::vector<double>(100, 0.0);
// a mutex for synchronization
static std::mutex mtx;

void update_container(int index, double value)
{
    // threads will idle here to acquire the lock
    // this creates a serial bottleneck because only
    // one thread can hold lock at a time
    mtx.lock()
    // update value
    my_container[index] += value;
    // release lock so other threads can acquire lock
    mtx.unlock();
}
```

The optimal solution is to have the thread operate on it's own own container and at the end, have each thread merge the
contents of it's thread-local container into the shared container. Having each thread have it's own container is
accomplished with thread-local memory.

NOTE: thread-local memory in C++ requires trivial constructors so unless one is using plain-old-data (int, double, etc.)
use a pointer and the pointer will be deleted when the thread exits.

```c++
// a shared container
static std::vector<double> my_container = std::vector<double>(100, 0.0);
// a container local to the thread
static thread_local std::vector<double>* tl_container = new std::vector<double>(100, 0.0);
// a mutex for synchronization
static std::mutex mtx;

void update_container(int index, double value)
{
    // container is local to thread so no need for lock
    (*tl_container)[index] += value;
}

// have each thread call this function when it is done
void merge_container()
{
    // block access to shared resource
    mtx.lock()
    // copy the thread-local memory to shared memory
    for(unsigned i = 0; i < tl_container->size(); ++i)
        my_container[i] += (*tl_container)[i];
    // release lock
    mtx.unlock();
}
```

### Vectorization

Modern CPUs have Vector Processing Units (VPUs) that allow the
processor to do the same instruction on multiple data (SIMD) per
cycle.

On KNL, the VPU will be capable of computing the operation on 8 rows
of the vector concurrently. This is equivalent to computing 8
iterations of the loop at a time.

The compilers on Cori want to give you this 8x speedup whenever
possible. However some things commonly found in codes stump the
compiler and prevent it from vectorizing. The following figure shows
examples of code the compiler won't generally choose to vectorize

## Off-node parallelism

### Strong scaling

How the time to solution varies with number of processing elements for
a fixed problem size.

### Weak scaling

How the time to solution varies with number of processing elements for
a fixed problem size _per processor_.
