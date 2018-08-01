# RAJA

RAJA is a collection of C++ software abstractions, being developed at
Lawrence Livermore National Laboratory (LLNL), that enable
architecture portability for HPC applications. The overarching goals
of RAJA are to:

Make existing (production) applications portable with minimal
disruption Provide a model for new applications so that they are
portable from inception.  RAJA uses standard C++11 -- C++ is the
predominant programming language in which many LLNL codes are
written. RAJA is rooted in a perspective based on substantial
experience working on production mesh-based multiphysics applications
at LLNL. Another goal of RAJA is to enable application developers to
adapt RAJA concepts and specialize them for different code
implementation patterns and C++ usage, since data structures and
algorithms vary widely across applications.

RAJA shares goals and concepts found in other C++ portability
abstraction approaches, such as Kokkos and Thrust. However, it
includes concepts that are absent in other models and which are
fundamental to LLNL codes.

It is important to note that RAJA is very much a work-in-progress. The
community of researchers and application developers at LLNL that are
actively contributing to it and developing new capabilities is
growing. The publicly-released version contains only core pieces of
RAJA as they exist today. While the basic interfaces are fairly
stable, the implementation of the underlying concepts is being
refined. Additional features will appear in future releases.

## Usage

To use RAJA at NERSC please consult the project's documentation for
the latest information.

 * https://github.com/LLNL/RAJA
 * [RAJA User Guide](http://raja.readthedocs.io/en/master/)
