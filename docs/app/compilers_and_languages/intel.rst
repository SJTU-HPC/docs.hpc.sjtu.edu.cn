.. _intel:

Intel Compiler
========================

π2.0系统目前支持多种版本的intel编译器。所有编译器套件都包含以下语言：
        
  C、 C++、 Fortran


加载预安装的Intel组件
---------------------

英特尔编译器套件安装在π2.0系统上。
英特尔编译器也可以通过环境模块访问。英特尔的工具还有一些其他特性需要注意。

  1、英特尔提供脚本来设置环境。环境模块允许加载、卸载和切换环境。英特尔的脚本不支持同时加载两种不同版本的环境模块。
  2、加载intel环境将加载与编译器一起分发的MKL环境。英特尔的命名方案没有体现出MKL的版本，单独的MKL模块和英特尔环境提供的MKL包的区别是单独的MKL可以和其他编译器一起使用。

有关其软件的信息，请参阅intel官网，详细阅读。

+-----------------+--------------------------+--------------------------+
| 版本            | 加载方式                 | 组件说明                 |
+=================+==========================+==========================+
| intel-18.0.4    | module load              | Intel编译器              |
|                 | intel/19.0.4             |                          |
+-----------------+--------------------------+--------------------------+
| intel-19.0.4    | module load              | Intel编译器              |
|                 | intel/19.0.4             |                          |
+-----------------+--------------------------+--------------------------+
| intel-19.0.5    | module load              | Intel 编译器             | 
|                 | intrl/19.0.5             |                          |
|                 |                          |                          |
+-----------------+--------------------------+--------------------------+
| intel-19.1.1    | module load              | Intel编译器              |
|                 | intel/19.1.1             |                          |
+-----------------+--------------------------+--------------------------+
| intel-mkl-2019. | module load              | Intel MKL库              |
| 3               | intel-mkl/2019.3.199     |                          |
+-----------------+--------------------------+--------------------------+
| intel-mkl-2019. | module load              | Intel MKL库              |
| 4               | intel-mkl/2019.4.243     |                          |
+-----------------+--------------------------+--------------------------+
| intel-mkl-2019. | module load              | Intel MKL库              |
| 5               | intel-mkl/2019.5.281     |                          |
+-----------------+--------------------------+--------------------------+
| intel-mkl-2020. | module load              | Intel MKL库              |
| 1               | intel-mkl/2020.1.217     |                          |
+-----------------+--------------------------+--------------------------+
| intel-mpi-2019. | module load              | Intel MPI库，由gcc编译   |
| 4.243/gcc-9.2.0 | intel-mpi/2019.4.243     |                          |
+-----------------+--------------------------+--------------------------+
| intel-mpi-2019. | module load              | Intel MPI库，由gcc编译   |
| 6.154/gcc-9.2.0 | intel-mpi/2019.6.154     |                          |
+-----------------+--------------------------+--------------------------+
| intel-parallel- | module load              | Intel全家桶19.4          |
| studio/cluster. | intel-parallel-studio/cl |                          |
| 2019.4          | uster.2019.4             |                          |
+-----------------+--------------------------+--------------------------+
| intel-parallel- | module load              | Intel全家桶19.5          |
| studio/cluster. | intel-parallel-studio/cl |                          |
| 2019.5          | uster.2019.5             |                          |
+-----------------+--------------------------+--------------------------+
| intel-parallel- | module load              | Intel全家桶20.1（默认）  |
| studio/cluster. | intel-parallel-studio/cl |                          |
| 2020.1          | uster.2020.1             |                          |
+-----------------+--------------------------+--------------------------+


优化
-------------

为了从您的软件中获得最佳性能，您可以在编译期间添加优化编译选项。请注意，某些项目已经在构建系统中进行了一些优化。这些可能不是硬件的最佳优化。请注意，Argon 系统具有多种类型的硬件，因此在做出优化决策时需要将其考虑在内。intel编译器文档中包含有详细的编译选项，可以有非常细粒度的控制，-O带有级别说明符的标志可以打开优化功能。级别通常为 -O1、 -O2和 -O3，并且具有越来越高优化等级。还有一个 -O0级别可以在构建中的某些点关闭优化。使用 -Ofast 通常会在运行时对代码的性能产生重大影响。请注意，可能会过度优化，这可能会导致更高优化级别的性能变慢或不稳定。在某些情况下，更高的优化级别也可能导致无法编译代码。

SIMD（单指令多数据）
-------------------------

现代 CPU 处理器能够从称为 SIMD（单指令多数据）的代码生成矢量指令。运行结果代码的处理器必须具有编译期间所针对的物理 SIMD 单元功能。如果不是这种情况，则在其功能集中没有该 SIMD 单元的处理器上执行 SIMD 指令时代码将失败。这意味着可以针对特定硬件获得最高级别的优化，但要权衡代码不能跨架构移植。 SIMD 版本迭代过程：
     
.. code::

   1、SSE
   2、SSE2
   3、SSE3
   4、SSSE3
   5、SSE4.1
   6、SSE4.2
   7、AVX
   8、AVX2
   9、AVX512

这些是向后兼容的，因此使用 AVX 指令编译的代码将在包含 AVX2 单元的机器上运行。通常，编译代码以使用 SIMD 扩展的最简单方法是使用打开硬件特定优化的编译器标志。这有时会在软件项目的构建系统中指定，这将有效地优化构建时检测到的特定硬件的代码。对于 Intel 编译器，该选项是 -xHost，对于 GNU 编译器，它是 -march=native. 如果您将在其上运行代码的所有机器都相同，那么这样做很好。但是，在像 Argon 这样具有多个 CPU 架构的系统上，不鼓励使用这些标志。
有一些侧策略可以解决这个问题：
1、如果您使用英特尔编译器编译代码，那么您可以构建一个多调度二进制文件，该二进制文件将为多个 SIMD 单元构建代码，然后在运行时选择正确的一个。例如，要使用英特尔编译器编译多调度二进制文件以用于具有 AVX2 单元的系统以及仅具有 AVX 单元的系统，您可以使用以下标志。

.. code::

   -axCORE-AVX2,AVX

2、编译选项
如果您的代码需要在一组机器上运行，其中一些使用 AVX2，而一些仅使用 AVX，则为 AVX 目标编译代码。这可以通过-mavx优化标志来完成 。这对 GNU 编译器特别有用。
3、维护代码的多个编译版本。
在这种情况下，编译一个带有 AVX 目标的版本和一个带有 AVX2 目标的版本。然后需要根据主机 CPU 架构手动选择适当的二进制文件。




使用Intel+Intel-mpi编译应用
---------------------------

这里，我们演示如何使用系统中的Intel和Intel-mpi编译MPI代码，所使用的MPI代码可以在\ ``/lustre/share/samples/MPI/mpihello.c``\ 中找到。
在使用intel-mpi的时候，请尽量保持编译器版本与后缀中的编译器版本一致，如intel-mpi-2019.4.243/intel-19.0.4和intel-19.0.4
另外我们建议直接使用Intel全家桶。



加载和编译：

.. code:: bash

   $ module load intel-parallel-studio/cluster.2019.5-intel-19.0.5
   $ mpiicc mpihello.c -o mpihello

提交Intel+Intel-mpi应用
-----------------------

准备一个名为job_impi.slurm的作业脚本

.. code:: bash

   #!/bin/bash

   #SBATCH --job-name=mpihello
   #SBATCH --partition=cpu
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -n 80
   #SBATCH --ntasks-per-node=40

   ulimit -s unlimited
   ulimit -l unlimited

   module load intel-parallel-studio/cluster.2019.5-intel-19.0.5

   export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
   export I_MPI_FABRICS=shm:ofi

   srun ./mpihello

若采用 intel 2018，脚本中 export I_MPI_FABRICS=shm:ofi
这行需改为 export I_MPI_FABRICS=shm:tmi

最后，将作业提交到SLURM

.. code:: bash

   $ sbatch job_impi.slurm

参考资料
--------

-  `intel-parallel-studio <https://software.intel.com/zh-cn/parallel-studio-xe/>`__
-  `参考文档 <https://wiki.uiowa.edu/display/hpcdocs/Compiling+Software/>`__
