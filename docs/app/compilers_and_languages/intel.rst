.. _intel:

Intel Compiler
========================

Intel编译套件是由Intel提供的编译器、MPI环境、MKL库等集成套件。本节讲解如何使用集群中预安装的相关组件。

加载预安装的Intel组件
---------------------

可以用以下命令加载集群中已安装的Intel组件:

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

在使用intel-mpi的时候，请尽量保持编译器版本与后缀中的编译器版本一致，如intel-mpi-2019.4.243/intel-19.0.4和intel-19.0.4
另外我们建议直接使用Intel全家桶

使用Intel+Intel-mpi编译应用
---------------------------

这里，我们演示如何使用系统中的Intel和Intel-mpi编译MPI代码，所使用的MPI代码可以在\ ``/lustre/share/samples/MPI/mpihello.c``\ 中找到。

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
