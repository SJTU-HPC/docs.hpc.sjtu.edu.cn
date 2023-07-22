*********
Hyper-MPI
*********

HyperMPI是一款华为自研的高性能通信库，它基于OpenMPI 4.0.3 和Open UCX
1.6.0构建，支持MPI-V3.1标准的并行计算接口。相较于openMPI,它在基于鲲鹏处理器的高性能计算集群上表现更好。

加载预安装的HyperMPI
--------------------

可以用以下命令加载集群中已安装的Hyper-MPI

+------------+--------------------------------------------------+
| 版本       | 组件说明                                         |
+============+==================================================+
| 4.0.3      | 使用9.3.0版本gnu编译套件编译的4.0.3版本的HyperMPI|
+------------+--------------------------------------------------+

::

  module load gcc/9.3.0-gcc-4.8.5
  module load hmpi/4.0.3-gcc-9.3.0


使用HyperMPI编译应用
--------------------

这里，我们演示如何使用系统中的HyperMPI编译MPI代码，使用的MPI代码可以在\ ``/lustre/share/samples/mpi/src/mpihello.c``\ 中找到。

加载和编译：

::

   module load gcc/9.3.0-gcc-4.8.5
   module load hmpi/4.0.3-gcc-9.3.0

   mpicc mpihello.c -o mpihello

提交HyperMPI编译的应用
----------------------

准备一个名为job_hmpi.slurm的作业脚本，用以提交上面编译的mpihello应用。

::

   #!/bin/bash

   #SBATCH --job-name=mpihello
   #SBATCH --partition=arm128c256g
   #SBATCH --output=%j.out
   #SBATCH --error=%j.err
   #SBATCH -n 2

   ulimit -s unlimited
   ulimit -l unlimited

   module load gcc/9.3.0-gcc-4.8.5
   module load hmpi/4.0.3-gcc-9.3.0

   mpirun -np 2 ./mpihello

使用以下命令提交作业脚本：

::

   sbatch job_hmpi.slurm

性能调优
--------

HyperMPI使用了大量华为自研的集合通信算法，默认情况下，HyperMPI会根据硬件配置和通信规模大小选择最优的算法。

但是在某些情况下，自动选择的算法无法达到最优的性能，需要手动进行集合通信算法的选择。例如，我们在测试时发现，在消息大小较大时，MPI_Allreduce默认选择的通信算法性能不是最优，这时我们可以使用以下选项手动选择MPI_Allreduce通信算法得到最优性能：

::

   mpirun -np ${process num} -x UCX_BUILTIN_ALLREDUCE_ALGORITHM=4 ${application}

更多相关内容介绍可以参考\ `HyperMPI通信算法指定 <https://www.hikunpeng.com/document/detail/zh/kunpenghpcs/hypermpi/userg_huaweimpi_0015.html>`__
## 参考资料 `Hyper
MPI简介 <https://support.huawei.com/enterprise/zh/doc/EDOC1100228708/c5d7ef16#ZH-CN_TOPIC_0000001165250320>`__
